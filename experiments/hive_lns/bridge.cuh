#pragma once
// Frozen integration. The evolvable header receives only immutable owning CPU data.
#include <pthread.h>
#include <atomic>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include "../../cpp/src/mip_heuristics/lns_improvement.hpp"
#include "repair_tools.cuh"

namespace cuopt::mathematical_optimization::mip {
template <typename i_t, typename f_t>
class hive_lns_bridge_t {
 public:
  hive_lns_bridge_t(mip_solver_context_t<i_t, f_t>& context,
                    population_t<i_t, f_t>& population,
                    cuopt::timer_t timer)
    : context_(context), population_(population), timer_(timer)
  {
    auto& pb    = *context_.problem_ptr;
    auto stream = pb.handle_ptr->get_stream();
    auto copy   = [stream](auto& dest, const auto& source) {
      auto values = cuopt::host_copy(source, stream);
      dest.assign(values.begin(), values.end());
    };
    copy(model_.offsets, pb.offsets);
    copy(model_.columns, pb.variables);
    copy(model_.coefficients, pb.coefficients);
    copy(model_.objective, pb.objective_coefficients);
    copy(model_.row_lower, pb.constraint_lower_bounds);
    copy(model_.row_upper, pb.constraint_upper_bounds);
    auto bounds = cuopt::host_copy(pb.variable_bounds, stream);
    auto types  = cuopt::host_copy(pb.variable_types, stream);
    pb.handle_ptr->sync_stream();
    for (size_t j = 0; j < bounds.size(); ++j) {
      model_.lower.push_back(bounds[j].x);
      model_.upper.push_back(bounds[j].y);
      model_.integer.push_back(types[j] == var_t::INTEGER);
    }
    cudaGetDevice(&device_);
    population_.lns_observer = [this](const std::vector<f_t>& x) { offer(x); };
    {
      std::lock_guard<std::recursive_mutex> lock(population_.write_mutex);
      for (auto& member : population_.solutions) {
        if (member.first && member.second.get_feasible())
          offer(member.second.get_host_assignment());
      }
    }
    worker_ = std::thread([this] {
      try {
        if (cudaSetDevice(device_) != cudaSuccess)
          throw std::runtime_error("LNS CUDA device setup failed");
        if (pthread_setname_np(pthread_self(), "cuopt-hive-lns") != 0)
          throw std::runtime_error("LNS thread attribution setup failed");
        // This handle and both repair backends belong exclusively to this worker.
        raft::handle_t repair_handle;
        model_.repair.cpufj = [this, &repair_handle](const auto& request) {
          return cuopt::hive_lns::repair_neighborhood(
            model_,
            request,
            cuopt::hive_lns::repair_backend_t::cpufj,
            [this] { return stopped(); },
            &repair_handle,
            timer_.remaining_time(),
            context_.preempt_heuristic_solver_);
        };
        model_.repair.submip = [this, &repair_handle](const auto& request) {
          return cuopt::hive_lns::repair_neighborhood(
            model_,
            request,
            cuopt::hive_lns::repair_backend_t::submip,
            [this] { return stopped(); },
            &repair_handle,
            timer_.remaining_time(),
            context_.preempt_heuristic_solver_);
        };
        CUOPT_LOG_INFO("HIVE_LNS_STARTED");
        cuopt::hive_lns::run_lns(
          model_,
          [this] { return snapshot(); },
          [this](const auto& x) { submit(x); },
          [this] { return stopped(); },
          context_.base_seed);
      } catch (...) {
        failure_ = std::current_exception();
      }
      CUOPT_LOG_INFO("HIVE_LNS_INPUTS %lu", inputs_);
      CUOPT_LOG_INFO("HIVE_LNS_FINISHED");
    });
  }
  ~hive_lns_bridge_t()
  {
    stop_.store(true);
    if (worker_.joinable()) worker_.join();
    population_.lns_observer = {};
  }
  void finish()
  {
    stop_.store(true);
    if (worker_.joinable()) worker_.join();
    population_.lns_observer = {};
    if (failure_) std::rethrow_exception(failure_);
  }
  const std::vector<f_t>& best_assignment() const { return best_; }

 private:
  bool stopped() const
  {
    return stop_.load() || timer_.check_time_limit() || context_.preempt_heuristic_solver_.load();
  }
  void offer(const std::vector<f_t>& source)
  {
    if (stopped()) return;
    std::vector<double> x(source.begin(), source.end());
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (cache_.size() == 8) cache_.pop_front();
    cache_.push_back(std::move(x));
  }
  cuopt::hive_lns::population_t snapshot()
  {
    cuopt::hive_lns::population_t result;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      result.assign(cache_.begin(), cache_.end());
    }
    result.erase(
      std::remove_if(
        result.begin(), result.end(), [this](const auto& x) { return !model_.feasible(x); }),
      result.end());
    inputs_ += result.size();
    return result;
  }
  void submit(const std::vector<double>& x)
  {
    if (stopped()) return;
    if (!model_.feasible(x)) throw std::runtime_error("Invalid LNS submission");
    const auto cost = model_.cost(x);
    if (cost >= best_cost_) return;
    if (stopped()) return;
    best_.assign(x.begin(), x.end());
    best_cost_ = cost;
    // Same lock ordering as population.add_solution: population, then publication.
    // Keep the population best stable through objective recomputation/publication.
    std::lock_guard<std::recursive_mutex> lock(population_.write_mutex);
    if (stopped() || !population_.is_feasible()) return;
    f_t population_best = std::numeric_limits<f_t>::infinity();
    for (auto& member : population_.solutions) {
      if (member.first && member.second.get_feasible())
        population_best = std::min(population_best, member.second.get_objective());
    }
    context_.solution_publication.publish_if_better(
      context_.problem_ptr, best_, f_t(cost), population_best);
  }
  mip_solver_context_t<i_t, f_t>& context_;
  population_t<i_t, f_t>& population_;
  cuopt::timer_t timer_;
  cuopt::hive_lns::model_t model_;
  std::mutex cache_mutex_;
  std::deque<std::vector<double>> cache_;
  std::atomic<bool> stop_{false};
  std::thread worker_;
  std::exception_ptr failure_;
  std::vector<f_t> best_;
  double best_cost_ = std::numeric_limits<double>::infinity();
  size_t inputs_    = 0;
  int device_       = 0;
};
}  // namespace cuopt::mathematical_optimization::mip
