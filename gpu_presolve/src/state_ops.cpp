/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/state.hpp>

namespace gprs {

double state_t::dead_fraction() const
{
  gprs_assert(false && "not implemented");
  return {};
}

int64_t state_t::find_entry(int32_t, int32_t) const
{
  gprs_assert(false && "not implemented");
  return {};
}

bool state_t::implied_lower_valid(int32_t) const
{
  gprs_assert(false && "not implemented");
  return {};
}

bool state_t::implied_upper_valid(int32_t) const
{
  gprs_assert(false && "not implemented");
  return {};
}

void state_t::set_col_lower(int32_t, double) { gprs_assert(false && "not implemented"); }

void state_t::set_col_upper(int32_t, double) { gprs_assert(false && "not implemented"); }

void state_t::fix_col(int32_t, double) { gprs_assert(false && "not implemented"); }

void state_t::remove_col(int32_t) { gprs_assert(false && "not implemented"); }

void state_t::set_cost(int32_t, double) { gprs_assert(false && "not implemented"); }

void state_t::add_objective_offset(double, double) { gprs_assert(false && "not implemented"); }

void state_t::record_implied_lower(int32_t, double, int32_t)
{
  gprs_assert(false && "not implemented");
}

void state_t::record_implied_upper(int32_t, double, int32_t)
{
  gprs_assert(false && "not implemented");
}

void state_t::delete_row(int32_t) { gprs_assert(false && "not implemented"); }

void state_t::set_row_lower(int32_t, double) { gprs_assert(false && "not implemented"); }

void state_t::set_row_upper(int32_t, double) { gprs_assert(false && "not implemented"); }

void state_t::shift_row_sides(int32_t, double, double) { gprs_assert(false && "not implemented"); }

void state_t::scale_row(int32_t, double, bool) { gprs_assert(false && "not implemented"); }

void state_t::set_coefficient(int64_t, double) { gprs_assert(false && "not implemented"); }

void state_t::delete_entry(int64_t) { gprs_assert(false && "not implemented"); }

void state_t::move_entry(int64_t, int32_t, double) { gprs_assert(false && "not implemented"); }

}  // namespace gprs
