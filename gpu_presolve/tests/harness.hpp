/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Minimal test harness.
//   GPRS_TEST(name) { ... }   defines and registers a test; names are unique per source file.
//   GPRS_CHECK(condition)     records a failure and goes on.
//   GPRS_CHECK_EQ(a, b)       the same for a == b, printing both values (doubles with 17 digits).
//   GPRS_REQUIRE(condition)   records a failure and leaves the test.
// The checks are active in every build; they do not depend on GPRS_ASSERT.
// run_all runs the tests in the order of their names, all of them or those whose name contains
// the first command line argument, and returns 1 when a check failed, else 0.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace gprs {
namespace test {

struct test_t {
  const char* name;
  void (*run)();
};

inline std::vector<test_t>& registry()
{
  static std::vector<test_t> tests;
  return tests;
}

inline int& failure_count()
{
  static int count = 0;
  return count;
}

struct registrar_t {
  registrar_t(const char* name, void (*run)()) { registry().push_back(test_t{name, run}); }
};

inline void fail(const char* file, int line, const std::string& what)
{
  ++failure_count();
  std::fprintf(stderr, "    %s:%d: %s\n", file, line, what.c_str());
}

inline bool check(bool condition, const char* text, const char* file, int line)
{
  if (!condition) { fail(file, line, std::string(text) + " is false"); }
  return condition;
}

template <typename T>
void print_value(std::ostream& out, const T& value)
{
  if constexpr (std::is_enum_v<T>) {
    out << +(std::underlying_type_t<T>)value;
  } else if constexpr (std::is_arithmetic_v<T>) {
    out << +value;
  } else {
    out << value;
  }
}

template <typename A, typename B>
void check_eq(
  const A& a, const B& b, const char* text_a, const char* text_b, const char* file, int line)
{
  if (a == b) { return; }
  std::ostringstream out;
  out.precision(17);
  out << text_a << " == " << text_b << " fails: ";
  print_value(out, a);
  out << " against ";
  print_value(out, b);
  fail(file, line, out.str());
}

inline int run_all(int argc, char** argv)
{
  std::vector<test_t> tests = registry();
  std::sort(tests.begin(), tests.end(), [](const test_t& a, const test_t& b) {
    return std::strcmp(a.name, b.name) < 0;
  });
  const char* filter = argc > 1 ? argv[1] : "";
  int ran = 0, failed = 0;
  for (const test_t& t : tests) {
    if (!std::strstr(t.name, filter)) { continue; }
    int before = failure_count();
    t.run();
    bool ok = failure_count() == before;
    std::printf("%s %s\n", ok ? "pass" : "FAIL", t.name);
    ++ran;
    failed += !ok;
  }
  std::printf("%d tests, %d failed\n", ran, failed);
  return failed != 0;
}

}  // namespace test
}  // namespace gprs

#define GPRS_TEST(name)                                                            \
  static void gprs_test_##name();                                                  \
  static ::gprs::test::registrar_t gprs_registrar_##name(#name, gprs_test_##name); \
  static void gprs_test_##name()

#define GPRS_CHECK(condition) ::gprs::test::check((condition), #condition, __FILE__, __LINE__)

#define GPRS_CHECK_EQ(a, b) ::gprs::test::check_eq((a), (b), #a, #b, __FILE__, __LINE__)

#define GPRS_REQUIRE(condition)                                                        \
  do {                                                                                 \
    if (!::gprs::test::check((condition), #condition, __FILE__, __LINE__)) { return; } \
  } while (false)
