/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace gprs {

[[noreturn]] void assert_fail(const char* condition, const char* file, int line);

}  // namespace gprs

#if defined(GPRS_ASSERT) && GPRS_ASSERT
#define gprs_assert(condition) \
  ((condition) ? (void)0 : ::gprs::assert_fail(#condition, __FILE__, __LINE__))
#else
// The condition is compiled and never evaluated, so a value that only an assertion reads counts
// as used in a build without assertions.
#define gprs_assert(condition) (false ? (void)(condition) : (void)0)
#endif
