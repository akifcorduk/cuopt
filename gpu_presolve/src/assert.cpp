/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>

#include <cstdio>
#include <cstdlib>

namespace gprs {

void assert_fail(const char* condition, const char* file, int line)
{
  std::fprintf(stderr, "gprs_assert failed: %s at %s:%d\n", condition, file, line);
  std::fflush(stderr);
  std::abort();
}

}  // namespace gprs
