/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// A row read as the reduced model would have it: its free entries, and its sides less the terms
// of its fixed columns. Shared by the doubleton equations, which read an edge that way, and by
// the column rules, whose records save rows.

#include <gprs/numerics.hpp>
#include <gprs/state.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace gprs {

// Calls entry(col, value) for every free entry of the row, in storage order, and returns the sum
// of the terms of its fixed columns, summed in storage order as rebuild() does.
// O(csr.used[row]).
template <typename entry_t>
inline dd_t scan_row(const state_t& st, int32_t row, entry_t&& entry)
{
  const line_store_t& csr = st.csr;
  dd_t fixed_terms;
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    if (st.is_live(col)) {
      entry(col, csr.value[e]);
    } else {
      dd_add_product(fixed_terms, csr.value[e], st.col_lower[col]);
    }
  }
  return fixed_terms;
}

// A side of a row less the terms of its fixed columns, the arithmetic of rebuild(); an infinite
// side stays. The difference is one double: next to a column fixed at 1e20 it holds nothing of
// a side of 3, so a rule that takes it as a side of the free part tests its size first
// (|side| * tol.side_error <= tol.feasibility, as the row rules do).
inline double free_side(double side, dd_t fixed_terms)
{
  return std::isinf(side) ? side : dd_residual(side, fixed_terms);
}

// Entries of saved rows, one row after the other.
struct row_pool_t {
  std::vector<int32_t> col;
  std::vector<double> value;
};

// The copy of a row at one moment: its sides less the terms of its fixed columns, and its
// entries [begin, end) of a pool. It names live columns only.
struct saved_row_t {
  int32_t row;
  double lower;
  double upper;
  int64_t begin;
  int64_t end;
};

// Appends the free entries of the row that `keep(col)` accepts to the pool, in storage order,
// and returns the saved row. O(csr.used[row]).
template <typename keep_t>
inline saved_row_t save_row(const state_t& st, int32_t row, row_pool_t& pool, keep_t&& keep)
{
  saved_row_t saved;
  saved.row        = row;
  saved.begin      = pool.col.size();
  dd_t fixed_terms = scan_row(st, row, [&pool, &keep](int32_t col, double value) {
    if (!keep(col)) { return; }
    pool.col.push_back(col);
    pool.value.push_back(value);
  });
  saved.end        = pool.col.size();
  saved.lower      = free_side(st.row_lower[row], fixed_terms);
  saved.upper      = free_side(st.row_upper[row], fixed_terms);
  return saved;
}

}  // namespace gprs
