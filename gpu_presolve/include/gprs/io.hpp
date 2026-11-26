/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <gprs/state.hpp>

#include <string>

namespace gprs {

// Binary CSR dump "GPRSCSR1" (baseline/README.md), little endian:
//   char magic[8]; int64 nrows, ncols, nnz; int64 row_start[nrows + 1]; int32 col_index[nnz];
//   double value[nnz]; double row_lower[nrows], row_upper[nrows];
//   double col_lower[ncols], col_upper[ncols]; double cost[ncols]; uint8 col_type[ncols];
//   double objective_offset.
// The file holds a minimisation.

// Reads a dump. Returns false with a message in error when the file cannot be read, is not a
// dump of exactly the size its header gives, or does not hold a well_formed model.
bool read_csr_dump(const std::string& path, model_t& model, std::string& error);

// Writes a well_formed model. Returns false with a message in error when the file cannot be
// written.
bool write_csr_dump(const std::string& path, const model_t& model, std::string& error);

}  // namespace gprs
