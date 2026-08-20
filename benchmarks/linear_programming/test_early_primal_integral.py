#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import tempfile
import unittest


_MODULE_PATH = Path(__file__).with_name("early_primal_integral.py")
_SPEC = importlib.util.spec_from_file_location(
    "early_primal_integral", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

parse_incumbents = _MODULE.parse_incumbents
primal_gap = _MODULE.primal_gap
primal_integral = _MODULE.primal_integral


class EarlyPrimalIntegralTest(unittest.TestCase):
    def test_primal_gap_definition(self) -> None:
        self.assertEqual(primal_gap(1.0, -1.0), 1.0)
        self.assertEqual(primal_gap(2.0, 1.0), 0.5)
        self.assertEqual(primal_gap(0.0, 0.0), 0.0)

    def test_integrates_incumbent_steps_and_ignores_regressions(self) -> None:
        # 1 for two seconds, then gap 1/2 for three seconds, then gap 1/4 for five seconds.
        incumbents = [(2.0, 2.0), (4.0, 3.0), (5.0, 4.0 / 3.0)]
        self.assertAlmostEqual(primal_integral(incumbents, 1.0, 10.0), 0.475)

    def test_integral_is_bounded_without_an_incumbent(self) -> None:
        self.assertEqual(primal_integral([], 1.0, 10.0), 1.0)

    def test_parses_global_and_legacy_early_times(self) -> None:
        log = """\
New solution from early primal heuristics (CPUFJ). Objective +2.0e+00. Time 0.25
Papilo presolve time: 4.00
New solution from early primal heuristics (CPUFJ). Objective +1.5e+00. Time 1.00
H                              +1.2e+00     +1.0e+00  16.7%  7.50
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solver.log"
            path.write_text(log)
            self.assertEqual(
                parse_incumbents(path, 10.0),
                [(0.25, 2.0), (1.0, 1.5), (7.5, 1.2)],
            )
            self.assertEqual(
                parse_incumbents(path, 10.0, legacy_local_early_time=True),
                [(0.25, 2.0), (5.0, 1.5), (7.5, 1.2)],
            )

    def test_records_solution_completed_by_presolve(self) -> None:
        log = """\
Papilo presolve time: 0.02
Optimal solution found during presolve.
Best objective 3.700000e+01, best bound 3.700000e+01, gap 0.00%.
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solver.log"
            path.write_text(log)
            self.assertEqual(parse_incumbents(path, 10.0), [(0.02, 37.0)])


if __name__ == "__main__":
    unittest.main()
