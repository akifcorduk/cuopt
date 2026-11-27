# Synthesis notes: what was checked in the code (read only)

All paths under /Users/acoerduek/Documents/workspace/cuopt-wt/gpu-presolve/gpu_presolve unless said otherwise.

Read in full: include/gprs/numerics.hpp, state.hpp, rules.hpp, presolve.hpp; src/rules_propagation.cpp,
rules_bounds.cpp, state_ops.cpp, presolve.cpp; CMakeLists.txt, build_host_local.sh, check/CMakeLists.txt;
tests/oracle.hpp. Read in part: src/state_build.cpp 55-350, src/state_storage.cpp 1-230 and 540-636,
microbench/sweeps.cu 196-770, tools/gprs_fuzz.cpp 190-310, tools/run_info.py 1-80,
design_summaries/gpu_mip_presolve/DESIGN.md 1-780.

Findings that decide disputed points:

1. Summation order (numerics.hpp:139-155, state_build.cpp:75-125, 176-209). 256 lanes for every row above
   64 entries, whatever its length. An infinite term enters as a * 0 and leaves the lane unchanged in bits
   because a lane starts at (+0, +0) and never holds -0. So a device kernel that skips infinite terms
   (sweeps.cu:200-207) gives the host bits. The Stage 0 light kernel (sweeps.cu:346-386) has the host tree;
   the block kernel merges with cub::BlockReduce (sweeps.cu:251-270) and the chunk kernels cut rows
   (433-469): both must be replaced for rows.
2. Every entry edit increments coefficient_changes and the row's version: scale_row (state_ops.cpp:433-436),
   set_coefficient (455-456), delete_entry (478-479), move_entry (519-520). Sides: side_changes and version
   (372-373, 385-386, 398-406, 426-431). Bounds: bound_changes (195, 290). Liveness: live_rows (353),
   live_cols (288, 299). Only rebuild changes device-relevant data without a counter (sides shifted,
   fixed columns removed, state_storage.cpp:142-187). So the first milestones can detect every change
   from existing hashed counters plus a note of the rebuild, with no edit of state.hpp.
3. Which family edits entries (grep over src/rules_*.cpp): tightening calls scale_row
   (rules_tightening.cpp:204) and set_coefficient (301), both value-only; doubleton calls move_entry,
   delete_entry, set_coefficient, remove_col (rules_doubleton.cpp:134-158); row rules and parallel rows
   only delete rows and change sides. Inside a cascade (presolve.cpp:207-221: row rules, propagation,
   tightening, dual fixing) no entry is added, deleted or moved. Structural edits happen only between
   cascades (doubleton) and in rebuild (end_phase).
4. Host rule (rules_propagation.cpp:51-134): guard uses the stored update count (+1 when the own term is
   removed); integer candidate moved outward by 4 rounding units, then tighten_* rounds and applies at
   once; continuous candidate only recorded. tighten_lower / tighten_upper (rules_bounds.cpp:15-49) never
   relax a bound and fix the column when the bounds meet.
5. oracle::check_presolve(case, options) calls presolve(model, options) (tests/oracle.hpp:231), so a
   backend that travels in options_t reaches the oracle and gprs_fuzz without an edit of the oracle.
6. presolve.hpp:16-17 lists numerics.hpp and rules.hpp as shared files with fixed interfaces. A macro in
   front of the inline functions of numerics.hpp does not change the interface.
7. DESIGN.md 2.3 (lines 224-225) describes the bins "up to 64, up to 8,192 (one block), chunks above";
   2.4 (lines 261-263) says a longer row is folded by one block. The host code follows 2.4.
8. DESIGN.md 13 (line 659) names the hand-back threshold as "consecutive small rounds after a hand-back".
