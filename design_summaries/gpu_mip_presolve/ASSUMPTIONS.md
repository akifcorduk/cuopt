# Assumptions register of the GPU MIP presolve

Version of 2026-11-27, 316 entries. Merged from eight readers (the plan, the design of the nine cheap column rules, the design of the five further rules, the two halves of the history, the engine code, the cuOpt adapter, the measurement scripts), reviewed by three critics (completeness, accuracy, strengthening), and brought up to HISTORY Entry 46 and commit `1112631b6` of branch `feat/gpu-presolve`.

The new presolve ("gprs") is built on choices that nobody has tested on what cuOpt is for: the quality of its solutions. This file lists every such choice, with its switch and with the test it needs, so that all of them can be tested at the end.

The owner's instruction of 2026-11-27 (HISTORY Entry 45), verbatim:

> Take a note of the assumptions as you go, we will test all of them at the end. remember nnz reduction is not the main goal is to strengthen the formulation. so we will test all of them at the end over the solution quality. strengthening might mean adding those mergings and additional constraints like indicator. but we will test them at the end. first goal is the nnz reduction.

**Order of the goals.** The first goal is the reduction in nonzeros (with rows and columns next to it), measured against the presolve of HiGHS only; every stage gate up to the end measures that. The main goal is a stronger formulation; it is tested at the end, over the solution quality of cuOpt. Strengthening may mean a reduction that removes no nonzero or that grows the model: the merges of integer columns that are held back today (AS-001), added constraints such as indicator strengthening (AS-003). No entry of this register is closed by a nonzero count alone (DESIGN.md amendment A10).

**How an entry is added.** When a choice is made, it gets an entry with all eight fields and the next free number, in the section it belongs to: one entry per choice that can be tested on its own. Numbers are never reused, and an entry that is settled stays in the file with its result. A choice that cannot be switched gets a switch when it is built, so that it can be tested ("no switch yet: needs one" marks the ones that still lack it).

**How the end test works.** Two versions of cuOpt that differ in one switch are run on the benchmark cluster on the 240 MIPLIB benchmark instances. In the entries this is called the *standard cluster comparison*: both arms run the new presolve followed by cuOpt's own four reductions (adapter mode 3, `presolve_gprs = 3`) with cuOpt's scaling off (`mip_scaling` 0), unless the entry names other arms. The two arms are compared on the primal side (primal gap against the best known value, optimal count, primal integral) and on the dual side, where a stronger formulation shows first (AS-217): the root relaxation objective (cuOpt's log line "Root relaxation objective"), the final bound with its gap to the best known value, the node count (log line "Explored ... nodes") and the time to optimality on the instances that both arms solve. Every arm also reports wrong statuses, postsolved solutions that are infeasible in the original model, presolve seconds, and the size and the hash of the reduced model. A pair is read first on the instances whose reduced model differs between the two arms, and for all 240 next to it.

Four things have to be in place before the first pair (they are entries themselves): the adapter passes only two engine options today, so an arm is a benchmark commit that changes one default, or the adapter gets a pass-through (AS-152); the spread between two runs of the same arm is not known, and the baseline with PaPILO and scaling off has no stored run (AS-214); the hash of the reduced model is not in cuOpt's log (AS-294); and a pair turns one switch only, so choices that stand in for each other are not seen (AS-216).

An entry whose choice cannot change the reduced model is tested by its own means, which the entry names: a *size run* (presolve only on the dev node, on the 240 benchmark instances and the 40 testbed instances: remaining nonzeros, rows and columns, and the hash of the reduced model), a timing run, the random *campaign* with the exact oracle, the structure check and the crush check, or a count.

**Closing rule.** A size run or a count closes an entry only when the hash of the reduced model is the same on all 240 with the choice at both values, or when the count is zero on the 240: no solver result can then differ. Any difference, however small in nonzeros, sends the entry to the standard cluster comparison, read on the instances that differ. Where the other value of a choice cannot be set, the entry stays open as "needs a switch".

**How to read an entry.**

- **Set by** says who chose: the owner (with their words and the HISTORY entry), taken as recommended (the assistant went on with its own recommendation; the owner has not decided), design (a choice of a plan or design text), or code (a value that exists only in the code). A finding or a belief (section 7) is nobody's choice; its Set by line says "design" for the plan or design that relies on it and names, in brackets, where it was measured or read. An entry marked "taken as recommended", "design" or "code" has no owner decision behind it.
- Two attributions hold for many entries and are stated once here. The thirteen plan changes of section 8 of the Stage 0 report entered the plan as a block: the owner's reply names five numbered decisions ("3. do the recommendations", "5. do as recommended", HISTORY Entry 13) and has no sentence on the thirteen as such (DESIGN.md section 0, decision 6, line 193; STAGE0_REPORT.md lines 396-439). And everything fixed in HISTORY Entries 14 to 21 (the readings of plan version 4, the Stage 1 code structure, the budgets, the repairs, the design of the checks) was fixed during six hours of independent work without a check-in (Entry 13: "From now on, work for the rest of the 6 hours of node independenly.").
- **Switch** gives the exact name and values of the option, constant or script argument that sets the choice, or "no switch yet: needs one". Where a finding or a convention has no setting of its own, the line names the script that reproduces it or the switch of the choice that rests on it.
- **State** says where the choice lives: in the engine, in the adapter, in cuOpt as is, designed (with the milestone), planned, or measurement only. "In the engine and no rule reads it yet" marks an option that the groundwork added for a rule that is not built.
- Paths are relative to the worktree `/Users/acoerduek/Documents/workspace/cuopt-wt/gpu-presolve` (branch `feat/gpu-presolve`), and line numbers are those of commit `1112631b6`, which contains the groundwork of the column rules (commits `6eed652cf` and `74350acb3`, verified on the node, HISTORY Entry 46). The one exception is `cpp/src/mip_heuristics/solve.cu`: its line numbers are those of the working tree with the uncommitted pre-flight patch (AS-231); in the committed file subtract 5 after line 739. DESIGN.md, HISTORY.md, `stage1b_design/` and `stage1c_design/` are in `design_summaries/gpu_mip_presolve/`; `v2_*.txt` is the design of the nine cheap column rules, `lowworth_v2_*.txt` the design of the five further rules. Each line of those design files is a long paragraph, so a line number names a paragraph.
- Instance sets: "the 240" are the 240 MIPLIB 2017 benchmark instances (`gpu_presolve/testbed/benchmark240.json`: 147 at or above 50,000 nonzeros, 93 below); "237 of the 240" leaves out ex9, ex10 and enlight_hard (AS-191); "the 40" are the testbed instances (`gpu_presolve/testbed/testbed.json`, 31 below 1e7 nonzeros and 9 above); "the 31" are those below 1e7 nonzeros and "the 9" those above; "the 17 local dumps" are the model dumps on the Mac.
- Adapter modes (`presolve_gprs`): 0 PaPILO as today, 1 the new presolve and then all of PaPILO, 2 the new presolve alone, 3 the new presolve and then cuOpt's own four reductions (indicator strengthening, GF2, BHW coefficient reduction, single-lock dual aggregation).
- Milestones of the cheap column rules (Stage 1c), each written in the entries as a name with its design code in brackets. The nine column rules: the groundwork [M0]; the singleton-column rules and the forcing column [M1]; simple probing on equations [M2]; dominance between parallel columns [M3]; dual bound strengthening, then the single-lock alias [M4]; the merge of parallel columns [M5]. The five further rules: the analysis script and the reference runs, without engine code [L0]; the implied equation [L1]; the implied-integer conversion with column scaling [L2]; column stuffing [L3]; the single-equation extension of dual fixing [L4]; the dominated column [L5]. The design codes of the rules (R1 to R9) also appear only in brackets after the rule's name.
- Where a number is quoted from the history, the plan or a design, it was copied, not recomputed. Numbers marked "recounted" were recomputed from the stored result files for this register.

## Overview

| ID | Assumption | Switch | State | Set by |
|---|---|---|---|---|
| AS-001 | The general merge of two parallel integer columns is held back (`range_kept`) | `options_t::merge_integers` = `never`, `range_kept` (default), `always` | designed (milestone of the merge of parallel columns [M5]) | taken as recommended |
| AS-002 | `range_kept` needs dual bound strengthening, so no integer pair is merged in primal-only mode | `options_t::merge_integers` together with `options_t::primal_only` and `options_t::sub_enabled[sub_rule::dual_bound]` | designed (milestone of the merge of parallel columns [M5]) | design |
| AS-003 | Indicator strengthening stays on and grows the model; its native port is not built | `mip_hyper_params_t::presolve_indicator_strengthening` (default true; parameter `mip_hyper_presolve_indicator_strengthening`) | in cuOpt as is (reached in modes 0, 1 and 3; in mode 2 only when the model falls back to PaPILO, AS-156) | design |
| AS-004 | The zero-cost singleton rule and the forcing column are off in primal-only mode | no switch yet: needs one | designed (milestone of the singleton-column rules and the forcing column [M1]) | taken as recommended |
| AS-005 | The longest cliques of knapsack and inferred set-packing rows are not added as model rows | no switch yet: needs one | planned, off | design |
| AS-006 | Clique merging is built with the probing stage and gated apart from probing | no switch yet: needs one | planned (step 4 of the build order) | the owner |
| AS-007 | Clique export to cuOpt, clique-aware propagation and knapsack cliques are undecided | no switch yet: needs one | planned, undecided | design |
| AS-008 | A clique row is replaced or removed only by a clique in the model | no switch yet: needs one | planned | design |
| AS-009 | Coefficient lifting from probing is not in the plan | no switch yet: needs one | not planned | design |
| AS-010 | Substitutions from cliques and implications wait for the probing stage, without a switch or gate | no switch yet: needs one | planned (step 4 of the build order), no design | design |
| AS-011 | All of HiGHS's cheap rules are built; the five low-worth ones come after the nine | no switch yet: needs one | designed (milestones of the five further rules [L1 to L5], which come after the nine column rules [M1 to M5]) | the owner |
| AS-012 | Sparsify and the aggregator are built second and valued by their size effect | no switch yet: needs one | planned (step 2 of the build order) | the owner |
| AS-013 | Pairwise dominated columns are fixed only under strict conditions | no switch yet: needs one | planned (step 3 of the build order) | design |
| AS-014 | Fourier-Motzkin, variable-bound aggregation and dual-fix probing are left out; enumeration is planned last | no switch yet: needs one | enumeration planned (step 4) | design |
| AS-015 | Removal of dependent rows and the split into components are not in the plan | no switch yet: needs one | not planned | design |
| AS-016 | Network implied integers, two-row bound tightening and two-column cancellation are not in the plan | no switch yet: needs one | not planned | design |
| AS-017 | The engine has no dual postsolve | no switch yet: needs one | in the engine | design |
| AS-018 | Gcd transforms, gcd substitution and variable-bound extraction have no place in the build order | no switch yet: needs one | planned without a place | design |
| AS-019 | Cover strengthening is valued by nonzeros and has no place in the build order | no switch yet: needs one | planned without a place | design |
| AS-020 | An integer column of range 1 is not moved to the bounds 0 and 1 | no switch yet: needs one | not planned | design |
| AS-021 | Row tests use one activity per side, over explicit bounds only | no switch yet: needs one | in the engine (single sum) | design |
| AS-022 | Every test of the nine column rules reads explicit bounds only | no switch yet: needs one | designed (milestones of simple probing [M2] and of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-023 | A bound counts as implied only by one row read with explicit bounds; no chains | no switch yet: needs one | designed | design |
| AS-024 | Dual arguments use continuous columns, strict inequalities and one implied premise | no switch yet: needs one | designed (rules of the milestones of the nine column rules and the five further rules [M1 to L5]) | design |
| AS-025 | An integer column is substituted only when the pivot row proves integrality | no switch yet: needs one | in the engine for doubleton equations | design |
| AS-026 | Simple probing works only on equations and only on a binary of range exactly 1 | `options_t::enabled[rule::simple_probing]` (tool `--off simple_probing`) turns the rule | designed (milestone of simple probing [M2]) | design |
| AS-027 | An inequality-pivot free singleton is taken only when its own cost gives the dual sign | `options_t::sub_enabled[sub_rule::free_singleton_dual]` turns the built case | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-028 | The zero-cost singleton rule takes only rows with two finite sides | `options_t::sub_enabled[sub_rule::zero_cost_singleton]` (tool `--off zero_cost_singleton`) turns the rule | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-029 | The zero-cost singleton rule leaves integer columns | no switch yet: needs one | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-030 | Dual bound strengthening never tightens a continuous column, it only fixes it | no switch yet: needs one | designed (milestone of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-031 | The single-lock alias tests its forcing on the lock row only | no switch yet: needs one | designed (milestone of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-032 | Two forms that the design attributes to PaPILO are not built with the nine rules | no switch yet: needs one | planned | design |
| AS-033 | BHW coefficient reduction runs after the new presolve, never inside it, and overlaps coefficient strengthening | no switch yet: needs one | in cuOpt as is | design |
| AS-034 | The equation test of the conversion counts as primal, the weak test as dual | `options_t::primal_only` together with the designed `implied_integer_weak` | designed (milestone of the implied-integer conversion [L2]) | design |
| AS-035 | The weak conversion test accepts a dual-implied equation only through the implied-equation rule | `implied_equations` and `implied_integer_weak` (designed engine switches; not in code) | designed (milestone of the implied-integer conversion [L2]) | design |
| AS-036 | A converted column loses its roles in other rules | no switch yet: needs one | designed (milestone of the implied-integer conversion [L2]) | design |
| AS-037 | Column stuffing takes integer candidates only when all candidates are integer with equal coefficient | no switch yet: needs one | designed (milestone of column stuffing [L3]) | design |
| AS-038 | The single-equation extension is on, with continuous absorbers only | `single_equation` (designed; not in code) | designed (milestone of the single-equation extension [L4]) | design |
| AS-039 | The dominated-column rule is on, with continuous compensators only | `dominated_columns` (designed; not in code) | designed (milestone of the dominated column [L5]) | design |
| AS-040 | A strictly dominated column with an infinite target gives no "unbounded or infeasible" status | no switch yet: needs one | designed (milestone of the dominated column [L5]) | design |
| AS-041 | A dominated-column fixing against the cost needs the implied equation first | no switch yet: needs one | designed (milestone of the dominated column [L5]) | design |
| AS-042 | Conversion, single equation and dominated column do nothing on models without a continuous column | no switch yet: needs one | designed (milestones of the implied-integer conversion [L2] and of the single-equation extension [L4]) | design |
| AS-043 | All rule families of the Stage 1 engine are on by default | `options_t::enabled[rule::row_rules, propagation, tightening, doubleton, parallel_rows, dual_fixing]` | in the engine | design |
| AS-044 | The engine uses dual reductions by default; primal-only mode switches them off | `options_t::primal_only` (default false) | in the engine | design |
| AS-045 | Dual fixing is on, also for integer columns; zero-cost columns go to the lower bound | `options_t::enabled[rule::dual_fixing]` (tool `--off dual_fixing`) | in the engine | design / code |
| AS-046 | Row addition is on: equations are subtracted from rows parallel up to singleton columns | `options_t::near_parallel_rows` (default true) | in the engine | the owner / design |
| AS-047 | Implied bounds of continuous columns become explicit bounds of the reduced model | no switch yet: needs one | in the engine | design |
| AS-048 | Coefficient strengthening of integer columns is on | no switch yet: needs one | in the engine | design |
| AS-049 | Sides of all-integer rows are rounded, with the gcd of the coefficients | no switch yet: needs one | in the engine | design |
| AS-050 | All nine column reductions are on by default | `options_t::sub_enabled[sub_rule::free_singleton, zero_cost_singleton, forcing_column, free_singleton_dual, parallel_dominance, dual_bound, lock_alias, parallel_merge]`, `options_t::enabled[rule::simple_probing, singleton_cols, dual_cols, parallel_cols]` | designed (milestones of the nine column rules [M1 to M5]) | design |
| AS-051 | The single-lock alias is required, as one rule for HiGHS's and cuOpt's forms | `options_t::sub_enabled[sub_rule::lock_alias]` (tool `--off lock_alias`) | designed (milestone of dual bound strengthening and the single-lock alias [M4], second step) | taken as recommended |
| AS-052 | Which columns the single-lock alias takes | `options_t::sub_enabled[sub_rule::lock_alias]` turns the rule | designed (milestone of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-053 | The single-lock alias tests one candidate binary per row | no switch yet: needs one | designed (milestone of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-054 | The single-lock alias releases only the locking side of its row | no switch yet: needs one | designed (milestone of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-055 | Dual bound strengthening asks only for the lock in its own direction | `options_t::sub_enabled[sub_rule::dual_bound]` (tool `--off dual_bound`) | designed (milestone of dual bound strengthening and the single-lock alias [M4], first step) | design |
| AS-056 | The forcing column goes by lock counts; its undo takes the minimum over rows | `options_t::sub_enabled[sub_rule::forcing_column]` (tool `--off forcing_column`) | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-057 | Zero-cost singleton columns of a row share one record, with a fixed undo order | no switch yet: needs one | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-058 | Two continuous parallel columns are always merged | `options_t::sub_enabled[sub_rule::parallel_merge]` turns every merge | designed (milestone of the merge of parallel columns [M5]) | taken as recommended |
| AS-059 | An integer column is merged into a continuous one, and no column changes type | `options_t::sub_enabled[sub_rule::parallel_merge]` | designed (milestone of the merge of parallel columns [M5]) | design |
| AS-060 | Which of two parallel integer columns is kept, and the no-hole condition | `options_t::merge_integers` (AS-001) | designed (milestone of the merge of parallel columns [M5]) | design |
| AS-061 | The undo of a merge is a closed form preferring the removed column's lower bound | no switch yet: needs one | designed (milestone of the merge of parallel columns [M5]) | design |
| AS-062 | Singleton columns take part in the parallel-column rules | no switch yet: needs one | designed (milestone of dominance between parallel columns [M3]) | design |
| AS-063 | The implied equation is on by default and turns inequalities into equations | `implied_equations` (designed as an engine switch; not in code) | designed (milestone of the implied equation [L1]) | design |
| AS-064 | An integer column may be the source of an implied equation | `integer_sources` (designed, default on; not in code) | designed (milestone of the implied equation [L1]) | design |
| AS-065 | The implied-integer conversion is on by default and runs last in every round | `implied_integer_equation`, `implied_integer_weak`, `implied_integer_scaling` (designed; not in code) | designed (milestone of the implied-integer conversion [L2]) | design |
| AS-066 | Column scaling is built and on, but only where the equation test fails without it | `implied_integer_scaling` (designed; not in code) | designed (milestone of the implied-integer conversion [L2], second step) | design |
| AS-067 | Column stuffing is on by default, with no cap on the row length | `column_stuffing` and the option `stuffing_row_limit` = 0 (designed; not in code) | designed (milestone of column stuffing [L3]) | design |
| AS-068 | One dominated-column fixing may use several implied premises, one per locking row | no switch yet: needs one | designed (milestone of the dominated column [L5]) | design |
| AS-069 | The feasibility tolerance of the engine is 1e-6, absolute, and its own | `options_t::tol.feasibility` (default 1e-6) | in the engine | design |
| AS-070 | Bounds of integer columns are rounded with the feasibility tolerance | no switch yet: needs one | in the engine | code |
| AS-071 | The integrality tolerance 1e-5 is read by no reduction | `options_t::tol.integrality` (default 1e-5) | in the engine | code |
| AS-072 | The zero tolerance 1e-9 decides proportionality of rows and integrality of coefficients | `options_t::tol.zero` (default 1e-9) | in the engine | design |
| AS-073 | The dual tolerance has no reader; dual fixing reads the sign of the cost exactly | no switch yet: needs one | in the engine | design / code |
| AS-074 | A term above 1e20 is infinite in an activity; a bound only as IEEE infinity | `options_t::tol.infinite_term` (default 1e20) | in the engine and the adapter | design |
| AS-075 | An activity is summed again, and a candidate dropped, at a tenth of the tolerance | `options_t::tol.activity_error_share` (default 0.1) | in the engine | design / code |
| AS-076 | Sides, constants and fixing values above 1e8 are left alone | `options_t::tol.side_error` (default 1e-14) together with `options_t::tol.feasibility` | in the engine for the Stage 1 rules | design |
| AS-077 | An implied bound of a continuous column counts only when it improves by over 1e-3 | `options_t::tol.implied_bound_step` (default 1000) | in the engine | design |
| AS-078 | Derived bounds are moved outward by their rounding and by the activity error | no switch yet: needs one | in the engine | code |
| AS-079 | One explicit bound of a continuous column is tightened at most 16 times | no switch yet: needs one | in the engine | code |
| AS-080 | A combined coefficient is a structural zero below 1e-12 relative; small coefficients are not dropped | `options_t::tol.cancellation` (default 1e-12) | in the engine (cancellation) | design |
| AS-081 | An equation is a row whose two sides are equal bit for bit | no switch yet: needs one | in the engine | code |
| AS-082 | Sides and bounds that cross by at most 1e-6 are joined, not reported as infeasible | no switch yet: needs one | in the engine | design |
| AS-083 | Forcing rows: tolerance over dynamism; with a continuous column only up to rounding | no switch yet: needs one | in the engine | design / code |
| AS-084 | Rows of integer columns are scaled to coprime integer coefficients within three limits | no switch yet: needs one | in the engine | design / code |
| AS-085 | Bounds moved to an integer column through a doubleton equation round with tolerance over scale | no switch yet: needs one | in the engine | code |
| AS-086 | Rows and columns are compared with at most 4 bucket neighbours in each direction | `options_t::parallel_bucket_limit` (default 4) | in the engine for rows | code / design |
| AS-087 | The subtraction of parallel rows is made only inside five guards | no switch yet: needs one | in the engine | design |
| AS-088 | Where the column leaves the model, a bound is implied within about 2e-7 | no switch yet: needs one | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-089 | For dominance between parallel columns the absorber's bound is implied up to rounding only | no switch yet: needs one | designed (milestone of dominance between parallel columns [M3]) | design |
| AS-090 | Implied equation and locked compensator: a bound is implied up to rounding, with a cap | no switch yet: needs one | designed (milestones of the implied equation [L1] and of the dominated column [L5]) | design |
| AS-091 | Simple probing uses the engine's forcing-row test, not HiGHS's absolute 1e-9 | no switch yet: needs one | designed (milestone of simple probing [M2]) | design |
| AS-092 | A column is replaced by a binary only when its range is at most 1e6 | no switch yet: needs one | designed (milestones of simple probing [M2] and of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-093 | One shared integrality test: ratios within 1e-9 and a drift guard of 0.1 tolerance | no switch yet: needs one | designed (to be built with the milestone of the singleton-column rules and the forcing column [M1]; binding for the implied equation [L1] and the implied-integer conversion [L2]) | design |
| AS-094 | A chain of free singleton substitutions may grow a deviation by at most 1000 | `options_t::pivot_growth_limit` (default 1000) | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-095 | A free singleton substitution has no limit on the cost entries it creates | no switch yet: needs one | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-096 | The zero-cost singleton rule and the forcing column need a cost of exactly zero | no switch yet: needs one | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-097 | Postsolve records may save rows up to 4 times the nonzeros of the input model | `options_t::saved_row_factor` (default 4) | designed (milestone of the singleton-column rules and the forcing column [M1] on) | design |
| AS-098 | Two columns are parallel within 1e-9 relative and 0.1 of the tolerance over the bounds | `options_t::tol.zero`, `options_t::tol.activity_error_share`, `options_t::tol.feasibility` (shared with other tests) | designed (milestone of dominance between parallel columns [M3]) | design |
| AS-099 | Costs of parallel columns count as equal only to 1e-12 relative | `options_t::tol.cancellation` (default 1e-12; shared with the structural-zero test) | designed (milestones of dominance between parallel columns [M3] and of the merge of parallel columns [M5]) | design |
| AS-100 | Dual bound strengthening rounds to the safe side and needs an improvement above the tolerance | no switch yet: needs one | designed (milestone of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-101 | The single-lock alias forces a continuous column only up to rounding | no switch yet: needs one | designed (milestone of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-102 | The single-lock alias has no limit on the length of its lock row | `options_t::lock_alias_row_limit` (default 0, no limit) | designed (milestone of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-103 | An implied-equation source needs a single lock and a cost beyond the dual tolerance | `options_t::tol.dual` (default 1e-7) | designed (milestone of the implied equation [L1]) | design |
| AS-104 | A conversion rounds bounds inward and is refused where rounding would widen a bound | no switch yet: needs one | designed (milestone of the implied-integer conversion [L2]) | design |
| AS-105 | The column scaling works inside fixed limits | no switch yet: needs one | designed (milestone of the implied-integer conversion [L2], second step) | design |
| AS-106 | The sums of column stuffing and the single-equation rule allow rounding only | no switch yet: needs one | designed (milestones of column stuffing [L3] and of the single-equation extension [L4]) | design |
| AS-107 | The dominated-column rule limits prices to 1e7 and coefficient ratios to 1000 | no switch yet: needs one | designed (milestone of the dominated column [L5]) | design |
| AS-108 | The sign of the dominated-column rate is tested to 1e-12 relative, with a capped rise | `options_t::tol.cancellation` (default 1e-12) | designed (milestone of the dominated column [L5]) | design |
| AS-109 | The five rules' dual tolerances stay stricter than HiGHS until their price is measured | no switch yet: needs one | designed (milestones of the five further rules [L1 to L5]) | design |
| AS-110 | Aggregator substitutions: pivot at least 0.01, fill-in at most 10, coefficient ratio at most 1e6 | no switch yet: needs one | planned (step 2 of the build order) | design |
| AS-111 | Probing is limited by work in entries, a stall stop and caps per pass | no switch yet: needs one | planned (step 4 of the build order) | the owner |
| AS-112 | At run time all rules are phases of one loop with fixed alternations | no switch yet: needs one | planned (the Stage 1 rounds are in the engine) | design |
| AS-113 | After Stage 1 rules are built in order of nonzero worth on the 240 | `options_t::enabled`, `options_t::sub_enabled`, `HIGHS_DIAG_OFF` | in force | the owner |
| AS-114 | A round runs the cascade, parallel rows, doubleton equations, then the column families | no switch yet: needs one | in the engine (the row phases and four column families that are still empty) | design |
| AS-115 | The two families of the five rules run last in a round, the conversion last | no switch yet: needs one | designed | design |
| AS-116 | Inside a column family the reductions are tried in a fixed order | no switch yet: needs one | designed (milestones of the nine column rules [M1 to M5]) | design |
| AS-117 | The column rules run both before and after the switch to explicit bounds | no switch yet: needs one | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-118 | The presolve has at most 20 rounds in all and 100 cascade passes per round | `options_t::max_rounds` (default 20) and `options_t::max_cascade_passes` (default 100) | in the engine | code / taken as recommended |
| AS-119 | A new equation or converted column reaches the other rules only in the next round | `options_t::max_rounds` bounds it | designed | design |
| AS-120 | The presolve ends at the first round without a finding, even after a budget stop | no switch yet: needs one | in the engine | design |
| AS-121 | Every rule family has a work budget and a stall stop; the values are placeholders | `options_t::budget[rule]` = {passes, stall_passes, floor} | in the engine (ten families, four of them still empty) | the owner / code |
| AS-122 | Work counts one unit per matrix entry and eight per row or column visited | no switch yet: needs one | in the engine | design |
| AS-123 | Work ceilings of the column families are reports only until measured | no switch yet: needs one | measurement only | taken as recommended |
| AS-124 | The time limit is only an abort; it never changes which reductions are tried | `options_t::time_limit` (default infinite) | in the engine | design |
| AS-125 | The dual column rules run once per round, not inside the cascade | no switch yet: needs one | designed (milestone of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-126 | Simple probing applies the aliases of a call as one batch through the alias forest | no switch yet: needs one | designed (milestone of simple probing [M2]) | design |
| AS-127 | The free singleton rule takes the smallest column index and finishes chains inside one call | no switch yet: needs one | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-128 | Dominance between parallel columns follows fixed type rules and a fixed order of cases | `options_t::sub_enabled[sub_rule::parallel_dominance]` turns the rule | designed (milestone of dominance between parallel columns [M3]) | design |
| AS-129 | A parallel pair is merged only when no dominance case applies | `options_t::sub_enabled[sub_rule::parallel_dominance]` and `[sub_rule::parallel_merge]` | designed (milestones of dominance between parallel columns [M3] and of the merge of parallel columns [M5]) | design |
| AS-130 | The single-lock alias applies all candidates of one binary in one transaction | no switch yet: needs one | designed (milestone of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-131 | The implied equation runs between dual bound strengthening and the single-lock alias | no switch yet: needs one | designed (milestone of the implied equation [L1]) | design |
| AS-132 | Doubleton equations form a spanning forest; the kept column is chosen by type, then entries | no switch yet: needs one | in the engine | design |
| AS-133 | Inside cuOpt the new presolve is off by default; its mode is no runtime parameter | `mip_hyper_params_t::presolve_gprs` = 0, 1, 2, 3 (default 0) | in the adapter | design |
| AS-134 | Mode 3 is the comparison arm: the new presolve, then cuOpt's four own reductions | `presolve_gprs` = 3 | in the adapter (27 unit tests pass in modes 1 to 3; no solver run on the cluster exists) | design |
| AS-135 | Mode 1, all of PaPILO after the new presolve, has no number on the 240 | `presolve_gprs` = 1 | in the adapter | design |
| AS-136 | cuOpt's four reductions run once after the new presolve, hosted by PaPILO, until ported | no switch yet: needs one | in the adapter through the PaPILO framework | design |
| AS-137 | cuOpt's single-lock rule keeps running in mode 3 after the engine has its own | no switch yet: needs one | planned (milestone of dual bound strengthening and the single-lock alias [M4]) | design |
| AS-138 | Two PaPILO methods of the baseline arm have no rule in the plan | no switch yet: needs one | in cuOpt as is | code / design |
| AS-139 | Indicator strengthening sees the model after the new presolve instead of before | no switch yet: needs one | in the adapter as "after" | design |
| AS-140 | Block BVE is taken to be unaffected by replacing PaPILO | `mip_hyper_params_t::presolve_block_bve` (default true; parameter `mip_hyper_block_bve`) | in cuOpt as is | design |
| AS-141 | cuOpt's probing cache stays, with a work budget fitted on PaPILO-reduced models | `mip_probing` (default true) for on and off | in cuOpt as is | code |
| AS-142 | cuOpt's own presolve runs after every mode and repeats part of a switched-off rule | `mip_probing` (default true) for the probing cache and `mip_hyper_block_bve` for block BVE | in cuOpt as is | code |
| AS-143 | cuOpt looks for symmetry on the reduced model, so the presolve decides what is left | cuOpt parameter `mip_symmetry` (default -1; 0 switches detection off; values up to 2) | in cuOpt as is | code |
| AS-144 | Probing and cliques enter cuOpt first as replacements, with PaPILO unchanged | no switch yet: needs one | planned | design |
| AS-145 | PaPILO's limits come from the original model even when it runs on the reduced one | `mip_hyper_params_t::presolve_papilo_probing_max_badgesize` and `presolve_max_rounds` (both default -1) | in cuOpt and the adapter | code |
| AS-146 | Early heuristics run as long as the presolve lasts; four threads stay reserved for PaPILO | no switch yet: needs one | in cuOpt as is | code |
| AS-147 | The adapter keeps a second model copy; the plan's memory rule is not in code | no switch yet: needs one | in the adapter (copy) | code / design |
| AS-148 | cuOpt's scaling is off in all development and in every benchmark arm | cuOpt setting `mip_scaling` = 0 off, 1 on, 2 on without the objective (the code default is 2) | in force as a convention | the owner (for now) |
| AS-149 | cuOpt scales before its presolve; "scale after the presolve" is the expected question | no switch yet: needs one | in cuOpt as is | code / design |
| AS-150 | cuOpt's sub-MIPs keep PaPILO | no switch yet: needs one | in cuOpt as is | the owner (for now) |
| AS-151 | The new presolve runs only on the main MIP, and only when presolve is on | `presolve_gprs` together with the parameter `presolve` (-1 default, 0 off, 1 PaPILO, 2 PSLP) | in the adapter | design |
| AS-152 | The adapter passes two options; every other engine setting keeps its engine default | no switch yet: needs one | in the adapter | code |
| AS-153 | cuOpt's tolerances are not passed to the engine | no switch yet: needs one | in the adapter | code |
| AS-154 | Inside cuOpt the engine runs primal-only only when a set-solution callback is registered | no switch yet: needs one | in the adapter | code |
| AS-155 | The presolve's time limit inside cuOpt is the whole remaining solve time | the solver's `time_limit` and `mip_determinism_mode` | in the adapter | code |
| AS-156 | A refused model, or a presolve that hits its limit, gets the full PaPILO presolve | no switch yet: needs one | in the adapter | code |
| AS-157 | The adapter knows two variable types: integer and continuous | no switch yet: needs one | in the adapter | design |
| AS-158 | A maximisation is negated, objective scale and offset are carried, names are dropped | no switch yet: needs one | in the adapter | design |
| AS-159 | The adapter drops stored zeros, sorts rows, refuses duplicate entries and needs double precision | no switch yet: needs one | in the adapter | design |
| AS-160 | The engine's answer "infeasible" ends the solve without a second check | no switch yet: needs one | in the adapter | code |
| AS-161 | The status "unbounded or infeasible" is passed on as it is | no switch yet: needs one | in the adapter | design |
| AS-162 | A model reduced to nothing counts as solved by the presolve | no switch yet: needs one | in the adapter | design |
| AS-163 | An unchanged model goes on to the third-party step only in modes 1 and 3 | `presolve_gprs` | in the adapter | design |
| AS-164 | No implied-integer flags reach the solver in modes 2 and 3 | no switch yet: needs one | in the adapter (a blind spot) | code |
| AS-165 | A converted column reaches cuOpt as a true integer; there is no third column type | no switch yet: needs one | designed (milestone of the implied-integer conversion [L2]) | design |
| AS-166 | A scaled column keeps its index with a new meaning; cuOpt's "transformed" marker is missing | no switch yet: needs one | designed in the engine (milestone of the implied-integer conversion [L2]) | design |
| AS-167 | The by-products of the plan's interface are not exported | no switch yet: needs one | planned (probing and clique stage) | design |
| AS-168 | A postsolved solution that violates the original model gives a warning only | no switch yet: needs one | in cuOpt as is | code |
| AS-169 | Solutions found on the original model are carried over by crush and may be lost | no switch yet: needs one | in the adapter | code |
| AS-170 | Injected solutions are checked against the adapter's own copy of the original model | cuOpt's tolerances `absolute_tolerance`, `relative_tolerance`, `integrality_tolerance` | in the adapter | design |
| AS-171 | Until the end every gate measures size against HiGHS presolve only | no switch yet: needs one | measurement only | the owner |
| AS-172 | A gain is claimed only from 50,000 nonzeros up; results are reported per size class | no switch of its own: a reporting boundary, held in the band field of the testbed files (gpu_presolve/testbed/make_benchmark_testbed.py:104-112) | measurement only | the owner |
| AS-173 | Presolve is measured on the dev node, solver quality only on the cluster | no switch yet: needs one | measurement only | the owner |
| AS-174 | Decisions rest on the 240 benchmark instances, and every figure names its instance set | runner option `--testbed gpu_presolve/testbed/benchmark240.json --set all` | measurement only | the owner |
| AS-175 | The testbed is 40 instances drawn by a feature cover of the MIPLIB 2017 collection | runner option `--testbed gpu_presolve/testbed/testbed.json` with `--set dev`, `large`, `status` or `all` | measurement only | the owner |
| AS-176 | Evidence above 1e7 nonzeros rests on nine testbed instances | runner option `--set large` | measurement only | the owner |
| AS-177 | Status instances have their own rule, and the rule behind neos-3230511-yuna is unknown | runner option `--set status` | measurement only | taken as recommended |
| AS-178 | Seventeen local model dumps stand in for the node | the directory passed to the runners with `--dumps` | measurement only | design |
| AS-179 | The standalone engine reads each model as HiGHS's MPS reader built it | no switch yet: needs one | measurement only | design |
| AS-180 | The diagnostic patch was checked in the full setting only on 204 of the 240 | no switch yet: needs one | measurement only | design |
| AS-181 | HiGHS is the only standalone baseline | no switch yet: needs one | measurement only | the owner |
| AS-182 | The reference is HiGHS 1.16.0 at a pinned commit: 64-bit integers, one thread, diagnostic patch | `COMMIT` and the cmake options in build_highs_diag.sh (`-DHIGHSINT64=ON`) | measurement only | design |
| AS-183 | A reference is one presolve-only call of HiGHS with a time limit | `--time-limit` of baseline_tool and run_baseline.py | measurement only | design |
| AS-184 | The diagnostic patch is taken not to change HiGHS when no switch is set | `HIGHS_DIAG_OFF=<names>`, `HIGHS_DIAG_PROBING_CONTINGENT`, `HIGHS_DIAG_PROBING_NOSTOP`, `HIGHS_DIAG_PROBE_ROWS`, `HIGHS_DIAG_REDUNDANCY_EXPLICIT` | measurement only | design |
| AS-185 | The Stage 1 reference is HiGHS restricted to our rules, with its lock rule on | `HIGHS_DIAG_OFF=probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns,parallel_columns,simple_probing,dual_fixing_substitution,implied_equations,implied_integer_conversion`, `--presolve-rule-off 9175456` | measurement only | the owner / taken as recommended |
| AS-186 | The reference of the cheap column rules is HiGHS with ten expensive stages off | `HIGHS_DIAG_OFF=probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns` | measurement only | design |
| AS-187 | The first three milestones are accepted against a HiGHS ladder that has no forcing column | `--presolve-rule-off 8913056`, `--presolve-rule-off 524288`, `M1_OFF`, `M2_OFF`, `M3_OFF` | measurement only | design |
| AS-188 | The worth of a rule is measured by switching it off in HiGHS, in nonzeros | `--presolve-rule-off`, `parallel_columns`, `simple_probing`, `dual_fixing_substitution`, `implied_equations`, `implied_integer_conversion`, `dual_fixing_single_equation`, `implied_row_dual_bounds`, `HIGHS_DIAG_OFF` | measurement only | design |
| AS-189 | A rule's predicted gain is the smaller of the engine's gap and HiGHS's loss | no switch yet: needs one | measurement only | design |
| AS-190 | The reductions measure is a geometric mean of the ratio of remaining nonzeros | no switch yet: needs one | measurement only | the owner |
| AS-191 | Means over the 240 leave out ex9, ex10 and enlight_hard | the constant `LEFT_OUT = ("ex9", "ex10", "enlight_hard")` in the two scripts | measurement only | taken as recommended |
| AS-192 | A size of zero counts as one; instances without a model are handled two ways | no switch yet: needs one | measurement only | code |
| AS-193 | An instance is left out of a comparison when one side has no result | `--time-limit` of run_baseline.py | measurement only | code |
| AS-194 | The reductions gate is 0.99 in the mean, below 0.95 explained, a lower bound only | the constant 0.99 in gpu_presolve/analysis/stage1_compare.py:323 and `BELOW = 0.95` in gpu_presolve/analysis/bench240_stage1c.py:79 | measurement only | the owner |
| AS-195 | The cheap-rules gate has the same two numbers in nonzeros, rows and columns | `BELOW = 0.95` in gpu_presolve/analysis/bench240_stage1c.py:79 and the 0.99 of the design | measurement only | taken as recommended |
| AS-196 | In the gate mean the engine's leads offset its lags | no switch yet: needs one | measurement only | code |
| AS-197 | "All instances" of the testbed gate means every instance with a reference, largest band included | `--without NAME[,NAME]` in gpu_presolve/analysis/bench240_stage1c.py prints a further line | measurement only | taken as recommended |
| AS-198 | No instance may end larger than with the Stage 1 engine | `BEFORE_BENCH` in gpu_presolve/analysis/bench240_stage1c.py:66 names the results of the Stage 1 engine (tag v4pre) to compare with | measurement only | design |
| AS-199 | The time rule is 1.2 times HiGHS plus 0.02 s, minimum of three alternating runs | the constants 1.2 and 0.02 in gpu_presolve/analysis/stage1_time_gate.py:96-98, stage1_compare.py:196 and overheads.py:163 | measurement only | the owner |
| AS-200 | What the clock covers on each side; warm times; a shifted mean with 0.1 s | the constant `SHIFT = 0.1` in the two scripts | measurement only | taken as recommended |
| AS-201 | The time reference is HiGHS on the same rule set, with its own inefficiencies | the HiGHS switch set (`HIGHS_DIAG_OFF` and `--presolve-rule-off`) of the timing scripts gpu_presolve/baseline/stage1_timing_runs.sh and bench240_timing_runs.sh | measurement only | design |
| AS-202 | Times come from a shared node and are compared only within one machine | `--jobs 1` and `--reps 1` per pass in the timing scripts | measurement only | the owner / design |
| AS-203 | The engine is timed in an -O2 build, HiGHS in its CMake Release build | no switch yet: needs one | measurement only | code |
| AS-204 | A new engine version is timed against the one before, with a mark at 3% | `--limit PERCENT` in stage1c_time.py (default 3) | measurement only | design |
| AS-205 | With its new rules off, a version must give the results of the version before | `GPRS_OFF=<families or sub-rules>` for the runs | measurement only | design |
| AS-206 | Each milestone of the nine rules has named acceptance instances | `options_t::merge_integers` for the two settings of the table of the merge of parallel columns [M5] | planned | design |
| AS-207 | The cheap-rule references are size runs, not timing runs | `--jobs` in the reference script | planned | design |
| AS-208 | Overheads are tracked in every run and optimised only when they block beating HiGHS | no switch yet: needs one | measurement only | the owner |
| AS-209 | Sizes are compared as rows, columns and nonzeros; added nonzeros count against the measure | no switch yet: needs one | planned | taken as recommended |
| AS-210 | Objective nonzeros are not part of the size measure | no switch yet: needs one | planned | code |
| AS-211 | HiGHS sizes are read from two sources: the DIAG total line and the returned model | no switch yet: needs one | measurement only | design / code |
| AS-212 | The presolve-only report inside cuOpt approximates how the solver calls the presolve | environment variables `GPRS_REDUCTIONS_LIST`, `GPRS_REDUCTIONS_THREADS` (default 4), `GPRS_REDUCTIONS_MODES` (default "023") | measurement only | design |
| AS-213 | "cuOpt as is" means PaPILO as cuOpt configures it, not stock PaPILO | `presolve_gprs` = 0 with cuOpt's defaults | in cuOpt as is | code |
| AS-214 | A cluster arm is a flipped-default commit, run once; the baseline has no stored run | the edited defaults `presolve_gprs` and `mip_scaling` in a benchmark commit | planned (two baseline commits and two mode 2 commits prepared, none submitted) | design / the owner |
| AS-215 | Acceptance is a cluster A/B with a threshold set before the run; none is given | `presolve_gprs` 0 against 3 | planned | design |
| AS-216 | The end test turns one switch at a time; interactions between switches are not tested | no switch yet: needs one | measurement only | design |
| AS-217 | The quality measures named so far are on the primal side only | no switch yet: needs one | measurement only | taken as recommended |
| AS-218 | Quality against HiGHS is to be measured by cuOpt, presolve off, on HiGHS-presolved models | the parameter `presolve` = 0 on a presolved MPS file | planned | taken as recommended |
| AS-219 | The probing gate has no threshold; the clique gate rests on eight instances | no switch yet: needs one | planned | taken as recommended |
| AS-220 | The two arms of the main comparison differ in presolve seconds as well as reductions | no switch yet: needs one | measurement only | design |
| AS-221 | Correct means structure check, crush check and solver check pass; solver quality never decides | flags of gpu_presolve/check/run_check.py: `--primal-only`, `--lp-time-limit`, `--mip-time-limit`, `--require-points`, `--cost-variants`, `--push-points` | measurement only | design |
| AS-222 | The correctness checks pass violations up to their own tolerances | the constants of the table, such as `structure_tolerance`, `point_tolerance` and `check_tolerance` (check headers and oracle) | measurement only | design |
| AS-223 | The solver-based check uses HiGHS with short limits, below 1e7 nonzeros only | `--lp-time-limit`, `--mip-time-limit`, `--threads`, `--solution-time-limit`, `--max-points` of run_check.py | measurement only | design |
| AS-224 | On the 240 a known objective value is a bound, not the optimum | `--known-is-bound` of gprs_check (set by run_check.py when the testbed value ends with a star) | measurement only | design |
| AS-225 | Each stored point gives up to four pushed points; instances get other cost vectors too | `--push-points`, `--cost-variants`, `--max-points` of run_check.py | measurement only | design |
| AS-226 | A check counts only after it failed on a deliberately wrong engine | make_mutants.py `<out dir> [name ...]` | measurement only | design |
| AS-227 | Every rule change needs a random campaign of tens of millions of models | gprs_fuzz `<first seed> <seeds per profile> [--singletons] [--profile <name>]` | measurement only (the tool is in the repository) | design |
| AS-228 | The check extensions designed for the nine rules accept what the rules' tolerances allow | no switch yet: needs one | designed (milestones of the singleton-column rules and the forcing column [M1] and of the merge of parallel columns [M5]) | design |
| AS-229 | Cluster builds carry no engine assertions | the CMake option `DEFINE_ASSERT` of the cuOpt build | measurement only | taken as recommended |
| AS-230 | Before a cluster run the adapter passes a pre-flight: 40 models, 30 s per mode | the arguments of preflight.sh (output directory, seconds, parallel runs, instances) | measurement only | design |
| AS-231 | The working tree reads the adapter mode from the environment through an uncommitted patch | the environment variable `GPRS_PREFLIGHT_MODE` in a build with the patch | measurement only | design |
| AS-232 | The adapter's unit tests cover four small instances and no refusal, limit or unbounded path | no switch yet: needs one | measurement only | design |
| AS-233 | Finding: the Stage 1 engine is at or above HiGHS with the same rule set | no switch of its own: the finding is reproduced with gpu_presolve/analysis/stage1_compare.py | measurement only | design |
| AS-234 | Finding: what each presolve leaves on the 240; belief that fewer nonzeros mean better quality | `presolve_gprs` for the test | measurement only | design |
| AS-235 | Finding: HiGHS spends most of its presolve time in probing and enumeration for little size | `probing`, `enumeration`, `HIGHS_DIAG_OFF` | measurement only | design |
| AS-236 | Finding: four cheap rules carry the stage; the textbook column rules are worth nothing measurable | in the engine `options_t::sub_enabled` and `options_t::enabled` hold the switches of these rules (no rule reads them yet) | measurement only | design |
| AS-237 | Finding: the five further rules are worth about 0.06% in nonzeros on the 240 | no switch yet: needs one | measurement only | design |
| AS-238 | Finding: the rule ladder, and where the engine stands before the column rules | no switch of its own: a measurement of HiGHS reference runs (gpu_presolve/baseline/stage1c_reference_runs.sh) | measurement only | design |
| AS-239 | Finding: two large instances carry the testbed distance, and the two sets rank rules differently | no switch of its own: a measurement | measurement only | design |
| AS-240 | Finding: the zero-cost singleton rule followed by the lock rule reaches neos-3402294-bobin | `options_t::sub_enabled[sub_rule::zero_cost_singleton]` in the engine (no rule reads it yet) | measurement only | design |
| AS-241 | Belief: the nine rules reach the gate if the engine's forms reach what HiGHS's reach | no switch yet: needs one | measurement only | design |
| AS-242 | Finding: part of the gap to HiGHS has no single-rule attribution | no switch of its own: recomputed from the HiGHS runs with one rule off (`HIGHS_DIAG_OFF` and `--presolve-rule-off`, with the names and rule bits of AS-188) | measurement only | design |
| AS-243 | Finding: in Stage 1 the time is in building the state, not in propagation | no switch of its own: a measurement, repeated on every run by gpu_presolve/analysis/overheads.py | measurement only | design |
| AS-244 | Belief: below 500,000 nonzeros the GPU wins only on wide work; cascades stay on host | no switch yet: needs one | measurement only | design |
| AS-245 | Belief: a surplus over HiGHS comes from where HiGHS limits itself, and from cliques | no switch in the engine yet: needs one when probing is built | measurement only (one of three sources measured) | design |
| AS-246 | Finding: what HiGHS's rule families are worth on the 31 (Stage 0) | on the HiGHS side the names of `HIGHS_DIAG_OFF` | measurement only | design |
| AS-247 | Finding: what each rule family of the engine is worth in nonzeros | `options_t::enabled` (one flag per family) | measurement only | design |
| AS-248 | Finding: cuOpt's four own reductions look almost worthless by nonzero count | `presolve_gprs` 3 against 2 | measurement only | design |
| AS-249 | Finding: the engine is 6.9 (the 40) and 8.4 (the 240) times faster than HiGHS | no switch of its own: a measurement of the timing scripts gpu_presolve/baseline/stage1_timing_runs.sh and bench240_timing_runs.sh | measurement only | design |
| AS-250 | Finding: HiGHS's slow row deletion on rd-rplusc-21 was a hypothesis that a later run weakened | no switch of its own: a finding about HiGHS | measurement only | design |
| AS-251 | Finding: what cuOpt's probing and clique table do today (read from source on 2026-11-26) | no switch of its own: a reading of cuOpt's source | measurement only | design |
| AS-252 | Belief: every row-dual bound HiGHS uses is a function of the present state | no switch in the engine: no dual state is kept (AS-301) | measurement only | design |
| AS-253 | Belief: the exchange rules are valid; evidence is exact tests of tiny models and reading | no switch of its own: evidence, not a setting | measurement only | design |
| AS-254 | Belief: a sequential engine reaches more than pure batch application | no switch yet: needs one | in the engine (the host engine is sequential) | design |
| AS-255 | Finding: HiGHS's own result depends on where it was built | no switch of its own: a finding about builds | measurement only | design |
| AS-256 | Finding: a HiGHS counter names the test that took a fixing, not the rule needed | no switch of its own: a rule for reading HiGHS's counters | measurement only | design |
| AS-257 | Inferences about reach that no engine run has confirmed, and unknowns left open | the designed `integer_sources` (not in code) for the unknown source types | measurement only (inference) | design |
| AS-258 | Three numerical defects of the base engine on models with extreme constants are open | no switch yet: needs one | in the engine | code |
| AS-259 | Two oracle failures of 1.1e-9 are untriaged: dual fixing at an outward-moved bound | no switch yet: needs one | in the engine (base commit) | code |
| AS-260 | A column can be fixed at any finite bound, however large | no switch yet: needs one | in the engine | code |
| AS-261 | The error bound of an activity is a first-order estimate, not a proven bound | no switch yet: needs one | in the engine | design |
| AS-262 | Doubleton equations are substituted without a numerical acceptance test or a fill limit | no switch yet: needs one | in the engine | code |
| AS-263 | The alias forest accepts an integer map without a drift guard | no switch yet: needs one | in the engine | code |
| AS-264 | Parallel rows are found through a hash on 20 mantissa bits, blind at rounding boundaries | no switch yet: needs one | in the engine | design |
| AS-265 | A free singleton chain's growth factor does not follow a column through an alias | no switch yet: needs one | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-266 | The zero-cost singleton rule turns equations into ranged rows | `options_t::sub_enabled[sub_rule::zero_cost_singleton]` | designed (milestone of the singleton-column rules and the forcing column [M1]) | design |
| AS-267 | No reduction looks at what it does to the integrality of the objective | no switch yet: needs one | in the engine (doubleton substitution) | design |
| AS-268 | After the singleton rules the parallel rows hash many rows again; that cost is unmeasured | no switch yet: needs one | designed | design |
| AS-269 | Each column family keeps its own row list; a switched-off family still pays for it | no switch yet: needs one | in the engine (merged into feat/gpu-presolve as `6eed652cf` and `74350acb3`) | taken as recommended |
| AS-270 | Implementation starts from designs with known open points | no switch of its own: each open point is binding for its milestone and has a fixed test or a wrong engine named with it (stage1c_design/v2_amendments.txt, lowworth_v2_amendments.txt) | designed (binding per milestone) | taken as recommended |
| AS-271 | The presolve can end without a fixed point, and the reason is not reported | no switch yet: needs one | in the engine | design |
| AS-272 | The limit on row dynamism is stated but not measured for the engine | no switch yet: needs one | planned (not enforced in the scripts) | design |
| AS-273 | Instances with indicator constraints have no standalone baseline | no switch yet: needs one | measurement only | taken as recommended |
| AS-274 | The held-out set is no longer unseen: 23 of its 36 are among the 240 | no switch of its own: the list `heldout` of gpu_presolve/testbed/testbed.json against gpu_presolve/testbed/benchmark240.json | planned | design |
| AS-275 | The checks cannot see a restriction that cuts no tested point | `--push-points` (at most 4 per point) and `--cost-variants` of gpu_presolve/check/run_check.py widen what is tested | measurement only | design |
| AS-276 | Instances without a usable point get no crush check | `--require-points` of run_check.py | measurement only | design |
| AS-277 | The structure check cannot prove the dual restrictions | no switch yet: needs one | designed | design |
| AS-278 | The exact oracle decides only a small class of models and sees no tolerance | `generator_options_t`, `p_singleton_col`, `--singletons`, `p_parallel_row`, `p_parallel_col` | measurement only | design |
| AS-279 | The checks' proofs cover the rules that exist; a new rule must bring its part | no switch of its own: each new rule brings its check extension | designed (per milestone) | design |
| AS-280 | Real instances are checked with default settings only | no switch yet: needs one | planned | code |
| AS-281 | The engine inside cuOpt differs from the standalone engine on 2 of the 240, unexplained | no switch yet: needs one | open | code |
| AS-282 | The adapter's conversions cost several times the engine; tracked, not optimised | no switch yet: needs one | in the adapter | the owner |
| AS-283 | No memory figure of the engine is measured or gated | no switch yet: needs one | measurement only (open) | design |
| AS-284 | Known cuOpt defects can hit both arms; one assertion failure is unexplained | no switch yet: needs one | in cuOpt (base) | the owner |
| AS-285 | A possible defect in cuOpt's probing cache is noted, not confirmed | no switch yet: needs one | not confirmed | design |
| AS-286 | Early stored results carry no identity, and some step markers were written by hand | no switch of its own: a defect of record keeping | measurement only | code |
| AS-287 | Statements about PaPILO were not read in its source | no switch of its own: a gap of reading | measurement only | design |
| AS-288 | The same model and settings give the same reduced model on every machine | `options_t::phase_hashes` (default false) records a hash after every phase, and `--reps` of gpu_presolve/tools/run_engine.py fails on unequal hashes | in the engine (host) | the owner / taken as recommended |
| AS-289 | Every tie goes to the smaller index; the reduced model depends on the input order | no switch yet: needs one | in the engine for Stage 1 | design |
| AS-290 | A row activity has one summation order, fixed by entry position | the constants `max_light_row` = 64 and `max_activity_lanes` = 256 (gpu_presolve/include/gprs/numerics.hpp:139-155) | in the engine (host) | the owner |
| AS-291 | Activities are compensated sums, updated incrementally, kept without a measured benefit | no switch yet: needs one | in the engine | the owner |
| AS-292 | Counts of infinite terms are exact, and one infinite term still gives one bound candidate | no switch yet: needs one | in the engine (host) | design |
| AS-293 | No floating-point atomics and no contracted floating-point operations | `-ffp-contract=off` | in the engine and the build | the owner / design |
| AS-294 | The engine is compiled with other flags inside cuOpt than standalone; equal results are assumed | no switch yet: needs one | in the build | code |
| AS-295 | Fixed columns stay in their rows as constants until a rebuild | `options_t::rebuild_dead_fraction` and `options_t::rebuild_fixed_fraction` (both default 0.25) | in the engine | design |
| AS-296 | The state cannot add a row or a column, and a row cannot grow | no switch yet: needs one | in the engine | design |
| AS-297 | The engine's model has continuous and integer columns only and keeps the input order | no switch yet: needs one | in the engine | design |
| AS-298 | Postsolve keeps few kinds of records and rounds integer columns before the undo | no switch yet: needs one | in the engine | design |
| AS-299 | A point is crushed by dropping removed columns; for the column rules by forward replay | `options_t::primal_only` decides which promise holds | in the engine (dropping) | design |
| AS-300 | No column changes type until the implied-integer conversion, which is the first type change | `implied_integer_equation`, `implied_integer_weak`, `implied_integer_scaling` | designed (milestone of the implied-integer conversion [L2]; parts are due with the singleton-column rules [M1] and with the merge of parallel columns [M5]) | design |
| AS-301 | No bounds on row duals are kept as state | no switch yet: needs one | designed | design |
| AS-302 | Engine and cheap rules run on one host thread; inside cuOpt nothing uses the GPU | no switch yet: needs one | in the engine and the adapter | design / the owner |
| AS-303 | The prices of the dominated-column rule are cached for one pass | no switch yet: needs one | designed (milestone of the dominated column [L5]) | design |
| AS-304 | The device builds the state with the host's bits; integration comes after the cheap rules | no switch yet: needs one | designed | the owner / design |
| AS-305 | Device bound propagation is postponed, ships off, and need not agree with the host | `options_t::device` (designed `device_options_t` with `engine` = `host`, `automatic`, `device`) | designed (Stage 1b) | design |
| AS-306 | The engine of a phase is chosen by a work estimate against provisional thresholds | no switch yet: needs one | planned (device engine and wide phases) | the owner |
| AS-307 | A device threshold is fixed once, at the smallest summed time over the testbed | no switch of its own: a rule for setting the defaults of the designed `device_options_t` (AS-305) | designed (milestone 7 of Stage 1b) | design |
| AS-308 | Memory: pinned mirror on x86, managed memory on Grace Hopper, unmeasured until the end | no switch yet: needs one | planned | the owner |
| AS-309 | The device matrix is binned by row and column length, partly without a measurement | no switch yet: needs one | planned (device) | the owner |
| AS-310 | GPU batches are made independent by a reservation table; a losing candidate is dropped | no switch yet: needs one | planned | design |
| AS-311 | The planned probing launches many sources at once that do not see each other's results | no switch yet: needs one | planned (step 4 of the build order) | design |
| AS-312 | The clique pipeline replaces the paper's random choices by fixed rules | no switch yet: needs one | planned | design |
| AS-313 | The presolve is a standalone library with no cuOpt code and no PaPILO inside | `presolve_gprs` inside cuOpt: mode 1 runs PaPILO after the engine and mode 2 the engine alone | in the repository | the owner |
| AS-314 | The standalone code follows the semantics of three open cuOpt pull requests | no switch yet: needs one | in the engine | design |
| AS-315 | Postsolve and crush of the engine are called from several threads at once | no switch yet: needs one | in the adapter | design |
| AS-316 | Storage housekeeping follows fixed thresholds | the named constants `min_moved_capacity` = 4, `sort_share` = 32 and `comparison_sort_size` = 64 | in the engine | code |

## 1. Reductions held back, restricted or not yet built (candidates for a stronger formulation)

### AS-001 The general merge of two parallel integer columns is held back (`range_kept`)
- **Assumption:** Two parallel columns that are both integer are merged only when the merged column gets the range of the kept column back in the same step, so that two binaries stay one binary. The merge into a wider general integer column, which HiGHS makes, is off. Two continuous columns, and an integer column merged into a continuous one, are merged with every setting.
- **Set by:** Taken as recommended (the owner was asked and did not answer, HISTORY Entry 35; listed as taken in Entry 40 and in DESIGN.md amendment A8: "Not decided by the owner, taken as recommended until they say otherwise"). The owner named it for the end test on 2026-11-27 (Entry 45): "strengthening might mean adding those mergings". What was reported to the owner in HISTORY Entry 35 was a switch that is off inside cuOpt; version 2 of the design changed that to `range_kept` as the default everywhere, cuOpt included, and the adapter passes no option (AS-152). Inside cuOpt integer pairs will therefore be merged whenever the merged column gets the kept range back; the owner has seen neither form decided.
- **Where:** DESIGN.md amendment A8 (lines 147-151) and section 4 (lines 523-524); stage1c_design/v2_design.txt:277 (decision 2); v2_amendments.txt:84-86; v2_design.txt:84, :87, :216, :271; v2_answers.txt:131-134; v2_reviews.txt:77; gpu_presolve/include/gprs/presolve.hpp:64-70 and :127; HISTORY Entry 35 (lines 2064-2067) and Entry 40 (lines 2211-2213).
- **Switch:** `options_t::merge_integers` = `never`, `range_kept` (default), `always`; no rule reads it yet, the tool gprs_presolve has no flag for it and the adapter does not pass it (AS-152).
- **Alternatives:** `always` (every pair whose merged values leave no hole becomes one general integer column, as in HiGHS); `never`; `never` inside cuOpt only, which is what HISTORY Entry 35 reported to the owner.
- **May change:** `always` removes more columns and takes away interchangeable solutions, but the merged column has a range of 2 or more: it is no clique member, no binary to probe on, and no candidate for simple probing, the single-lock alias, GF2, indicator strengthening or block BVE. With `range_kept`, by HiGHS's counters on 237 of the 240, about 0.0006 of the mean log in nonzeros and 0.0010 in columns stay behind (neos-4722843-widden at 0.917 in nonzeros and 0.829 in columns of the reference, fiball at 0.983). Neither harm nor benefit to the solve is measured: that a general integer made from two binaries hurts cliques, probing and cuOpt's heuristics is a belief of the design, and the size figures are an attribution from HiGHS's counters, not an ablation (rail507, fast0507 and seymour are reached with `range_kept` because HiGHS's own bound strengthening takes 807 of 828 merged columns back to [0,1]). Interchangeable columns that stay unmerged are left to cuOpt's symmetry handling (AS-143).
- **Test at the end:** Standard cluster comparison in three arms: `merge_integers` = `range_kept`, `always`, `never`; the three arms once more with cuOpt's symmetry detection off (`mip_symmetry` 0, AS-143). Before it, the structure check and the crush check on the 240 with `always` (AS-280), and the count of findings of simple probing and of the single-lock alias that change with `always`, which the design asks for when the merge is built.
- **State:** designed (milestone of the merge of parallel columns [M5]); the option is in the engine since the groundwork (HISTORY Entry 46) and no rule reads it yet; the rule is not built.

### AS-002 `range_kept` needs dual bound strengthening, so no integer pair is merged in primal-only mode
- **Assumption:** The setting `range_kept` computes the bound of the merged column with dual bound strengthening before the merge. In primal-only mode that rule is off, so no pair of integer columns is merged there, and the result depends on the switch of dual bound strengthening.
- **Set by:** Design (v2_design.txt:83).
- **Where:** stage1c_design/v2_design.txt:83; gpu_presolve/include/gprs/presolve.hpp:168-177 (`sub_rule_active`).
- **Switch:** `options_t::merge_integers` together with `options_t::primal_only` and `options_t::sub_enabled[sub_rule::dual_bound]`; whether `dual_bound` off also stops the `range_kept` merges is not stated in the design.
- **Alternatives:** A range test that needs no dual rule (a clique or a row bound that caps the sum of the pair).
- **May change:** In primal-only mode, which cuOpt uses with a solution callback (AS-154), integer twins stay unmerged.
- **Test at the end:** Standard cluster comparison with primal-only mode forced in both arms: `merge_integers` = `never` against `always` (`range_kept` equals `never` there).
- **State:** designed (milestone of the merge of parallel columns [M5]).

### AS-003 Indicator strengthening stays on and grows the model; its native port is not built
- **Assumption:** cuOpt's indicator strengthening does not read indicator constraints of an MPS file: it looks for ordinary two-entry rows x - z <= 0 between binaries (side exactly 0, coefficients exactly +1 and -1), then appends implied rows for disjunction rows over such members and lifts capacity rows. It is kept in the arm with the new presolve (adapter mode 3) although it adds nonzeros, which no size gate credits. It runs through the PaPILO framework; the port into the new presolve, a first phase that appends rows, is planned.
- **Set by:** Design (keeping it on through mode 3 is DESIGN.md amendment A6; `true` is cuOpt's existing default). The owner named it as a strengthening candidate on 2026-11-27 (HISTORY Entry 45): "additional constraints like indicator".
- **Where:** DESIGN.md amendment A6 (table line 72) and amendment A10 (lines 167-170); cpp/include/cuopt/mathematical_optimization/mip/mip_hyper_params.hpp:38-40; cpp/src/presolve/third_party_presolve.cpp:918-922; cpp/src/presolve/indicator_strengthening.cpp:25-30, 78-110, 409-414; HISTORY Entry 37.
- **Switch:** `mip_hyper_params_t::presolve_indicator_strengthening` (default true; parameter `mip_hyper_presolve_indicator_strengthening`); in mode 3 also the allowlist name `indicatorstrengthening`.
- **Alternatives:** Off; the native port, which could run before the cascade on the original model (AS-139).
- **May change:** More rows and nonzeros and a tighter LP relaxation on models with indicator structure. Measured without a solver on the 240 (mode 3 against mode 2, HISTORY Entry 37): cuOpt's four reductions add nonzeros on ns1116954 (27%), ex10 (20%), ex9 (12%) and physiciansched6-2 (3%), which the entry attributes to the rows indicator strengthening appends.
- **Test at the end:** Standard cluster comparison: `presolve_indicator_strengthening` true against false, once in mode 3, where the rule sees the model the new presolve left (AS-139), and once in mode 0, where it sees the original model. Both pairs are read on the instances where the rule reports appended or lifted rows, with the root relaxation objective and the final bound next to the primal measures. Its counts are a debug log line (indicator_strengthening.cpp:409-414), so the arms need a build or a log level that prints it.
- **State:** in cuOpt as is (reached in modes 0, 1 and 3; in mode 2 only when the model falls back to PaPILO, AS-156); the port is planned.

### AS-004 The zero-cost singleton rule and the forcing column are off in primal-only mode
- **Assumption:** In primal-only mode the zero-cost singleton rule [R3] and the forcing column [R4] are switched off together with the four reductions that rest on a lock or cost argument, although both are exact projections that keep every feasible point with its objective. Only the free singleton column with an equation as pivot row and the merge of parallel columns stay on. The design recommends "on"; until the owner decides they are off.
- **Set by:** Taken as recommended (HISTORY Entry 40: "R3 and R4 stay off in primal-only mode"; v2_amendments.txt:88; DESIGN.md amendment A8). Entry 45 makes it an entry of this register.
- **Where:** DESIGN.md amendment A8 (lines 151-152) and section 8 (lines 698-700); stage1c_design/v2_design.txt:279 (decision 4); gpu_presolve/include/gprs/presolve.hpp:168-177. The nine-rule design calls the forcing column a lock rule "by the plan"; the plan's table (DESIGN.md:505) lists forcing columns among the dual reductions. Both classes are off in the mode.
- **Switch:** no switch yet: needs one; `sub_rule_active` hard-codes that only `free_singleton` and `parallel_merge` are active in primal-only mode (`options_t::sub_enabled` can only switch a reduction off).
- **Alternatives:** Both on in primal-only mode, at the price of one exemption in the round-trip clause of the crush check.
- **May change:** Only solves in primal-only mode: rows with slack-like singleton columns and free zero-cost columns stay in the model. Measured on HiGHS on the 240, the zero-cost singleton rule is worth 1.0064 in columns.
- **Test at the end:** Standard cluster comparison with primal-only mode forced in both arms: the two rules off against on, after the crush check has shown that every feasible point still crushes with equal objective. A default-mode run cannot show it.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]); the off state is coded in the engine (`sub_rule_active`), the two rules are not built.

### AS-005 The longest cliques of knapsack and inferred set-packing rows are not added as model rows
- **Assumption:** The clique paper (Dai and Chen, 2025) adds the longest cliques it finds to the model as new rows, up to a doubling of the rows, and that is where its gain comes from. The plan does not: it only replaces set-packing rows by cliques that contain them and removes dominated clique rows, which is parity with HiGHS. The plan says the cluster run with the new probing and cliques decides whether to add them.
- **Set by:** Design (DESIGN.md section 7; HISTORY Entries 3 and 4). It falls under the owner's "additional constraints" of Entry 45.
- **Where:** DESIGN.md section 7 (lines 680-685).
- **Switch:** no switch yet: needs one.
- **Alternatives:** Add the clique rows to the model; pass them to cuOpt's clique table or cut pool only.
- **May change:** More rows and nonzeros, a stronger LP bound on models with knapsack or conflict structure, more LP time. The paper's gain (66.8% of the nodes) was measured with Gurobi 11 on the 99 of its 176 instances that have such rows (HISTORY Entry 3).
- **Test at the end:** Standard cluster comparison when built: clique rows added to the model against not added; a third arm exports them to cuOpt's clique table.
- **State:** planned, off.

### AS-006 Clique merging is built with the probing stage and gated apart from probing
- **Assumption:** Extension of set-packing rows and removal of dominated clique rows are built in the last step of the build order and get their own gate. Basis on the 31: HiGHS's merging changes rows on 14 instances; without it HiGHS keeps 4.6% more nonzeros in the geometric mean (0.7% without sorrell3, 0.02% without sorrell3 and savsched1).
- **Set by:** The owner (HISTORY Entry 13, 2026-11-26: "2: build it").
- **Where:** DESIGN.md section 0 decision 2 (line 189) and section 7 (lines 670-678); gpu_presolve/results/stage0/STAGE0_REPORT.md lines 95-98, 237-241, 370-377.
- **Switch:** no switch yet: needs one; on the HiGHS side `clique_merging` in `HIGHS_DIAG_OFF`.
- **Alternatives:** Leave clique merging to cuOpt's clique table; export cliques only.
- **May change:** Clique extension lengthens set-packing rows (on sorrell3 from 2 entries to a median of 37): nonzeros grow while the LP relaxation tightens.
- **Test at the end:** Before it is built: cuOpt with its presolver set to none on the 240 as presolved by HiGHS with `clique_merging` on and with it off (the family pairs of AS-113). When built: standard cluster comparison, clique merging off against on, with extension and removal of dominated rows each on its own switch.
- **State:** planned (step 4 of the build order).

### AS-007 Clique export to cuOpt, clique-aware propagation and knapsack cliques are undecided
- **Assumption:** Only clique merging is decided. Export of cliques to cuOpt's clique table, clique-aware propagation and cliques from knapsack rows are listed for the probing and clique stage, but whether they are built is open; Stage 0 has no data on them.
- **Set by:** Design (open point 3 of DESIGN.md section 0, to be decided with the approval of that stage).
- **Where:** DESIGN.md section 0 (lines 215-218) and section 7 (lines 661-662, 675-678).
- **Switch:** no switch yet: needs one for each of the three.
- **Alternatives:** Build each of the three; build none.
- **May change:** Exported cliques feed cuOpt's cuts and propagation without changing the size of the model: a strengthening without a size effect.
- **Test at the end:** Standard cluster comparison when built, one pair each: clique export, clique-aware propagation, knapsack cliques, off against on.
- **State:** planned, undecided.

### AS-008 A clique row is replaced or removed only by a clique in the model
- **Assumption:** A set-packing row is replaced only by a clique that contains it, and removed only when a containing clique is a model row at the end of the same batch. A clique that exists only in the clique table or the cut pool never justifies a removal; for an equality clique row the extension literals are fixed to zero and the row is not changed.
- **Set by:** Design (guard from the design reviews, DESIGN.md section 4).
- **Where:** DESIGN.md section 4 (lines 544-548).
- **Switch:** no switch yet: needs one.
- **Alternatives:** Remove rows dominated by table cliques (a smaller model that is correct only while the solver keeps the table).
- **May change:** Fewer rows removed than a table-based rule; the reduced model stays self-contained.
- **Test at the end:** Covered by the pair "clique merging off against on" (AS-006); a pair of its own only if the table-based variant is built.
- **State:** planned.

### AS-009 Coefficient lifting from probing is not in the plan
- **Assumption:** The planned probing harvests fixings, tightened bounds, aliases, implications and variable bounds. HiGHS's probing stage does one thing more: a probe stores lifting opportunities and the stage applies them to rows ("lifting for probing"), which changes coefficients and can add entries to a row. Neither the plan nor a design names it, so it is left out without a decision.
- **Set by:** Design (an omission of the plan's probing section; never put to the owner).
- **Where:** DESIGN.md section 7 (lines 619-632; the harvest is lines 627-628); gpu_presolve/baseline/HIGHS_DIAG.md:102 and :234-237; gpu_presolve/baseline/highs_diag.patch:2357-2358 (`implications.storeLiftingOpportunity`). Not read: by which rule HiGHS lifts, and whether HiGHS 1.16.0 does it with default options (the HiGHS source is not on this machine).
- **Switch:** no switch yet: needs one. The diagnostic patch has none either: lifting sits inside the stage that `probing` in `HIGHS_DIAG_OFF` switches as a whole.
- **Alternatives:** Build it with the probing stage (it needs a row that can grow, AS-296); export the opportunities to cuOpt without changing rows; leave it out for good.
- **May change:** Rows with binaries get tighter coefficients or further entries: a stronger LP relaxation with the same number of nonzeros or more, which a size gate counts as nothing or as a loss.
- **Test at the end:** When probing is built: standard cluster comparison, lifting on against off, read on the root relaxation objective and the final bound next to the primal measures. Before that no pair exists for lifting alone; if HiGHS has an option for it, which is to be read in its source, it joins the HiGHS family pairs of AS-113.
- **State:** not planned.

### AS-010 Substitutions from cliques and implications wait for the probing stage, without a switch or gate
- **Assumption:** HiGHS replaces a column by another when its clique table shows two binaries to be equal or complementary (x = y, x = 1 - y) and when the two probes of a binary give y = a + b x. The plan has these, and an implication graph whose components over rows with two literals give equivalences and contradictions, in the last step of the build order together with probing. They have no design, no switch and no gate of their own; the probing gate compares sizes without a threshold (AS-219).
- **Set by:** Design (plan sections 5 and 7); the place in the order follows the owner's "Yes, cheap rules first" (HISTORY Entry 29).
- **Where:** DESIGN.md section 5 (line 561), section 7 (lines 627-628 and 687), amendment A8 (line 109), section 12 (line 772: the rule is off in every HiGHS reference used so far); gpu_presolve/baseline/HIGHS_DIAG.md:209-213 and :441.
- **Switch:** no switch yet: needs one per kind (clique substitution, implication substitution, equivalences of the graph); on the HiGHS side `clique_substitutions` in `HIGHS_DIAG_OFF`.
- **Alternatives:** Build the implication graph over two-literal rows early (it needs no probing); export equivalences to cuOpt without substituting; leave them to cuOpt's probing cache, which also finds substitutions (cpp/src/mip_heuristics/diversity/diversity_manager.cu:366).
- **May change:** Columns leave the model and a binary is replaced by another or by its complement, so the solver gets fewer branching candidates and rows with merged literals. In Stage 0, HiGHS's probing gave 12,230 fixings and substitutions on the 31, 90.9% of them on five instances (STAGE0_REPORT.md:228); the split between fixings and substitutions was not looked up.
- **Test at the end:** Before they are built: cuOpt with its presolver set to none on the 240 as presolved by HiGHS with `clique_substitutions` off against on (the family pairs of AS-113). When built: standard cluster comparison, one pair per kind, off against on.
- **State:** planned (step 4 of the build order), no design.

### AS-011 All of HiGHS's cheap rules are built; the five low-worth ones come after the nine
- **Assumption:** The cheap column rules stage builds every cheap rule HiGHS has. The five that are worth little in size (dominated and weakly dominated column by the bound on the reduced cost, implied equations, column stuffing, implied-integer conversion with column scaling, the single-equation extension of dual fixing) are required and come after the nine rules of the design. Switched off together in HiGHS on the 240 they cost 1.0006 in nonzeros, 1.0015 in rows and 1.0006 in columns, and 19 instances change.
- **Set by:** The owner, in three steps on 2026-11-27: "Yes, leave them out" (HISTORY Entry 35), "wait a sec. this is on testbed instances. i want to see the impact on all instance set" (Entry 36), "Build them too" (Entry 38). The assistant had recommended leaving them out.
- **Where:** DESIGN.md amendment A8 (lines 114-131); stage1c_design/v2_amendments.txt:7-13; lowworth_v2_design.txt:66-70; lowworth_v2_amendments.txt:1-8.
- **Switch:** no switch yet: needs one; designed are eight engine switches (`implied_equations`, `integer_sources`, `implied_integer_equation`, `implied_integer_weak`, `implied_integer_scaling`, `column_stuffing`, `single_equation`, `dominated_columns`, lowworth_v2_design.txt:70), none in code. On the HiGHS side: `implied_equations`, `implied_integer_conversion`, `dual_fixing_single_equation` in `HIGHS_DIAG_OFF` and rule bits 5, 19, 23.
- **Alternatives:** Leave them out; build only the one a measurement asks for.
- **May change:** Little in size (predicted for the engine: at most 0.0003 / 0.0009 / 0.0005 in the mean log of nonzeros / rows / columns on the 240). Beyond size they turn inequalities into equations, continuous columns into integer columns and rescale columns, which changes what the solver sees.
- **Test at the end:** Standard cluster comparison: all eight switches on against all off; then one pair per switch (AS-063, AS-065, AS-066, AS-067, AS-038, AS-039).
- **State:** designed (milestones of the five further rules [L1 to L5], which come after the nine column rules [M1 to M5]); nothing of it is in the engine.

### AS-012 Sparsify and the aggregator are built second and valued by their size effect
- **Assumption:** After the cheap column rules the plan builds sparsify (row combinations that cancel entries) and the aggregator. In HiGHS on the 240, sparsify removes 1.7% of an instance's nonzeros in the mean for 0.4% of the time; the aggregator removes 1.7% of the rows and adds 0.3% nonzeros.
- **Set by:** The owner for the order ("Yes, cheap rules first", HISTORY Entry 29); the numbers are from Entry 28.
- **Where:** DESIGN.md amendment A8 (line 107) and section 5 (line 566).
- **Switch:** no switch yet: needs one for each; on the HiGHS side `sparsify`, `aggregator` in `HIGHS_DIAG_OFF`.
- **Alternatives:** Leave sparsify out; an aggregator without fill-in.
- **May change:** Sparsify removes nonzeros without removing a row or a column, by adding equations to other rows. The rows keep the feasible set but change what a single row tells cuts and propagation: the clearest case of a rule that serves the size count.
- **Test at the end:** Before they are built: cuOpt with its presolver set to none on the 240 as presolved by HiGHS, `sparsify` on against off and `aggregator` on against off (the family pairs of AS-113). When built: standard cluster comparison, sparsify off against on; aggregator off against on.
- **State:** planned (step 2 of the build order).

### AS-013 Pairwise dominated columns are fixed only under strict conditions
- **Assumption:** A dominated column is fixed only when the dominating bound is infinite, or implied by a named row, or both columns are in a common clique row; an integer column never dominates a continuous one; only columns at odd depth of the domination forest are fixed.
- **Set by:** Design (guard from the design reviews, DESIGN.md section 4).
- **Where:** DESIGN.md section 4 (lines 525-527) and amendment A8 (line 108).
- **Switch:** no switch yet: needs one; on the HiGHS side `dominated_columns` in `HIGHS_DIAG_OFF`.
- **Alternatives:** HiGHS's conditions.
- **May change:** Possibly fewer fixings than HiGHS's pairwise dominated columns, which on the 31 fixed nothing in 41.9 of the 42.1 s they took (DESIGN.md section 5).
- **Test at the end:** Before it is built: cuOpt with its presolver set to none on the 240 as presolved by HiGHS with `dominated_columns` on and with it off (the family pairs of AS-113). When built: standard cluster comparison, pairwise dominated columns off against on.
- **State:** planned (step 3 of the build order).

### AS-014 Fourier-Motzkin, variable-bound aggregation and dual-fix probing are left out; enumeration is planned last
- **Assumption:** Fourier-Motzkin elimination, aggregation of variable lower and upper bounds and dual-fix probing are not built. Enumeration of small binary rows, left out in plan version 4, is now named in the last step of the build order. Evidence on the 31: Fourier-Motzkin fires on 3 instances, variable-bound aggregation removes 722 rows on one, enumeration takes 7.7% of HiGHS's time.
- **Set by:** Design (DESIGN.md section 5 on Stage 0 data; the new place of enumeration is DESIGN.md amendment A8).
- **Where:** DESIGN.md section 5 (line 567) and amendment A8 (line 109); HISTORY Entries 10 and 28.
- **Switch:** no switch yet: needs one; on the HiGHS side `fourier_motzkin`, `enumeration` in `HIGHS_DIAG_OFF`.
- **Alternatives:** Build each.
- **May change:** On the 240, enumeration removes 56 to 76% of the nonzeros on ex9, ex10, neos8 and sorrell3 (HISTORY Entry 28), and with enumeration off HiGHS does not reduce sorrell3 at all, because the clique merging that matters runs inside it (Entry 10). Without these rules such instances stay far from HiGHS.
- **Test at the end:** Standard cluster comparison when a rule is built. Before that: cuOpt with its presolver set to none on the 240 as presolved by HiGHS with the rule off against on (`fourier_motzkin`, `enumeration`; the family pairs of AS-113).
- **State:** enumeration planned (step 4); the other three not planned.

### AS-015 Removal of dependent rows and the split into components are not in the plan
- **Assumption:** The plan leaves out the removal of dependent rows and the split of a model into independent components. PaPILO has an option for each, and cuOpt switches them off only when duals are wanted, so for a MIP they run at PaPILO's defaults in the baseline arm.
- **Set by:** Design (DESIGN.md section 5, "Left out").
- **Where:** DESIGN.md section 5 (lines 588-589); cpp/src/presolve/third_party_presolve.cpp:781-784 (`componentsmaxint`, `detectlindep`), :925 (dual postsolve off for a MIP). PaPILO's defaults for the two options were not read: the pinned PaPILO is not on this machine.
- **Switch:** no switch yet: needs one in the engine when either is built. On the PaPILO side the options `componentsmaxint` and `detectlindep`, which cuOpt sets only for dual postsolve and exposes as no parameter.
- **Alternatives:** Build either in the engine; keep them as a PaPILO step after the engine.
- **May change:** Components can decide single instances in a solver and change the search, not only the size; dependent rows change the rank of the LP.
- **Test at the end:** Two pairs that need no engine work, in mode 0 with scaling off: cuOpt as is against `componentsmaxint` = -1, and cuOpt as is against `detectlindep` = 0, each as a benchmark commit that sets the option for a MIP (PaPILO's defaults are to be read first). They give what each is worth in cuOpt today. When built in the engine: standard cluster comparison, off against on.
- **State:** not planned.

### AS-016 Network implied integers, two-row bound tightening and two-column cancellation are not in the plan
- **Assumption:** The plan leaves out implied integers from network matrices, two-row bound tightening and two-column cancellation. It also leaves out multi-GPU, which is no reduction, and symmetry, which has its own entry (AS-143).
- **Set by:** Design (DESIGN.md section 5, "Left out").
- **Where:** DESIGN.md section 5 (lines 588-589). No switch under these names was found in HiGHS's diagnostic switch table (gpu_presolve/baseline/HIGHS_DIAG.md:430-453) or in cuOpt's list of PaPILO methods (cpp/src/presolve/third_party_presolve.cpp:725-759); whether HiGHS or PaPILO has such rules under other names was not read.
- **Switch:** no switch yet: needs one when one of them is built.
- **Alternatives:** Build any of them.
- **May change:** Two-row tightening and network implied integers give bounds and integrality without removing a nonzero, so a size gate would give them no credit.
- **Test at the end:** No pair is possible before one of them is built; then a standard cluster comparison, off against on. The main comparison (mode 0 against mode 3, AS-134) does not isolate any of them.
- **State:** not planned.

### AS-017 The engine has no dual postsolve
- **Assumption:** The postsolve restores column values only. Rows deleted as empty, redundant, singleton or forcing leave no record; a merged parallel row and a row subtraction leave a record that the primal postsolve ignores. No duals or reduced costs of the original model come back, so the engine serves MIP solves only.
- **Set by:** Design (DESIGN.md section 5 "Left out: dual postsolve" and section 8).
- **Where:** gpu_presolve/include/gprs/postsolve.hpp:17-20, 48-49, 56-59, 104-107; cpp/src/presolve/gprs_presolve.hpp:37; cpp/src/mip_heuristics/solve.cu:706 (`dual_postsolve = false`); cpp/src/presolve/third_party_presolve.cpp:1090 (only the MIP category reaches the adapter).
- **Switch:** no switch yet: needs one only if the engine is to presolve LPs.
- **Alternatives:** Records for deleted rows and a dual postsolve, which also restricts which reductions may run (with duals wanted cuOpt leaves out five PaPILO methods and its own single-lock rule, cpp/src/presolve/third_party_presolve.cpp:749-762).
- **May change:** Nothing for MIP solution quality.
- **Test at the end:** None over MIP quality; a design decision and a test of its own if the engine is ever used for LP.
- **State:** in the engine.

### AS-018 Gcd transforms, gcd substitution and variable-bound extraction have no place in the build order
- **Assumption:** The plan's parity and completion stages listed the reservation table, gcd transforms and gcd substitution, extraction of variable lower and upper bounds, cover strengthening and tuning. DESIGN.md amendment A8 replaces the order of the stages and names four steps; it does not say where these items go. This entry holds the three rules among them that change the model or what the solver is told; cover strengthening has its own entry (AS-019), the reservation table is AS-310, and tuning is no reduction.
- **Set by:** Design (a gap in the plan text, seen by reading DESIGN.md amendment A8 against sections 5 and 13).
- **Where:** DESIGN.md amendment A8 (lines 99-109) against section 5 (lines 565-566) and section 13 (lines 799-800); gpu_presolve/baseline/HIGHS_DIAG.md:539-542 (HiGHS's diagnostic switches do not touch the gcd substitution).
- **Switch:** no switch yet: needs one per rule; HiGHS's diagnostic patch has no switch for the gcd substitution either.
- **Alternatives:** Place each in one of the four steps, or drop it explicitly.
- **May change:** Extraction of variable bounds strengthens without a size effect (a by-product for the solver); the gcd rules change coefficients and columns of all-integer equations. Left unplaced they may never be built or tested.
- **Test at the end:** The placement has to be decided first. Extraction of variable bounds: tested with the export pair of AS-167 (export off against on). Gcd transforms and gcd substitution: no pair is possible before they are built, because HiGHS has no switch for them; when built, standard cluster comparison, off against on.
- **State:** planned without a place.

### AS-019 Cover strengthening is valued by nonzeros and has no place in the build order
- **Assumption:** HiGHS's inequality strengthening, which the plan lists as cover strengthening, lowers coefficients of inequalities and removes almost no nonzero. Stage 0 valued it by size: 144 nonzeros removed on the 31, and its 6.7 s counted among the stage runs "that removed nothing". The plan had it in the completion stage; the new build order does not place it (AS-018), and it is switched off in every HiGHS reference, so no gate misses it.
- **Set by:** Design (plan section 5; the valuation is from the Stage 0 report). It was not put to the owner as a strengthening candidate.
- **Where:** DESIGN.md section 4 (line 502), section 5 (line 566), section 12 (line 772), section 13 (line 800), amendment A8 (lines 99-109); gpu_presolve/results/stage0/STAGE0_REPORT.md:138-139 and :207-208; gpu_presolve/baseline/HIGHS_DIAG.md:98, :432 and :1055 (switch `strengthen_inequalities`). What the rule does inside HiGHS was not read: the HiGHS source is not on this machine.
- **Switch:** no switch yet: needs one; on the HiGHS side `strengthen_inequalities` in `HIGHS_DIAG_OFF`.
- **Alternatives:** Build it next to row tightening; leave it to BHW coefficient reduction, which covers one-sided all-binary rows of at most 12 entries (AS-033), and to cuOpt's cuts.
- **May change:** Coefficients of knapsack-like rows: a tighter LP relaxation at the same size. It can be slow: 62% of HiGHS's presolve time on hawaiiv10-130, a testbed instance above 1e7 nonzeros (STAGE0_REPORT.md:598).
- **Test at the end:** Before it is built: cuOpt with its presolver set to none on the 240 as presolved by HiGHS with `strengthen_inequalities` off against on and every other HiGHS rule the same (the family pairs of AS-113), read on the root relaxation objective and the final bound next to optimal count and primal gap. When built: standard cluster comparison, the rule off against on.
- **State:** planned without a place.

### AS-020 An integer column of range 1 is not moved to the bounds 0 and 1
- **Assumption:** When the presolve tightens an integer column to two neighbouring values that are not 0 and 1, the column stays as it is. HiGHS shifts such a column to a binary with lower bound 0. cuOpt marks a column as binary only when its bounds are 0 and 1, indicator strengthening reads binaries the same way, and cuOpt's own source carries a to-do for the shift. The plan names an integer shift only as a kind of postsolve record; no step of the build order holds it.
- **Set by:** Design (an omission: the plan's record list names "integer shift/scale" and nothing builds it).
- **Where:** DESIGN.md section 8 (line 693) and section 4 (line 506); gpu_presolve/baseline/HIGHS_DIAG.md:541; cpp/src/mip_heuristics/problem/problem.cu:945-961; cpp/src/presolve/indicator_strengthening.cpp:76-82. The engine has no column shift (searched gpu_presolve/include/gprs and gpu_presolve/src). What else in cuOpt reads the binary mark was not traced.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Shift in the engine with a record; shift in the adapter when the reduced model is written; leave it to cuOpt.
- **May change:** How many binaries cuOpt counts in the reduced model, and so how many columns its binary heuristics, probing and indicator strengthening work on. No size effect at all.
- **Test at the end:** First a count on the 240 of integer columns of the reduced model with range 1 and bounds other than 0 and 1; zero closes the entry for the benchmark set. Otherwise standard cluster comparison: shift off against on.
- **State:** not planned.

### AS-021 Row tests use one activity per side, over explicit bounds only
- **Assumption:** The engine keeps one activity sum per side, over the explicit bounds. HiGHS keeps a second sum over recorded implied bounds and tests row redundancy on it; the plan adds the second sum only if a gate shows it is needed. Measured with a HiGHS switch on the 31: testing redundancy on explicit-bound sums changes only unitcal_7 (1.1% more nonzeros) and supportcase7 (0.4%).
- **Set by:** Design, kept after a measurement (STAGE0_REPORT.md section 11: "The plan's choice for Stage 1 stands").
- **Where:** DESIGN.md section 2.4 (lines 430-433); gpu_presolve/results/stage0/STAGE0_REPORT.md lines 543 and 564-565; gpu_presolve/baseline/stage1_reference_runs.sh:85-87.
- **Switch:** no switch yet: needs one (there is no second sum); on the HiGHS side the environment variable `HIGHS_DIAG_REDUNDANCY_EXPLICIT=1`.
- **Alternatives:** A second sum over implied bounds with redundancy tested on it.
- **May change:** A few rows stay that HiGHS removes; when a redundant row leaves changes what other rules derive in the meantime. Not measured on the 240.
- **Test at the end:** First the HiGHS switch on the 240, presolve only. That run is a size estimate for HiGHS, not for the engine: the switch changes HiGHS's redundancy test only (HIGHS_DIAG.md:579-583). The engine-side test needs the second sum behind a switch; then a size run on the 240 with the hash of the reduced model at both values. Identical hashes on every instance close the entry; any difference sends it to the standard cluster comparison, read on the instances that differ. Until the switch exists the entry stays open.
- **State:** in the engine (single sum).

### AS-022 Every test of the nine column rules reads explicit bounds only
- **Assumption:** No rule of the nine reads a recorded implied bound of a continuous column; all activities are over explicit bounds. HiGHS reads implied bounds in simple probing and in dual bound strengthening.
- **Set by:** Design (v2_design.txt, state and driver item 1; listed among the "weaker forms" in its risks).
- **Where:** stage1c_design/v2_design.txt:28, :70, :99, :261.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Read valid recorded implied bounds; make bounds explicit earlier.
- **May change:** A row that balances only through an implied bound fires only in the rounds after the switch to explicit bounds, or not at all. Named instance: neos-1445765 (18,444 continuous columns substituted by HiGHS's simple probing); what it costs is marked unknown.
- **Test at the end:** First a size run at the milestones of simple probing [M2] and of dual bound strengthening and the single-lock alias [M4] (engine against HiGHS on neos-1445765 and the radiation instances). A standard cluster comparison needs a switch that lets the tests read recorded bounds.
- **State:** designed (milestones of simple probing [M2] and of dual bound strengthening and the single-lock alias [M4]).

### AS-023 A bound counts as implied only by one row read with explicit bounds; no chains
- **Assumption:** A column bound is taken as implied when one named row, read with the explicit bounds of its other columns, gives it; the proof is void when that row changes. Implied bounds are not chained through several rows, as HiGHS keeps them.
- **Set by:** Design (guard from the design reviews, DESIGN.md section 4).
- **Where:** DESIGN.md section 4 (lines 528-529); gpu_presolve/src/implied.hpp:33-40, gpu_presolve/include/gprs/rules.hpp:376-381.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Chained implied bounds.
- **May change:** Fewer implied-free columns, hence fewer free-singleton substitutions and dual reductions than HiGHS where the bound follows only through a chain.
- **Test at the end:** First the size comparison per instance against HiGHS with all cheap rules on the 240; a standard cluster comparison only if a chained variant is built.
- **State:** designed; the shared test is in the engine (`implied_tolerance`), and bound propagation already uses it.

### AS-024 Dual arguments use continuous columns, strict inequalities and one implied premise
- **Assumption:** By the plan, dual reductions use continuous columns only, strictly beyond the dual tolerance; the column bound that drops a side must be infinite or strictly implied; at most one implied premise is used per derivation; bounds on row duals are never carried across a structural batch. Each guard was introduced after a counterexample in review. The designs of the cheap rules depart from it in named places (AS-064, AS-068, AS-090).
- **Set by:** Design (DESIGN.md section 4, guards from the reviews).
- **Where:** DESIGN.md section 4 (lines 514-518); gpu_presolve/include/gprs/numerics.hpp:24 (`dual = 1e-7`).
- **Switch:** no switch yet: needs one per guard; the dual tolerance is `options_t::tol.dual` (AS-073).
- **Alternatives:** HiGHS's conditions where they are weaker.
- **May change:** Fewer dual reductions than HiGHS on models where the gain needs integer columns or chained implications.
- **Test at the end:** First the size comparison per instance against HiGHS with all cheap rules on the 240. Then a standard cluster comparison for any guard relaxed behind a switch, after the campaign.
- **State:** designed (rules of the milestones of the nine column rules and the five further rules [M1 to L5]).

### AS-025 An integer column is substituted only when the pivot row proves integrality
- **Assumption:** An integer column is substituted out only if all other columns of the pivot row are true integers and the ratios and the right-hand side are integral. Types are ordered integer above implied integer above continuous, and a representative that stands in for a true integer becomes a true integer.
- **Set by:** Design (guard from the design reviews, DESIGN.md section 4).
- **Where:** DESIGN.md section 4 (lines 519-522); gpu_presolve/src/alias_forest.cpp:39-64 and 149-164.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Leave integer columns alone; substitute them in mixed rows (integrality would be lost).
- **May change:** Fewer substitutions on rows that mix integer and continuous columns; when a continuous representative becomes integer the solver gets one more integer column.
- **Test at the end:** Covered by the off-against-on pairs of the doubleton, free-singleton and aggregator rules (AS-043, AS-050).
- **State:** in the engine for doubleton equations; designed for the rest.

### AS-026 Simple probing works only on equations and only on a binary of range exactly 1
- **Assumption:** Simple probing [R1] reads equations with three or more free entries only (two free entries are left to the doubleton rule; inequalities are not probed). The probing column must be a free integer column with upper minus lower exactly 1, every free column of the row must have finite bounds, and of several binaries the smallest column index is taken.
- **Set by:** Design (v2_design.txt:27, after HiGHS's rule "simple probing on equations").
- **Where:** stage1c_design/v2_design.txt:27; gpu_presolve/include/gprs/postsolve.hpp:28-30, gpu_presolve/src/rules_simple_probing.cpp:23-27 (empty).
- **Switch:** `options_t::enabled[rule::simple_probing]` (tool `--off simple_probing`) turns the rule; no switch yet for the restrictions.
- **Alternatives:** Probing on inequalities; an integer of larger range; the binary that gives the most aliases; full probing (step 4 of the build order).
- **May change:** Columns that an equation ties to a binary are replaced by it, the same structure in an inequality stays; which binary becomes the representative decides the branching candidates.
- **Test at the end:** Standard cluster comparison after the milestone of simple probing [M2]: simple probing on against off. A choice-of-binary pair would need a switch.
- **State:** designed (milestone of simple probing [M2]).

### AS-027 An inequality-pivot free singleton is taken only when its own cost gives the dual sign
- **Assumption:** The free singleton column with an inequality as pivot row [R5] fires only when the column has a nonzero cost that pushes it against a finite side of its only row. A row whose dual sign is known only through another column is left, because no row-dual state is kept (AS-301).
- **Set by:** Design (v2_design.txt:55-59).
- **Where:** stage1c_design/v2_design.txt:54-59; HISTORY Entry 42 (lines 2348-2352).
- **Switch:** `options_t::sub_enabled[sub_rule::free_singleton_dual]` turns the built case; no switch yet for the wider form.
- **Alternatives:** HiGHS's third pivot kind (dual bound from another column), with a compensator record.
- **May change:** Little expected: of HiGHS's 16,881 free column substitutions with all cheap rules on the 240, 11,596 have an equation pivot and 5,285 an inequality whose dual sign comes from the pivot column; none needs a dual bound from another column (a preview from a Mac build, not the reference; HISTORY Entry 42).
- **Test at the end:** First a size run against HiGHS's counters after the milestone of the singleton-column rules and the forcing column [M1], on the 44 instances that change with free column substitution off. The wider form cannot be set today; when it is built behind a switch, a size run with the hash of the reduced model at both values: identical hashes on all 240 close the entry, any difference sends it to the standard cluster comparison, read on the instances that differ. Until then the entry stays open as "needs a switch".
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]).

### AS-028 The zero-cost singleton rule takes only rows with two finite sides
- **Assumption:** The zero-cost singleton rule [R3] removes the continuous zero-cost singleton columns of a row that has two finite sides (an equation included) and at least one other free entry. A one-sided row is left to the lock rule or the forcing column. In a row visit it comes after the free singleton rules.
- **Set by:** Design (v2_design.txt:41, as in HiGHS).
- **Where:** stage1c_design/v2_design.txt:41, :119.
- **Switch:** `options_t::sub_enabled[sub_rule::zero_cost_singleton]` (tool `--off zero_cost_singleton`) turns the rule; no switch yet for the restriction.
- **Alternatives:** One-sided rows as well.
- **May change:** Slack-like columns disappear and their rows get wider sides (AS-266).
- **Test at the end:** Standard cluster comparison after the milestone of the singleton-column rules and the forcing column [M1]: `zero_cost_singleton` on against off.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]).

### AS-029 The zero-cost singleton rule leaves integer columns
- **Assumption:** An integer singleton column is not taken by the zero-cost singleton rule [R3], because the set of values of the rest of the row would have holes. The case of a row that is integral for the column is named as an extension and is not required.
- **Set by:** Design (v2_design.txt:41; the extension is named in v2_amendments.txt:118-119 "not required", and in lowworth_v2_design.txt:158 "named so that it is not found late").
- **Where:** stage1c_design/v2_design.txt:41; v2_amendments.txt:118-119; lowworth_v2_design.txt:158.
- **Switch:** no switch yet: needs one when the extension is built.
- **Alternatives:** Take an integer singleton column whose row is integral for it.
- **May change:** Integer slack columns stay, and so does a column that the implied-integer conversion made integer (AS-036).
- **Test at the end:** First a count on the 240 of zero-cost integer singleton columns whose row is integral for them. A count of zero closes the entry for the benchmark set; otherwise build the extension with a switch and run a standard cluster comparison, read on the instances whose reduced model differs.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]); the extension is not planned.

### AS-030 Dual bound strengthening never tightens a continuous column, it only fixes it
- **Assumption:** For an integer column dual bound strengthening [R7] tightens the bound. A continuous column is fixed only when every row gives its lower bound within tolerance and is otherwise left, so that explicit bounds of continuous columns stay as section 6 of the plan defines them.
- **Set by:** Design (v2_design.txt:69).
- **Where:** stage1c_design/v2_design.txt:69; DESIGN.md section 6 (lines 597-600).
- **Switch:** no switch yet: needs one.
- **Alternatives:** Tighten continuous columns too (PaPILO's dual strengthening of continuous columns, listed as not built, AS-032).
- **May change:** Looser bounds on continuous columns in the reduced model; weaker propagation in the solver.
- **Test at the end:** Needs the switch; then a standard cluster comparison: continuous tightening off against on.
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4]).

### AS-031 The single-lock alias tests its forcing on the lock row only
- **Assumption:** Whether the binary at its other value pushes the column to its far bound is read from the lock row alone. HiGHS's implication over all rows that the two columns share is not built unless the milestone of dual bound strengthening and the single-lock alias [M4] measures a gap.
- **Set by:** Design (v2_design.txt:77, :90, :249, :261).
- **Where:** stage1c_design/v2_design.txt:77, :90, :249, :261.
- **Switch:** no switch yet: needs one.
- **Alternatives:** HiGHS's implication over all shared rows.
- **May change:** Fewer aliases than HiGHS; candidates named are var-smallemery-m6j6, neos859080 and the radiation instances.
- **Test at the end:** A size run at the milestone of dual bound strengthening and the single-lock alias [M4] on those instances; if the wider form is built with a switch, a standard cluster comparison.
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4]); the extension only on evidence.

### AS-032 Two forms that the design attributes to PaPILO are not built with the nine rules
- **Assumption:** Not built: a singleton column with a cost when only one of its bounds is implied, and dual strengthening of continuous columns. The design attributes both forms to PaPILO; that was not read in PaPILO's source (AS-287).
- **Set by:** Design (v2_design.txt:90-94, :256).
- **Where:** stage1c_design/v2_design.txt:90-94, :256.
- **Switch:** no switch yet: needs one per form.
- **Alternatives:** Build them with the nine.
- **May change:** Reductions that PaPILO in cuOpt may have and the engine lacks.
- **Test at the end:** When built, each with a switch: standard cluster comparison, on against off.
- **State:** planned.

### AS-033 BHW coefficient reduction runs after the new presolve, never inside it, and overlaps coefficient strengthening
- **Assumption:** cuOpt's BHW coefficient reduction rewrites a one-sided row of 2 to 12 binaries with smaller integer coefficients and the same 0/1 solutions; a coefficient that becomes 0 drops the entry. In cuOpt as is it runs inside PaPILO's rounds; in mode 3 it runs once after the new presolve; in mode 2 not at all. The engine's own coefficient strengthening (AS-048) acts on the same rows before it. The port is planned after the gate of the cheap column rules and is to be measured against mode 3, because HiGHS has no such rule.
- **Set by:** Design (DESIGN.md amendments A6 and A8; "BHW comes after the gate", DESIGN.md:147); the owner asked for a way for it in the plan (HISTORY Entry 26: "cuopt has other special reductions in presolve. bhw ,bve, indicator, single_lock_dual_aggregation. try to find a way for them in the plan.").
- **Where:** cpp/src/presolve/bhw_coeff_reduce.hpp:30, 44-57, 62-69; cpp/src/presolve/bhw_coeff_reduce.cpp:335, :509; cpp/src/presolve/third_party_presolve.cpp:728, 1126-1129; DESIGN.md amendment A6 (table line 74) and amendment A8 (lines 111, 147).
- **Switch:** no switch yet: needs one. There is no hyper-parameter; the name `bhwcoeffreduce` is in the hard-coded list of mode 3.
- **Alternatives:** Port it into the row-tightening family so that it runs in every round; drop it where the engine's strengthening already acts; rows longer than 12.
- **May change:** Coefficients and sides of short binary rows: a tighter LP relaxation with almost no size effect, so the size figures give it no worth (AS-248).
- **Test at the end:** Standard cluster comparison in four arms: the engine's coefficient strengthening on or off (AS-048), each with `bhwcoeffreduce` in or out of the list of mode 3; read on the root relaxation objective and the final bound next to the primal measures. After the port: the ported rule inside the loop against the hosted one after it.
- **State:** in cuOpt as is; hosted by PaPILO in mode 3; the port is planned.

### AS-034 The equation test of the conversion counts as primal, the weak test as dual
- **Assumption:** The equation test of the implied-integer conversion and the column scaling are treated as keeping every feasible point and stay on in primal-only mode. The weak test keeps an optimal point only (the column is rounded in the direction of falling cost), is off in primal-only mode, and its records count as dual findings.
- **Set by:** Design (DESIGN.md section 8, lines 698-699, and lowworth_v2_design.txt:31-33, :91).
- **Where:** stage1c_design/lowworth_v2_design.txt:31-33, :91; DESIGN.md:501, :505, :698-699.
- **Switch:** `options_t::primal_only` together with the designed `implied_integer_weak`; neither weak-test switch is in code.
- **Alternatives:** Equation test off in primal-only mode as well.
- **May change:** In primal-only mode only equation-test conversions happen. The validity verifier showed that the equation test as first written could cut a feasible point in that mode too; the repair is binding (AS-093).
- **Test at the end:** Standard cluster comparison: `implied_integer_weak` on against off; and the pair default mode against primal-only mode (AS-044).
- **State:** designed (milestone of the implied-integer conversion [L2]).

### AS-035 The weak conversion test accepts a dual-implied equation only through the implied-equation rule
- **Assumption:** HiGHS's weak test also accepts a row that is an equation only by its dual bound. Here such a row must first have been made an equation by the implied-equation rule; HiGHS's clause "no row is an equation" is dropped; a fractional side over coefficient concludes nothing (no infeasibility is declared).
- **Set by:** Design (lowworth_v2_design.txt:31-32).
- **Where:** stage1c_design/lowworth_v2_design.txt:31-32.
- **Switch:** `implied_equations` and `implied_integer_weak` (designed engine switches; not in code).
- **Alternatives:** Accept a dual-implied equation directly; declare infeasibility on a fractional side.
- **May change:** With `implied_equations` off the conversion loses the columns whose proving row is an equation only by a dual argument. In HiGHS, with implied equations off the conversion costs 173 rows on physiciansched3-3 and 24 on savsched1 (run on the 240); the design infers another cause for that and expects the engine not to show it.
- **Test at the end:** The two-switch table of the milestone of the implied-integer conversion [L2] (physiciansched3-3 and savsched1); on the cluster it is part of the pair of AS-063.
- **State:** designed (milestone of the implied-integer conversion [L2]).

### AS-036 A converted column loses its roles in other rules
- **Assumption:** A column that the conversion made integer is no column of the zero-cost singleton rule, no compensator of the dominated-column rule, no absorber of the single-equation rule, and a source of an implied equation only with the integrality proof. The defence is the phase order: the conversion runs last in every round.
- **Set by:** Design (lowworth_v2_design.txt:32, :158, :227).
- **Where:** stage1c_design/lowworth_v2_design.txt:32, :158, :227; lowworth_v2_reviews.txt:128.
- **Switch:** no switch yet: needs one for the remedy; the designed statistic `blocked_r3` counts columns the zero-cost singleton rule lost to the conversion.
- **Alternatives:** Keep a converted column continuous for those rules (a third column type).
- **May change:** A conversion can make an instance larger. The worth verifier's reading supports that blp-ic98 does not end larger in the engine (its 89 zero-cost singleton columns are removed before the first conversion call).
- **Test at the end:** First a size run: the milestone of the implied-integer conversion [L2] traces every instance that is larger than after the implied equation [L1] on the 240 and the 40. The cluster pair is that of AS-065.
- **State:** designed (milestone of the implied-integer conversion [L2]).

### AS-037 Column stuffing takes integer candidates only when all candidates are integer with equal coefficient
- **Assumption:** In column stuffing either all candidates of a row are integer columns with equal absolute coefficient, or the integer ones are dropped and only the continuous ones remain. This is HiGHS's restriction.
- **Set by:** Design (lowworth_v2_design.txt:45).
- **Where:** stage1c_design/lowworth_v2_design.txt:45-46.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Mixed candidates or integer candidates with unequal coefficients (not valid by the one-row knapsack argument without further conditions).
- **May change:** Rows with integer singleton columns of unequal coefficients are left alone.
- **Test at the end:** Part of the pair of AS-067.
- **State:** designed (milestone of column stuffing [L3]).

### AS-038 The single-equation extension is on, with continuous absorbers only
- **Assumption:** In an equation, columns that only this equation locks and whose cost does not rise are fixed at a bound when continuous columns of the opposite kind can absorb the change. Absorbers are continuous columns only; an integer column, and a column the conversion made integer, is no absorber. The rule is off in primal-only mode. It is the fourth of the five further rules and not part of the nine: until it is built, a column whose only lock is an equation is left.
- **Set by:** Design (lowworth_v2_design.txt:51-56, following HiGHS; left out of the nine rules by v2_design.txt:70, :77, :90); the build follows the owner's "Build them too" (HISTORY Entry 38).
- **Where:** stage1c_design/lowworth_v2_design.txt:51-56; lowworth_v2_reviews.txt:86 and lowworth_v2_amendments.txt:31-33 (the measured figure, which replaces the bound of 77 rows, 84 columns and 188 nonzeros in line 56 of the design); v2_design.txt:70, :77, :90; v2_amendments.txt:8-13; HISTORY Entry 42 (lines 2336, 2343-2345).
- **Switch:** `single_equation` (designed; not in code).
- **Alternatives:** Off; integer absorbers under an integrality proof.
- **May change:** Almost nothing in size: in HiGHS, with the other dual-fixing parts on, the single-equation part changes two instances among 237 of the 240 (icir97_tension and neos-5093327-huahum: 1 row, 8 columns and 20 nonzeros in all), which is no change at the four decimals of the table of HISTORY Entry 42; no change on the 40.
- **Test at the end:** Standard cluster comparison: `single_equation` on against off; expected: no measurable change.
- **State:** designed (milestone of the single-equation extension [L4]).

### AS-039 The dominated-column rule is on, with continuous compensators only
- **Assumption:** A column is fixed at a bound when the cost it saves by moving there is at least what the cheapest compensating continuous columns of its locking rows cost. Compensators are continuous columns only (HiGHS also takes implied-integer columns); the fixed column may be of any type. The rule is off in primal-only mode.
- **Set by:** Design (lowworth_v2_design.txt:58-63: "Narrower than HiGHS: continuous compensators only"); the build follows the owner's "Build them too".
- **Where:** stage1c_design/lowworth_v2_design.txt:58-63.
- **Switch:** `dominated_columns` (designed; not in code).
- **Alternatives:** Off; integer or implied-integer compensators with an integrality proof.
- **May change:** Fixes columns at bounds, mostly ones the lock rule would also fix. Measured with HiGHS on 237 of the 240 beyond the forcing column: seymour1, b1c1s1 and savsched1 change, 1.00003 / 1.00003 / 1.00005 in nonzeros / rows / columns.
- **Test at the end:** Standard cluster comparison: `dominated_columns` on against off.
- **State:** designed (milestone of the dominated column [L5]).

### AS-040 A strictly dominated column with an infinite target gives no "unbounded or infeasible" status
- **Assumption:** When a strictly dominated column has an infinite target bound, HiGHS returns "unbounded or infeasible". The engine does not; it only counts the case (`dominated_unbounded`).
- **Set by:** Design (lowworth_v2_design.txt:59 "Not taken"; part of its decision 8 for the owner, not answered).
- **Where:** stage1c_design/lowworth_v2_design.txt:59, :245.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Return the status as HiGHS does.
- **May change:** An unbounded or infeasible model is not detected by this rule in presolve; the solver has to find it.
- **Test at the end:** Not a quality test on the benchmark set (its instances are feasible and bounded); a status test on unbounded models if the status is built.
- **State:** designed (milestone of the dominated column [L5]).

### AS-041 A dominated-column fixing against the cost needs the implied equation first
- **Assumption:** The dominated-column rule does not use a compensation with a negative price in a row that does not lock the candidate. A fixing against the cost therefore happens only after the implied-equation rule has made the covering rows equations; HiGHS keeps such fixings without that.
- **Set by:** Design (lowworth_v2_design.txt:59; lowworth_v2_answers.txt:110: the extension "is valid by the same exchange and is not built").
- **Where:** stage1c_design/lowworth_v2_design.txt:59, :189; lowworth_v2_answers.txt:110.
- **Switch:** no switch yet: needs one for the extension; the effect depends on the designed `implied_equations`.
- **Alternatives:** Build the compensation with a negative price in a non-locking row.
- **May change:** seymour1's 32 rows, 10 columns and 136 nonzeros are reached only by the implied equation followed by the dominated-column rule (predicted). The two switches are not independent.
- **Test at the end:** First a size run: the milestone of the dominated column [L5] measures seymour1 with stuffing off. On the cluster the effect is inside the pairs of AS-063 and AS-039.
- **State:** designed (milestone of the dominated column [L5]).

### AS-042 Conversion, single equation and dominated column do nothing on models without a continuous column
- **Assumption:** The two new families of the five rules are inactive while the state has no live continuous column, and they are seeded with the continuous columns only. On the 89 of the 240 that have integer columns only they never fire; the worth verifier states that this loses nothing against HiGHS, whose firings of the five rules on those instances are stuffing only.
- **Set by:** Design (lowworth_v2_design.txt:13, :88-89; seeding and inactivity by two binding amendments, lowworth_v2_amendments.txt:36-37 and :44).
- **Where:** stage1c_design/lowworth_v2_design.txt:13, :88-89, :103; lowworth_v2_amendments.txt:36-37, :44; lowworth_v2_reviews.txt:65, :89-92.
- **Switch:** no switch yet: needs one (a consequence of "compensators and absorbers are continuous").
- **Alternatives:** Integer compensators and absorbers with integrality proofs.
- **May change:** For pure integer models these rules cannot strengthen the formulation at all; only the implied equation with an integer source and stuffing can act there.
- **Test at the end:** No pair of its own; visible when the per-rule cluster results are split into instances with and without continuous columns.
- **State:** designed (milestones of the implied-integer conversion [L2] and of the single-equation extension [L4]).

## 2. Defaults of the rules

### AS-043 All rule families of the Stage 1 engine are on by default
- **Assumption:** Row rules, bound propagation, row tightening, doubleton equations, parallel rows and dual fixing are all switched on. No family has been tested for its effect on solution quality by switching it off; only their worth in nonzeros is known (AS-247).
- **Set by:** Design (the Stage 1 rule set of the plan).
- **Where:** gpu_presolve/include/gprs/presolve.hpp:111; gpu_presolve/src/presolve.cpp:179-182.
- **Switch:** `options_t::enabled[rule::row_rules, propagation, tightening, doubleton, parallel_rows, dual_fixing]` (six of the ten flags of the array, all on by default; the four column families are AS-050); tool `--off <family>`; not passed by the adapter (AS-152).
- **Alternatives:** Any family off.
- **May change:** Each family removes rows, columns or nonzeros or changes coefficients and bounds. Row tightening is worth 0.6% in nonzeros on the 40, but it is the family that strengthens coefficients, so that number does not say what it is worth for the formulation; the same holds for bound propagation.
- **Test at the end:** Standard cluster comparison, one pair per family: all on against that family off; row tightening and bound propagation first. The off arm of bound propagation is partly repaired by cuOpt's own bound update after the presolve (AS-142), so that pair needs the count of bounds cuOpt changes in each arm.
- **State:** in the engine.

### AS-044 The engine uses dual reductions by default; primal-only mode switches them off
- **Assumption:** `primal_only` is false by default. In primal-only mode every reduction that rests on a lock, cost or dominance argument is off, so that every feasible point of the original model crushes to a feasible reduced point with the same objective. Today that is dual fixing; the switch also covers six of the eight column reductions, none of which is built yet (all but the free singleton with an equation pivot and the merge of parallel columns); of the five further rules only the equation test of the conversion and the column scaling stay on.
- **Set by:** Design (DESIGN.md section 8: "The switch exists from Stage 1").
- **Where:** DESIGN.md section 8 (lines 698-700); gpu_presolve/include/gprs/presolve.hpp:106-109; gpu_presolve/src/presolve.cpp:179-182; gpu_presolve/src/rules_dual_fixing.cpp:29; gpu_presolve/include/gprs/presolve.hpp:168-177 (`sub_rule_active`); stage1c_design/lowworth_v2_design.txt:91.
- **Switch:** `options_t::primal_only` (default false); tool `--primal-only`. Inside cuOpt it follows the solution callback (AS-154).
- **Alternatives:** Primal-only as the default.
- **May change:** With dual reductions, feasible points that an optimum does not need are removed, and a point of the original model need not map to a point of equal objective. Without them the model is larger: with dual fixing off the engine left 1.077 times the nonzeros (geometric mean, the 40).
- **Test at the end:** Standard cluster comparison: `primal_only` false against true (needs a switch in cuOpt), repeated when the column rules are built.
- **State:** in the engine; the column reductions designed (milestones of the nine column rules and the five further rules [M1 to L5]).

### AS-045 Dual fixing is on, also for integer columns; zero-cost columns go to the lower bound
- **Assumption:** A column with cost at least 0 and no row that stops it from moving down is fixed at its lower bound, one with cost at most 0 and no row that stops it from moving up at its upper bound. With zero cost and no such row in either direction the value is the lower bound if finite, else the upper bound, else 0. The rule runs in every cascade pass.
- **Set by:** Design for the rule (a Stage 1 rule); code for the value at zero cost.
- **Where:** gpu_presolve/src/rules_dual_fixing.cpp:24-68; gpu_presolve/include/gprs/rules.hpp:356-362.
- **Switch:** `options_t::enabled[rule::dual_fixing]` (tool `--off dual_fixing`); `options_t::primal_only`.
- **Alternatives:** Off, which keeps every feasible point; for zero cost the bound nearest to 0.
- **May change:** Columns leave the model and with them feasible points that an optimum does not need: the optimal value stays, the set of solutions that heuristics can reach shrinks.
- **Test at the end:** Standard cluster comparison: dual fixing on against off.
- **State:** in the engine.

### AS-046 Row addition is on: equations are subtracted from rows parallel up to singleton columns
- **Assumption:** When two rows are proportional outside their singleton columns and one is an equation, the equation is subtracted from the other row (HiGHS's equality row addition). The rule runs with the Stage 1 rules and is on by default; the step itself deletes no row.
- **Set by:** The owner for the stage (HISTORY Entry 26, 2026-11-27: "moving the reduction to stage 1 is okay."); design for the default "on".
- **Where:** DESIGN.md amendment A2 (lines 36-39); gpu_presolve/include/gprs/presolve.hpp:159-162; gpu_presolve/include/gprs/rules.hpp:293-347; gpu_presolve/src/rules_parallel_rows.cpp:364-494, 808-827.
- **Switch:** `options_t::near_parallel_rows` (default true); tool `--exact-parallel-rows` turns it off; not passed by the adapter.
- **Alternatives:** Off: only rows whose free entries are proportional are merged.
- **May change:** It removes entries (dlr2 has 202,759 subtractions, HISTORY Entry 31; with it and the rounds on explicit bounds the engine went from 0.9955 to 1.0028 against the Stage 1 reference on the 40). The rows the solver sees are combinations of original rows: a row whose columns were all integer can get a continuous singleton column and is then no row for integer row tightening, and a bound the row gave is afterwards read from the equation.
- **Test at the end:** Standard cluster comparison: `near_parallel_rows` true against false.
- **State:** in the engine.

### AS-047 Implied bounds of continuous columns become explicit bounds of the reduced model
- **Assumption:** During the first rounds bound propagation records the bounds it derives for continuous columns and leaves the explicit bounds alone (integer columns are tightened at once). One call then writes the recorded bounds as explicit bounds and the rounds go on until nothing changes, so the reduced model that the solver gets carries the tightened bounds and is a fixed point of the rules.
- **Set by:** Design (DESIGN.md section 6, following HiGHS, which makes implied bounds explicit only when it prepares probing); the further rounds after the switch are a repair built by agents (HISTORY Entries 19 and 25).
- **Where:** DESIGN.md section 6 (lines 597-602); gpu_presolve/src/presolve.cpp:411-438; gpu_presolve/src/rules_propagation.cpp:225-278; gpu_presolve/include/gprs/presolve.hpp:275-304.
- **Switch:** no switch yet: needs one; the step runs whenever `options_t::enabled[rule::propagation]` is on.
- **Alternatives:** Return the original bounds of continuous columns and use the implied ones only inside the presolve; make explicit only a bound that replaces an infinite one; explicit from the first round.
- **May change:** Tighter boxes on continuous columns make more rows redundant, singleton or forcing (reference over engine nonzeros went from 0.9905 to 1.0004 on 15 local testbed instances, at 1.22 times the time), but take implied-free columns away from the dual and substitution rules. The LP optimum is the same; ranges for the solver's propagation and heuristics shrink and the LP becomes more degenerate. A strengthening that removes no nonzero by itself.
- **Test at the end:** Needs the switch. Standard cluster comparison: bounds made explicit (today) against original bounds of continuous columns returned. cuOpt's own bound update tightens bounds again after the presolve in both arms (AS-142), so the pair also needs the count of bounds that update changes in each arm.
- **State:** in the engine.

### AS-048 Coefficient strengthening of integer columns is on
- **Assumption:** In a row with exactly one finite side b and a finite maximum activity M, with M - b above the feasibility tolerance, every integer column whose coefficient exceeds M - b by more than the tolerance gets the coefficient M - b (plus the activity error) and the side moves with it. It applies to rows with continuous columns as well and is repeated at every visit.
- **Set by:** Design (DESIGN.md section 4, line 530-531).
- **Where:** DESIGN.md section 4 (lines 530-531); gpu_presolve/src/rules_tightening.cpp:254-312, :347; gpu_presolve/include/gprs/rules.hpp:224-228.
- **Switch:** no switch yet: needs one of its own; `options_t::enabled[rule::tightening]` turns the whole family.
- **Alternatives:** Off; binaries only; rows with two finite sides; leave it to cuOpt's BHW coefficient reduction, which mode 3 runs after the new presolve on one-sided rows of 2 to 12 binaries (AS-033).
- **May change:** A tighter LP relaxation with no nonzero removed: a strengthening that the size gates do not see. The new coefficient can be a non-integer such as 1.0000000000000002 (HISTORY Entry 18), which then stops the integral scaling of that row.
- **Test at the end:** Needs the sub-switch and a way to take `bhwcoeffreduce` out of the list of mode 3: in the standard comparison BHW runs after the engine in both arms and repairs part of the off arm. Standard cluster comparison in four arms: the engine's coefficient strengthening on or off, each with BHW in or out. The two arms without BHW give the worth of the engine's rule; the two arms with the engine's rule on give what BHW still adds. Read on the root relaxation objective and the final bound next to the primal measures, on the instances whose reduced model differs.
- **State:** in the engine.

### AS-049 Sides of all-integer rows are rounded, with the gcd of the coefficients
- **Assumption:** After the scaling to integral coefficients (AS-084) the sides of a row whose free columns are all integer are rounded to integers within the tolerance, divided by the gcd and rounded again. It is not done when the scaled side exceeds 2^52, or, with constant terms in the row, when side or terms times the factor exceed 1e8.
- **Set by:** Design (row tightening with Chvatal-Gomory rounding, DESIGN.md section 4); `max_exact_side = 0x1p52` is code.
- **Where:** gpu_presolve/src/rules_tightening.cpp:36-38, 148-166, 180-251; gpu_presolve/include/gprs/rules.hpp:219-223.
- **Switch:** no switch yet: needs one of its own; `options_t::enabled[rule::tightening]` turns the whole family.
- **Alternatives:** Off.
- **May change:** A tighter LP relaxation without a nonzero removed. An equation with a fractional right-hand side gives "infeasible".
- **Test at the end:** Needs the sub-switch. Standard cluster comparison: side rounding on against off.
- **State:** in the engine.

### AS-050 All nine column reductions are on by default
- **Assumption:** Simple probing and the eight reductions of the singleton-column, dual-column and parallel-column families (free singleton with an equation pivot, zero-cost singleton, forcing column, free singleton with an inequality pivot, dominance between parallel columns, dual bound strengthening, single-lock alias, merge of parallel columns) are on in default mode. Nothing is chosen per instance or by model size. None is built yet: each family takes its lists and reports nothing.
- **Set by:** Design (v2_design.txt:115); the scope "all of HiGHS's cheap rules" and the order are the owner's ("Yes, cheap rules first", HISTORY Entry 29; "Build them too", Entry 38).
- **Where:** stage1c_design/v2_design.txt:115; gpu_presolve/include/gprs/presolve.hpp:111, :124-125; gpu_presolve/include/gprs/rules.hpp:40-61; the four empty family sources under gpu_presolve/src/.
- **Switch:** `options_t::sub_enabled[sub_rule::free_singleton, zero_cost_singleton, forcing_column, free_singleton_dual, parallel_dominance, dual_bound, lock_alias, parallel_merge]` and `options_t::enabled[rule::simple_probing, singleton_cols, dual_cols, parallel_cols]`; tool `--off <name>`.
- **Alternatives:** Any single reduction off; a whole family off.
- **May change:** Rows, columns and nonzeros of the reduced model; aliases, merges and fixings change which columns the solver branches on and what the heuristics see; several rest on a cost argument and remove feasible points; the free singleton rules move a row into the objective (AS-095).
- **Test at the end:** Standard cluster comparison, nine pairs: all on against one reduction off; and one pair with the four column families off against on (what the stage is worth in quality).
- **State:** designed (milestones of the nine column rules [M1 to M5]); the switches are in the engine and no rule reads them yet.

### AS-051 The single-lock alias is required, as one rule for HiGHS's and cuOpt's forms
- **Assumption:** A column with a single lock against its cost is aliased to a binary of its lock row [R8]. The rule is built once and stands for both HiGHS's dual-fixing binary substitution and cuOpt's single-lock dual aggregation, and it is on by default, although in a full HiGHS presolve two of seven tested instances end larger with HiGHS's substitution on (var-smallemery-m6j6 276,002 against 273,018 nonzeros, piperout-d27 79,040 against 78,585).
- **Set by:** Taken as recommended (HISTORY Entry 40: "the single-lock alias stays in the required set"; v2_amendments.txt:89; decision 5 of v2_design.txt:280).
- **Where:** DESIGN.md amendment A8 (lines 123-124) and amendment A6 (table line 75); stage1c_design/v2_design.txt:80, :280; gpu_presolve/baseline/HIGHS_DIAG.md:923-924 (the design cites :828-829; the file has grown); gpu_presolve/include/gprs/rules.hpp:56-57.
- **Switch:** `options_t::sub_enabled[sub_rule::lock_alias]` (tool `--off lock_alias`).
- **Alternatives:** Off; an optional milestone, which leaves var-smallemery-m6j6 (0.86) and neos859080 (0.94) below 0.95 as accepted exceptions; leave it to cuOpt's own rule in mode 3 (AS-137).
- **May change:** A column is replaced by a binary times its range (up to 1e6): fewer columns, larger coefficients of big-M kind. Separable worth on HiGHS: 0.0012 of the mean log in nonzeros on 18 of the 240. It may hinder the later aggregator.
- **Test at the end:** Standard cluster comparison after the milestone of dual bound strengthening and the single-lock alias [M4], in four arms in mode 3: `lock_alias` on or off, each with cuOpt's `single_lock_dual_aggregation` in or out of the list of mode 3 (needs a way to edit that list, AS-137). The two arms without cuOpt's rule give the worth of the alias as such; the two arms with the engine's rule on give what cuOpt's rule still finds; the two arms with cuOpt's rule in give what the engine's wider form adds. In a plain on-against-off pair cuOpt's rule finds the binary cases again in the off arm. Again when the aggregator exists.
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4], second step).

### AS-052 Which columns the single-lock alias takes
- **Assumption:** A column is aliased only if it has finite bounds with a range of at most 1e6 and the lock row's side is its only lock against its cost (cost at least 0 with one down lock, or cost at most 0 with one up lock). The lock row is no equation and stays in the model. General integer and continuous columns are taken, not only binaries.
- **Set by:** Design (v2_design.txt:76-78).
- **Where:** stage1c_design/v2_design.txt:76-78, :90 (an equation is no lock row).
- **Switch:** `options_t::sub_enabled[sub_rule::lock_alias]` turns the rule; no switch yet for the choice of candidates.
- **Alternatives:** Binary columns only (cuOpt's rule); columns locked down only (HiGHS's substitution).
- **May change:** General integer and continuous columns become multiples of a binary: fewer columns, denser rows of the binary.
- **Test at the end:** Covered by the pair `lock_alias` on against off (AS-051).
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4]).

### AS-053 The single-lock alias tests one candidate binary per row
- **Assumption:** The single-lock alias tests only the binary with the largest coefficient in the lock row (the second largest when the column is that binary), ties to the smaller column index, because both of its tests are lower limits on that coefficient. HiGHS and cuOpt's rule test every binary of the row.
- **Set by:** Design (repair of a cost defect, v2_answers.txt item 21).
- **Where:** stage1c_design/v2_design.txt:76-77, :263; v2_answers.txt:83-86.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Every binary of the row.
- **May change:** Which binary a column is aliased to when several would pass with equal coefficients.
- **Test at the end:** Covered by the pair of AS-051.
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4]).

### AS-054 The single-lock alias releases only the locking side of its row
- **Assumption:** The release test is made on the side that locks the column, not on every finite side of the row. HiGHS and cuOpt's rule test every finite side, which is stricter than the proof needs.
- **Set by:** Design (v2_design.txt:77), confirmed by the validity critic's enumeration of 7,594 applications on tiny integer models without a failure.
- **Where:** stage1c_design/v2_design.txt:77; v2_reviews.txt:9.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Test every finite side.
- **May change:** More aliases than HiGHS or cuOpt's rule on ranged lock rows.
- **Test at the end:** A count of the aliases that only this form finds on the 240; then covered by the pair of AS-051.
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4]).

### AS-055 Dual bound strengthening asks only for the lock in its own direction
- **Assumption:** Dual bound strengthening [R7] needs cost at least 0 and at least one down lock for a new upper bound (mirror for a lower bound) and nothing about the other direction; with zero cost it tries down first and up only when down changed nothing.
- **Set by:** Design (repair after the validity critique, v2_answers.txt item 3).
- **Where:** stage1c_design/v2_design.txt:69-70; v2_answers.txt:11-14.
- **Switch:** `options_t::sub_enabled[sub_rule::dual_bound]` (tool `--off dual_bound`).
- **Alternatives:** Locks in both directions required (the first design, which lost reductions).
- **May change:** Upper bounds of integer columns and fixings; in HiGHS on the 240, supportcase12 loses 1.755 in nonzeros without it (184,311 fixings).
- **Test at the end:** Standard cluster comparison after the milestone of dual bound strengthening and the single-lock alias [M4]: `dual_bound` on against off. The off arm also stops the `range_kept` merges of AS-001.
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4], first step).

### AS-056 The forcing column goes by lock counts; its undo takes the minimum over rows
- **Assumption:** A zero-cost column of any type with no lock in a direction where its bound is infinite is removed with all its rows [R4], by lock counts without a tolerance. The undo takes the minimum over the saved rows and the bound (mirror for the other direction), rounded inward for an integer column. Until the rule is built such a column and its rows stay in the model: dual fixing leaves them.
- **Set by:** Design (v2_design.txt:48-50; the undo rule is fixed by a binding amendment after the review found two readings, v2_amendments.txt:67-69).
- **Where:** stage1c_design/v2_design.txt:48-50; v2_amendments.txt:67-69; gpu_presolve/src/rules_dual_fixing.cpp:47-55; gpu_presolve/include/gprs/rules.hpp:48-49.
- **Switch:** `options_t::sub_enabled[sub_rule::forcing_column]` (tool `--off forcing_column`); no rule reads it yet.
- **Alternatives:** The point nearest to a preferred value in the undo.
- **May change:** Rows and columns that an optimal solution never needs leave the model (in HiGHS on the 240: neos-2746589-doon 1.124 in rows, neos-3656078-kumeu, neos-3402454-bohle); the value of the column in the postsolved solution.
- **Test at the end:** Standard cluster comparison after the milestone of the singleton-column rules and the forcing column [M1]: `forcing_column` on against off.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]).

### AS-057 Zero-cost singleton columns of a row share one record, with a fixed undo order
- **Assumption:** A row visit removes all zero-cost singleton columns of the row with one record. In the undo every column starts at its preferred value (lower bound if finite, else upper bound, else 0) and the columns are moved in index order until the row holds.
- **Set by:** Design (the designer's own change in version 2, v2_design.txt:14).
- **Where:** stage1c_design/v2_design.txt:14, :41-43, :129.
- **Switch:** no switch yet: needs one.
- **Alternatives:** One record per column; another split of the row's slack among the columns.
- **May change:** The values of the removed columns in the postsolved solution (their cost is zero, so the objective is unchanged).
- **Test at the end:** No quality effect expected; tested by the structure check's own undo and the round-trip clause of the crush check.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]).

### AS-058 Two continuous parallel columns are always merged
- **Assumption:** Two continuous parallel columns with equal cost are merged with every setting of `merge_integers`, the smaller index kept.
- **Set by:** Taken as recommended (v2_amendments.txt:86: "Continuous pairs are always merged"; DESIGN.md amendment A8, lines 147-149).
- **Where:** stage1c_design/v2_amendments.txt:86; v2_design.txt:83.
- **Switch:** `options_t::sub_enabled[sub_rule::parallel_merge]` turns every merge; no switch for continuous pairs alone.
- **Alternatives:** No merge.
- **May change:** Fewer columns; the merged column has the summed range. Interchangeable columns that are merged are no longer there for cuOpt's symmetry handling (AS-143).
- **Test at the end:** Standard cluster comparison after the milestone of the merge of parallel columns [M5]: `parallel_merge` on against off, with `merge_integers` = `never` in both arms so that only continuous and mixed pairs differ.
- **State:** designed (milestone of the merge of parallel columns [M5]).

### AS-059 An integer column is merged into a continuous one, and no column changes type
- **Assumption:** For a mixed parallel pair the continuous column is kept and the merge needs its range to be at least the absolute ratio. HiGHS keeps the integer column and turns it continuous, which the state cannot do. This merge runs with every setting, also in primal-only mode.
- **Set by:** Design (v2_design.txt:83-84, :109: the state has no type change).
- **Where:** stage1c_design/v2_design.txt:83-84, :109, :277.
- **Switch:** `options_t::sub_enabled[sub_rule::parallel_merge]`; no switch for mixed pairs alone.
- **Alternatives:** Leave mixed pairs; HiGHS's form with a type change.
- **May change:** An integer column disappears into a continuous one: one branching candidate less.
- **Test at the end:** First a count of mixed merges on the 240 (the design infers pair types from column counts); then covered by the pair of the entry above.
- **State:** designed (milestone of the merge of parallel columns [M5]).

### AS-060 Which of two parallel integer columns is kept, and the no-hole condition
- **Assumption:** The kept column is the one for which the ratio is integral within 1e-9 (the smaller index when the absolute ratio is 1), and the absolute ratio must be at most the kept range plus 1 so that the merged values have no hole. The condition is HiGHS's.
- **Set by:** Design (v2_design.txt:83-84).
- **Where:** stage1c_design/v2_design.txt:83-84.
- **Switch:** `options_t::merge_integers` (AS-001); no switch for the choice of the kept column.
- **Alternatives:** None considered.
- **May change:** Which index survives as the merged column.
- **Test at the end:** Covered by the three arms of AS-001.
- **State:** designed (milestone of the merge of parallel columns [M5]).

### AS-061 The undo of a merge is a closed form preferring the removed column's lower bound
- **Assumption:** In the undo the removed column gets the admissible value nearest to its lower bound (else upper bound, else 0) and the kept column the rest; when no admissible value exists (a reduced point that uses the tolerance) the kept column takes the excess. HiGHS's search of up to 1000 unit steps is not copied.
- **Set by:** Design (fallback added by v2_answers.txt item 8).
- **Where:** stage1c_design/v2_design.txt:85; v2_answers.txt:31-34.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Split evenly; HiGHS's stepwise search.
- **May change:** How a merged value is split between the two original columns in the postsolved solution; the kept column can pass its bound by the tolerance.
- **Test at the end:** No quality effect expected (equal costs, same rows); tested by unit tests and the round-trip clause of the crush check.
- **State:** designed (milestone of the merge of parallel columns [M5]).

### AS-062 Singleton columns take part in the parallel-column rules
- **Assumption:** All singleton columns of one row form one bucket and are tested for dominance and merge as a group, as HiGHS does. The design says PaPILO skips them; that statement was not read in the PaPILO source.
- **Set by:** Design (v2_design.txt:62-63, :83).
- **Where:** stage1c_design/v2_design.txt:62-63, :83.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Skip singleton columns.
- **May change:** On neos-3004026-krka, satellites2-60-fs and satellites2-40 every column that HiGHS fixes by dominance between parallel columns is a singleton.
- **Test at the end:** Covered by the pair `parallel_dominance` on against off (AS-050); these instances are named in the milestone of dominance between parallel columns [M3].
- **State:** designed (milestone of dominance between parallel columns [M3]).

### AS-063 The implied equation is on by default and turns inequalities into equations
- **Assumption:** When a column with a cost has exactly one row that stops it from moving in the direction in which its cost falls, and its bound in that direction is infinite or implied by that row, the row is made an equation. The rule removes nothing itself; it is on by default and off in primal-only mode.
- **Set by:** Design for the default (lowworth_v2_design.txt:23-28); the build follows the owner's "Build them too" (HISTORY Entry 38).
- **Where:** stage1c_design/lowworth_v2_design.txt:23-28, :70.
- **Switch:** `implied_equations` (designed as an engine switch; not in code).
- **Alternatives:** Off; HiGHS's form with a strict margin for columns with more than one entry; continuous sources only.
- **May change:** More equations in the reduced model: both sides of the row then lock its columns, which can block lock-based fixings, and the new equations feed doubleton equations, simple probing and the free singleton rule in the next round. For the solver an equation is harder to satisfy for local-search heuristics than the inequality it came from. HiGHS counts 666 such conversions on 11 of the 240.
- **Test at the end:** Standard cluster comparison: `implied_equations` on against off. With it off the dominated-column rule also loses its fixings against the cost.
- **State:** designed (milestone of the implied equation [L1]).

### AS-064 An integer column may be the source of an implied equation
- **Assumption:** An integer column is accepted as the source of an implied equation when its lock row proves integrality for it. This goes beyond HiGHS, which takes implied-integer columns without a new test and no true integer, and beyond the plan's guard that dual arguments use continuous columns only.
- **Set by:** Design (recommendation "on", decision 4 of lowworth_v2_design.txt:241; no owner answer in HISTORY up to Entry 46).
- **Where:** stage1c_design/lowworth_v2_design.txt:24, :25, :241; lowworth_v2_answers.txt:126, :142; DESIGN.md:516.
- **Switch:** `integer_sources` (designed, default on; not in code).
- **Alternatives:** Off: continuous sources only; a column that the conversion made integer then stops being a source.
- **May change:** More implied equations on models with integer cost columns. The worth verifier counts 285,999 integer columns on 59 of the 240 that pass the cheap tests on the input model, for example 1,908 on bab6 and 206 on netdiversion that the single-lock alias would otherwise take. What "off" loses is unknown until the milestone of the implied-integer conversion [L2] reports both settings.
- **Test at the end:** Standard cluster comparison: `integer_sources` on against off, `implied_equations` on in both.
- **State:** designed (milestone of the implied equation [L1]).

### AS-065 The implied-integer conversion is on by default and runs last in every round
- **Assumption:** A continuous column that is the only continuous column of an equation with integral ratios and side (equation test), or of all its rows with integral bounds (weak test), is made an integer column. The design recommends "on, last in the round, with the three switches kept"; the milestone of the implied-integer conversion [L2] stops on any instance that grows, and then the owner decides the default.
- **Set by:** Design (decision 7 of lowworth_v2_design.txt:244; no owner answer in HISTORY up to Entry 46). The build follows "Build them too".
- **Where:** stage1c_design/lowworth_v2_design.txt:30-35, :91, :206, :227, :244.
- **Switch:** `implied_integer_equation`, `implied_integer_weak`, `implied_integer_scaling` (designed; not in code).
- **Alternatives:** Off by default; the form without a type change (the weak test only as a gate for dual bound strengthening).
- **May change:** The conversion removes nothing itself. It adds integrality: bounds of the column are rounded from then on, its rows can enter row tightening, dual bound strengthening may tighten it, and the solver gets more integer columns to branch on and to round (AS-165). HiGHS converts 23,621 columns on 31 of the 240; in HiGHS's own run blp-ic98 and blp-ar98 end 3% larger with the conversion on.
- **Test at the end:** Standard cluster comparison: the three conversion switches on against off; then one pair per switch.
- **State:** designed (milestone of the implied-integer conversion [L2]).

### AS-066 Column scaling is built and on, but only where the equation test fails without it
- **Assumption:** When an equation would prove integrality of its only continuous column after the column is multiplied by a factor (HiGHS's transform x = s z), the column is rescaled and converted in one step. Unlike HiGHS, which scales whenever its scale is not 1, the engine tries the unscaled equation test first, so that a column keeps its units where integrality can be had without a scaling.
- **Set by:** Design (decision 6 of lowworth_v2_design.txt:243; no owner answer in HISTORY up to Entry 46).
- **Where:** stage1c_design/lowworth_v2_design.txt:37-42, :243.
- **Switch:** `implied_integer_scaling` (designed; not in code).
- **Alternatives:** Do not build it; scale whenever the scale is not 1.
- **May change:** No nonzero is removed; coefficients, cost and bounds of the column are multiplied or divided by the scale and the column becomes integer in other units (1,400 columns on neos-4647030-tutaki by HiGHS's counters). Predicted size worth on the 240: at most physiciansched3-3's 28 rows, 14 columns and 106 nonzeros, shared with the conversion; none on the 40.
- **Test at the end:** Standard cluster comparison: `implied_integer_scaling` on against off, the two other conversion switches on in both.
- **State:** designed (milestone of the implied-integer conversion [L2], second step).

### AS-067 Column stuffing is on by default, with no cap on the row length
- **Assumption:** In a row with exactly one finite side, singleton columns whose cost pushes against the row are filled in the order of cost per unit; a candidate that still fits is fixed at the bound its cost prefers, one that can no longer matter at its other bound. The rule is off in primal-only mode. HiGHS's cap of 1000 on the row length is not used.
- **Set by:** Design (lowworth_v2_design.txt:44-49, :70); the build follows "Build them too".
- **Where:** stage1c_design/lowworth_v2_design.txt:44-49, :70.
- **Switch:** `column_stuffing` and the option `stuffing_row_limit` = 0 (designed; not in code).
- **Alternatives:** Off; a row length cap of 1000.
- **May change:** A few singleton columns are fixed. Measured with HiGHS on 237 of the 240: one column and one nonzero on sp97ar; 73 fixings on three instances; none on the 40.
- **Test at the end:** Standard cluster comparison: `column_stuffing` on against off; expected: no measurable change.
- **State:** designed (milestone of column stuffing [L3]).

### AS-068 One dominated-column fixing may use several implied premises, one per locking row
- **Assumption:** The plan's guard allows at most one implied premise per derivation. The dominated-column rule uses one per locking row that has a locked compensator, with the argument that the premises cannot lean on each other because each is used only in its own row.
- **Set by:** Design (decision 3 of lowworth_v2_design.txt:240, recommended "the per-row form"; no owner answer in HISTORY up to Entry 46).
- **Where:** stage1c_design/lowworth_v2_design.txt:60, :240; DESIGN.md:516-517.
- **Switch:** no switch yet: needs one; the design says "one named constant restores the limit of one" without naming it.
- **Alternatives:** At most one implied premise per fixing, at the price of the fixings that use two or more.
- **May change:** More dominated-column fixings; the worth verifier found that b1c1s1 does not depend on it.
- **Test at the end:** First the count of fixings with two or more premises on the 240 and the 40 (milestone of the dominated column [L5]). If it is not zero: standard cluster comparison, per-row form against the limit of one.
- **State:** designed (milestone of the dominated column [L5]).

## 3. Tolerances, limits and numerical guards

### AS-069 The feasibility tolerance of the engine is 1e-6, absolute, and its own
- **Assumption:** Every feasibility test of the presolve (infeasible and redundant rows, crossing bounds and sides, forcing rows, coefficient strengthening, the merge of parallel rows) uses one absolute tolerance, 1e-6. cuOpt checks rows with an absolute plus a relative tolerance, and PaPILO in the same place works with 1e-5; the engine's value does not follow either (AS-153).
- **Set by:** Design (DESIGN.md section 6: "Tolerances are inputs (feasibility 1e-6, dual 1e-7, zero 1e-9, integrality as cuOpt passes it)").
- **Where:** gpu_presolve/include/gprs/numerics.hpp:23; read in gpu_presolve/src/rules_rows.cpp:66-69, 212-219, rules_bounds.cpp:22, :40, presolve.cpp:91, :104, rules_tightening.cpp:272-275, rules_parallel_rows.cpp:342, :350. No tolerance table per rule, which the plan names, was found in the engine.
- **Switch:** `options_t::tol.feasibility` (default 1e-6); no tool flag, not passed by the adapter.
- **Alternatives:** The absolute tolerance cuOpt solves with; a tighter value (1e-7, 1e-9); a tolerance relative to the size of the row.
- **May change:** Which rows count as redundant or infeasible, which columns are fixed, how far coefficients are strengthened. A solution of the reduced model can violate a deleted row of the original model by up to the tolerance.
- **Test at the end:** Standard cluster comparison: 1e-6 against 1e-9 (a third arm at 1e-5), with the count of postsolved solutions that fail cuOpt's feasibility check on the original model.
- **State:** in the engine.

### AS-070 Bounds of integer columns are rounded with the feasibility tolerance
- **Assumption:** A lower bound of an integer column becomes ceil(lower - 1e-6) and an upper bound floor(upper + 1e-6), in the start step and at every tightening. A bound that lies up to 1e-6 past an integer is taken as that integer.
- **Set by:** Code (the plan names no separate value for this rounding).
- **Where:** gpu_presolve/include/gprs/numerics.hpp:129-137; used in gpu_presolve/src/rules_bounds.cpp:19, :37, presolve.cpp:99-102, rules_rows.cpp:118.
- **Switch:** no switch yet: needs one of its own; it shares `options_t::tol.feasibility` with every other feasibility test.
- **Alternatives:** Round with the integrality tolerance (1e-5); a tolerance that grows with the size of the bound; exact rounding.
- **May change:** An integer bound can end one unit tighter or looser; one unit of a general integer bound changes the LP relaxation and the branching.
- **Test at the end:** Needs the switch. Standard cluster comparison: rounding tolerance 1e-6 against 1e-9, with the crush check of known solutions as the correctness test.
- **State:** in the engine.

### AS-071 The integrality tolerance 1e-5 is read by no reduction
- **Assumption:** `tolerances_t::integrality` = 1e-5 is read only by the engine's own check of a point. The rules decide whether a coefficient, a scale or an offset is an integer with the zero tolerance 1e-9 and round bounds with the feasibility tolerance. The plan says "integrality as cuOpt passes it", but the adapter passes nothing.
- **Set by:** Code.
- **Where:** gpu_presolve/include/gprs/numerics.hpp:26; only reader gpu_presolve/src/postsolve.cpp:204-208.
- **Switch:** `options_t::tol.integrality` (default 1e-5); it changes no reduction today.
- **Alternatives:** Pass cuOpt's integrality tolerance; use it for the integrality tests of the rules in place of the zero tolerance.
- **May change:** Today nothing in the reduced model.
- **Test at the end:** No quality pair: the option changes no reduction. It becomes one if a rule starts to read it.
- **State:** in the engine.

### AS-072 The zero tolerance 1e-9 decides proportionality of rows and integrality of coefficients
- **Assumption:** `tolerances_t::zero` = 1e-9 is the relative tolerance of the parallel-row test and the absolute tolerance within which a scaled coefficient, an alias scale or an alias offset counts as an integer. One value serves the three uses.
- **Set by:** Design (DESIGN.md section 6: "zero 1e-9").
- **Where:** gpu_presolve/include/gprs/numerics.hpp:25; read in gpu_presolve/src/rules_parallel_rows.cpp:301, rules_tightening.cpp:119, :134, alias_forest.cpp:54-56, rules_bounds.cpp:69.
- **Switch:** `options_t::tol.zero` (default 1e-9).
- **Alternatives:** 1e-7 or 1e-12; separate tolerances for proportionality and for integrality.
- **May change:** How many parallel rows are found, how many all-integer rows are scaled to integral coefficients, how many doubleton equations between integers are used.
- **Test at the end:** First a size run with 1e-9 against 1e-7; if the reduced models differ, the standard cluster comparison with the same two arms.
- **State:** in the engine.

### AS-073 The dual tolerance has no reader; dual fixing reads the sign of the cost exactly
- **Assumption:** `tolerances_t::dual` = 1e-7 is read by no rule of the engine. Dual fixing treats a cost as zero only when it is exactly 0.
- **Set by:** Design for the value (DESIGN.md section 6: "dual 1e-7"); code for the exact test.
- **Where:** gpu_presolve/include/gprs/numerics.hpp:24; gpu_presolve/src/rules_dual_fixing.cpp:38-41, 49-55.
- **Switch:** no switch yet: needs one (`options_t::tol.dual` exists and has no reader; the designed implied equation will read it, AS-103).
- **Alternatives:** Treat an absolute cost of at most the dual tolerance as zero.
- **May change:** A column with a cost of 1e-12 and an infinite bound in the direction of the cost gives the status "unbounded or infeasible" instead of being left alone; a tiny cost decides the bound at which a column is fixed.
- **Test at the end:** First a count on the 240 of the columns with a cost above 0 and at most 1e-7 in absolute value; a cluster pair only if there are any.
- **State:** in the engine.

### AS-074 A term above 1e20 is infinite in an activity; a bound only as IEEE infinity
- **Assumption:** A product of coefficient and bound above 1e20 in absolute value is counted as an infinite term of a row activity. A bound or a side itself is infinite only when it is an IEEE infinity, so a bound of 1e20 or 1e30 that a model uses for "no bound" is a real bound for the engine. The adapter copies bounds as they are and refuses a model with NaN, a lower bound of plus infinity, an upper bound of minus infinity, non-finite coefficients or unset bound arrays.
- **Set by:** Design (DESIGN.md section 2.4: "A term with |a * bound| above 1e20 is counted as infinite"); the IEEE convention is part of the model contract.
- **Where:** gpu_presolve/include/gprs/numerics.hpp:28, 105-108; gpu_presolve/include/gprs/state.hpp:18-22; gpu_presolve/src/state_build.cpp:15-45; cpp/src/presolve/gprs_presolve.cpp:71-82, 106-109, 121-124.
- **Switch:** `options_t::tol.infinite_term` (default 1e20); no switch yet for the convention on bounds.
- **Alternatives:** A lower threshold for the term (1e10 or 1e15); a threshold on the bound itself above which the adapter takes it as infinite.
- **May change:** Whether rows with huge terms give bounds, and whether a column is fixed at a huge finite bound (AS-260). The 31 have no finite bound of 1e8 or more (DESIGN.md:849-850), so only the random campaign exercises this; no such count exists for the 240.
- **Test at the end:** First a count on the 240 of terms between 1e10 and 1e20 and of bounds at or above 1e20; the campaign with huge bounds through the adapter. A cluster pair `infinite_term` 1e20 against 1e10 only if reduced models differ.
- **State:** in the engine and the adapter.

### AS-075 An activity is summed again, and a candidate dropped, at a tenth of the tolerance
- **Assumption:** A stored row activity is summed from scratch when its error bound exceeds `activity_error_share` = 0.1 of the feasibility tolerance (and the side has had at least twice as many updates as the row has entries). For a column with coefficient a, no bound is derived from a row whose error bound, after one fresh summation, exceeds 0.1 * 1e-6 * |a|. The row rules and row tightening also sum again before a test when the error bound is above 1e-15. The plan says the guard itself is not measured.
- **Set by:** Design for the share (DESIGN.md sections 2.4 and 6, from the review of cuOpt pull request 2372, HISTORY Entry 5); code for the factor 2 and the threshold 1e-15.
- **Where:** DESIGN.md section 2.4 (lines 412-415, 452-454) and section 6 (lines 595-596); gpu_presolve/include/gprs/numerics.hpp:29-31, 121-126; gpu_presolve/src/state_ops.cpp:72-85; rules_bounds.cpp:60-74; gpu_presolve/src/implied.hpp:107-111; rules_propagation.cpp:162-185.
- **Switch:** `options_t::tol.activity_error_share` (default 0.1); the factor 2 and the product of zero and feasibility tolerance have no switch.
- **Alternatives:** Another share; sum again after every change; no guard; a guard on a proven bound (AS-261).
- **May change:** The last bits of stored activities and the tests they decide; columns with small coefficients in rows with large terms get no bound from those rows; presolve time.
- **Test at the end:** First a size run with the share at 0.01, 0.1 and 1 (reduced sizes, hashes, the correctness checks) and a count of dropped candidates, for which no counter exists. A cluster pair only if the reduced models differ.
- **State:** in the engine.

### AS-076 Sides, constants and fixing values above 1e8 are left alone
- **Assumption:** The limit feasibility / side_error = 1e-6 / 1e-14 = 1e8 is used in every family: no bound is propagated from a side above it, row tightening does not round a side or strengthen a coefficient when the value it would write exceeds it, two parallel rows are not merged when the constant terms of the kept row exceed it, a near-parallel subtraction is refused. The designed column rules and the five further rules take the same guard for sides, merged bounds and fixing values. Each family carries its own copy of the guard; the lesson that this "deserves one rule in the state" was not carried out.
- **Set by:** Design for propagation (DESIGN.md section 6: "no bound is derived when |b| * 1e-14 > feastol"); extended to the other families in repairs (HISTORY Entry 18) and in the rule designs (v2_answers.txt items 2 and 10).
- **Where:** gpu_presolve/include/gprs/numerics.hpp:35-36; gpu_presolve/src/rules_propagation.cpp:158-160; rules_tightening.cpp:74-80, 184-189, :297; rules_parallel_rows.cpp:307-320, 438-446; state_storage.cpp:152, :183; stage1c_design/v2_design.txt:27, :34, :41, :62, :69, :83; lowworth_v2_design.txt:24, :45, :52, :59.
- **Switch:** `options_t::tol.side_error` (default 1e-14) together with `options_t::tol.feasibility`.
- **Alternatives:** A higher limit (1e10 with `side_error` 1e-16); a limit relative to the row; sides kept in double-double.
- **May change:** On models with constants above 1e8 (large big-M sides) rows give no bounds and are not tightened or merged, and columns are not fixed at such values; those rows and columns stay where HiGHS removes them.
- **Test at the end:** First a count on the 240 of rows with a side above 1e8 and of reductions the limit blocks, plus the campaign with huge bounds. Then a standard cluster comparison: `side_error` 1e-14 against 1e-16, only if the count is not zero.
- **State:** in the engine for the Stage 1 rules; designed for the column rules.

### AS-077 An implied bound of a continuous column counts only when it improves by over 1e-3
- **Assumption:** For a continuous column, a bound that a row implies is recorded, and later written as explicit bound, only if it improves on what is there by more than `implied_bound_step` = 1000 times the larger of the feasibility tolerance and 1e-14 times the bound, which is at least 1e-3. Integer columns have no such step.
- **Set by:** Design (DESIGN.md section 6: "when they improve the record by more than 1000 * feastol").
- **Where:** gpu_presolve/include/gprs/numerics.hpp:32-34; gpu_presolve/src/rules_propagation.cpp:50-58, 100-120, :248, :262.
- **Switch:** `options_t::tol.implied_bound_step` (default 1000).
- **Alternatives:** 1 or 10 (any improvement beyond the tolerance); a step relative to the range of the column.
- **May change:** Bounds of continuous columns in the reduced model are up to 1e-3 looser than the rows imply; columns whose range is below 1e-3 are never tightened; fewer rounds.
- **Test at the end:** Standard cluster comparison: `implied_bound_step` 1000 against 10.
- **State:** in the engine.

### AS-078 Derived bounds are moved outward by their rounding and by the activity error
- **Assumption:** A bound derived for a continuous column is moved outward by 4 rounding units of its size plus the activity error over the coefficient, unless the residual and the division are exact. An integer bound, from propagation or from a singleton row, is moved outward by 4 rounding units before it is rounded to an integer.
- **Set by:** Code (a choice made with the implementation; the reason is in the source comment: a bound that is a rounding step too tight cuts a point off). No HISTORY entry that decided it was found.
- **Where:** gpu_presolve/src/implied.hpp:23-30, 163-181; gpu_presolve/src/rules_propagation.cpp:98-99, 142-146; gpu_presolve/src/rules_rows.cpp:30-33, 101-106.
- **Switch:** no switch yet: needs one (named constants `candidate_rounding_units` = 4 and `bound_rounding_units` = 4).
- **Alternatives:** No outward move, relying on the tolerances; a larger move.
- **May change:** Bounds of continuous columns are a few rounding units looser than the row gives. A column that dual fixing later fixes at such a bound gives an optimum that is off by about 1e-9 (AS-259).
- **Test at the end:** Not a quality pair (1e-9 is far below cuOpt's tolerances): the campaign and the triage of the two failing seeds.
- **State:** in the engine.

### AS-079 One explicit bound of a continuous column is tightened at most 16 times
- **Assumption:** After the implied bounds have become explicit, bound propagation tightens the same bound of a continuous column at most `max_explicit_steps` = 16 times. A further candidate is dropped and the presolve reports that it is not at a fixed point. The source comment says "the local testbed needs 3" without naming the instance set.
- **Set by:** Code.
- **Where:** gpu_presolve/src/rules_propagation.cpp:44-48, 130-139.
- **Switch:** no switch yet: needs one (named constant).
- **Alternatives:** 3; 64; no limit, with the improvement step as the only guard.
- **May change:** Chains of continuous bounds that converge slowly stop early; the fixed-point flag is lost (AS-271).
- **Test at the end:** First a count of the instances of the 240 that hit the limit. A cluster pair 16 against 64 only if some do.
- **State:** in the engine.

### AS-080 A combined coefficient is a structural zero below 1e-12 relative; small coefficients are not dropped
- **Assumption:** When a substitution adds two coefficients t and s*p, the entry is deleted if the absolute value of the sum is at most 1e-12 times the sum of the two absolute values; the same test zeroes a combined cost. The plan also names the removal of a small coefficient with a budget per row of 0.1 times the feasibility tolerance; no such rule is in the engine.
- **Set by:** Design (DESIGN.md section 4, lines 542-543).
- **Where:** gpu_presolve/include/gprs/numerics.hpp:37-38; gpu_presolve/src/rules_doubleton.cpp:134-142, 146-150.
- **Switch:** `options_t::tol.cancellation` (default 1e-12); no switch yet for the removal of small coefficients, which is not built.
- **Alternatives:** Exact zero only; 1e-9; dropping small coefficients as the plan describes.
- **May change:** Whether a residue of a cancellation stays as a tiny coefficient, which then keeps a lock on a column and can stop dual fixing and is bad for LP numerics, or a real small coefficient disappears.
- **Test at the end:** First a size run with 1e-12 against 0 and the correctness checks; a cluster pair only if reduced models differ.
- **State:** in the engine (cancellation); removal of small coefficients planned.

### AS-081 An equation is a row whose two sides are equal bit for bit
- **Assumption:** The doubleton rule and the subtraction of parallel rows take a row as an equation only when its lower side equals its upper side exactly. A row whose sides differ by 1e-12 is an inequality pair for them.
- **Set by:** Code.
- **Where:** gpu_presolve/src/rules_doubleton.cpp:198-199; gpu_presolve/src/rules_parallel_rows.cpp:359-362.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Sides within the feasibility tolerance count as an equation (the start step joins only sides that cross).
- **May change:** Rows whose sides differ by rounding are not used for substitution.
- **Test at the end:** First a count on the 240 of rows whose sides differ by more than 0 and at most 1e-6; a cluster pair only if there are any.
- **State:** in the engine.

### AS-082 Sides and bounds that cross by at most 1e-6 are joined, not reported as infeasible
- **Assumption:** In the start step row sides that cross by at most the feasibility tolerance are joined at the upper side, and column bounds likewise, after which the column is fixed. A later tightening that crosses the opposite bound by at most the tolerance fixes the column at that bound; the merge of parallel rows and the singleton row do the same with sides.
- **Set by:** Design (Stage 1 code structure).
- **Where:** gpu_presolve/src/presolve.cpp:83-117; gpu_presolve/src/rules_bounds.cpp:20-29, 38-47; rules_parallel_rows.cpp:341-353; rules_rows.cpp:107-115.
- **Switch:** no switch yet: needs one of its own (it uses `options_t::tol.feasibility`).
- **Alternatives:** Report infeasible on any crossing; fix at the midpoint of the crossing.
- **May change:** A model that is infeasible by less than 1e-6 is treated as feasible; the value at which the column is fixed is one end of the crossing.
- **Test at the end:** Not a quality pair: a correctness test on models that are infeasible by a small margin, and the status counts of every cluster arm.
- **State:** in the engine.

### AS-083 Forcing rows: tolerance over dynamism; with a continuous column only up to rounding
- **Assumption:** A row is forcing when an activity meets its side within the feasibility tolerance times the smallest coefficient over the larger of 1 and the largest coefficient. When the row has a free continuous column, it is forcing only if the activity can move towards the side by no more than its error bound plus 8 rounding units of its terms.
- **Set by:** Design for the tolerance (DESIGN.md section 4: "Forcing-row tolerance is feastol / dynamism"); code for the restriction on continuous columns (a repair; the reason is in the source comment, no HISTORY entry was found).
- **Where:** DESIGN.md section 4 (line 531); gpu_presolve/src/rules_rows.cpp:35-38, 140-190; gpu_presolve/include/gprs/rules.hpp:144-151.
- **Switch:** no switch yet: needs one (named constant `forcing_rounding_units` = 8); `options_t::enabled[rule::row_rules]` turns the whole family.
- **Alternatives:** The plain feasibility tolerance; forcing within the tolerance for continuous columns as well.
- **May change:** Fewer forcing rows on rows with continuous columns: those rows and columns stay in the model.
- **Test at the end:** Needs the switch. First a size run with the relaxed test and the correctness checks; then a standard cluster comparison if reduced models differ.
- **State:** in the engine.

### AS-084 Rows of integer columns are scaled to coprime integer coefficients within three limits
- **Assumption:** A row whose free columns are all integer is multiplied by a factor of at most `max_integral_scale` = 1000 and divided by the gcd of its coefficients when every scaled coefficient is within 1e-9 of an integer and at most `max_integral_coefficient` = 1e12, and the rounding of the coefficients moves the row by at most `coefficient_rounding_share` = 0.1 of the tolerance over the bounds of its columns.
- **Set by:** Design for the rule; code for the three constants.
- **Where:** gpu_presolve/src/rules_tightening.cpp:28-35, 53-72, 108-146, 169-205.
- **Switch:** no switch yet: needs one for the scaling alone; `options_t::enabled[rule::tightening]` (tool `--off tightening`) turns the whole family.
- **Alternatives:** No scaling; a larger scale limit; scaling without the division by the gcd.
- **May change:** The scale of all-integer rows changes (times up to 1000, or divided by the gcd): an absolute row tolerance of the solver then means something else in that row, and the LP sees other row norms. It is the precondition of the side rounding.
- **Test at the end:** Standard cluster comparison: row tightening on against off; with the sub-switch, scaling alone off.
- **State:** in the engine.

### AS-085 Bounds moved to an integer column through a doubleton equation round with tolerance over scale
- **Assumption:** When the bounds of a substituted column become bounds of an integer kept column, they are rounded with the slack feasibility / max(1, |scale|) plus 4 rounding units. When the kept column is fixed afterwards, the substituted column is fixed at its image, moved into its own bounds if rounding put it outside.
- **Set by:** Code (named constant `transfer_error_units` = 4).
- **Where:** gpu_presolve/src/rules_doubleton.cpp:38-40, 66-102, 228-234.
- **Switch:** no switch yet: needs one.
- **Alternatives:** The general integer rounding of the engine (tolerance 1e-6 whatever the scale).
- **May change:** One unit of an integer bound of the kept column.
- **Test at the end:** Not a quality pair: the campaign and the crush check of known solutions.
- **State:** in the engine.

### AS-086 Rows and columns are compared with at most 4 bucket neighbours in each direction
- **Assumption:** Inside one hash bucket of the parallel-row search a row is compared with at most `parallel_bucket_limit` = 4 rows above it and 4 below it; rows beyond that are not compared and the call reports "limited". The designed parallel-column rules reuse the option: 4 neighbours above and below, and at most 4 absorbers for a member of a bucket of singleton columns.
- **Set by:** Code for the rows (from the Stage 1 code structure, HISTORY Entry 14); design for the reuse (v2_design.txt:62, :65).
- **Where:** gpu_presolve/include/gprs/presolve.hpp:156-157; gpu_presolve/src/rules_parallel_rows.cpp:830-871; gpu_presolve/include/gprs/rules.hpp:279-284; stage1c_design/v2_design.txt:62, :65.
- **Switch:** `options_t::parallel_bucket_limit` (default 4).
- **Alternatives:** A higher limit (64); no limit, with the work budget as the only guard.
- **May change:** Parallel rows or columns in large buckets are missed and the fixed-point flag is lost.
- **Test at the end:** First a size run with 4 against 64 (rows, nonzeros, seconds) and the number of instances where a family reports "limited"; a cluster pair only if reduced models differ.
- **State:** in the engine for rows; designed for columns (milestone of dominance between parallel columns [M3]).

### AS-087 The subtraction of parallel rows is made only inside five guards
- **Assumption:** The subtraction is made only with a factor of at most 1 in size (exactly 1 when the equation has a singleton column), when the error growth factors are at most `max_growth` = 1e3, when no value involved exceeds 1e8, when the ratio of largest to smallest coefficient of the changed row does not rise above `max_dynamism` = 1e6 and above its old value, and, where an error would grow, only on a row that the presolve has not changed before. Of two equations the one with fewer singleton columns, then the smaller index, is subtracted.
- **Set by:** Design (made in the repairs of the rule, HISTORY Entries 18 to 31; Entry 23: "the repair takes only factors up to 1 in size"; the 1e6 follows the plan's veto for substitutions).
- **Where:** gpu_presolve/src/rules_parallel_rows.cpp:364-381, 410-479, 811-814; gpu_presolve/include/gprs/rules.hpp:308-338.
- **Switch:** no switch yet: needs one (named constants `max_growth` and `max_dynamism`); `options_t::near_parallel_rows` turns the whole step.
- **Alternatives:** Looser limits (factors above 1 with a wider error analysis, growth 1e6); no test on the history of the row.
- **May change:** Pairs that are left alone keep their nonzeros; a pair left because of the row's history costs the fixed-point flag.
- **Test at the end:** First a size run with relaxed guards and the correctness checks; a cluster pair only if the relaxed guards pass the checks and change reduced models.
- **State:** in the engine.

### AS-088 Where the column leaves the model, a bound is implied within about 2e-7
- **Assumption:** For the free singleton rules a bound of a column counts as implied when the bound that one row gives, recomputed at the test and moved outward by its rounding, is within `implied_tolerance` = 2 * (4 rounding units of the bound + 0.1 * feasibility tolerance), about 2e-7 for small bounds. Both ends of the column's range are tested. The shared rule since the design of the five further rules: a tolerance band is allowed only when the column then leaves the model; where bound and row both stay, only rounding is allowed.
- **Set by:** Design (v2_design.txt:34, :99; the shared rule is the lesson of HISTORY Entry 43 and of v2_amendments.txt:92-121).
- **Where:** stage1c_design/v2_design.txt:34, :99; v2_amendments.txt:92-121; gpu_presolve/src/implied.hpp:25-40, 169-181; gpu_presolve/include/gprs/rules.hpp:376-381, :79-81 (counter `free_singleton_rejected_explicit`).
- **Switch:** no switch yet: needs one; the band is a function of `options_t::tol.activity_error_share` and `options_t::tol.feasibility`.
- **Alternatives:** Rounding only; a wider band; HiGHS's absolute tolerances; bounds implied through several rows.
- **May change:** Which singleton columns count as free and are substituted; a restored value may pass its bound by the band; a band that is too wide removes a bound that was needed.
- **Test at the end:** First a count on the 240 of the decisions that the band changes, and the campaign. A standard cluster comparison (band against rounding only) needs the switch.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]); the function `implied_tolerance` is in the engine, and bound propagation already uses the same helper.

### AS-089 For dominance between parallel columns the absorber's bound is implied up to rounding only
- **Assumption:** For the dominance fixing between parallel columns [R6] the bound of the absorbing column counts as implied only up to rounding, not within the band of AS-088, because bound and row both stay in the model. The first row of the absorber that implies the bound is taken.
- **Set by:** Design (a binding amendment, v2_amendments.txt:108-112, from the design of the five further rules, with a model whose optimum goes from -0.25 to -0.2 under the band).
- **Where:** stage1c_design/v2_amendments.txt:108-112; v2_design.txt:62-63 (older wording with the band); lowworth_v2_design.txt:15, :224.
- **Switch:** no switch yet: needs one.
- **Alternatives:** The band (wrong on the model); a strict test over several rows.
- **May change:** Fewer dominance fixings where a bound is implied only within the band.
- **Test at the end:** A correctness choice, tested by the fixed oracle model; its size effect is measured at the milestone of dominance between parallel columns [M3]. No quality pair.
- **State:** designed (milestone of dominance between parallel columns [M3]).

### AS-090 Implied equation and locked compensator: a bound is implied up to rounding, with a cap
- **Assumption:** For the implied equation, and for a locked compensator of the dominated-column rule, a finite bound counts as implied by its row only when the row's candidate, moved inward by its own rounding slack (4 rounding units of its size plus the activity error), reaches the bound; no share of the feasibility tolerance is allowed. An amendment adds the cap of the forcing row: coefficient times the allowed excess is at most feasibility times the smallest coefficient over the larger of 1 and the largest coefficient of the row.
- **Set by:** Design (decision 2 of lowworth_v2_design.txt:239, replacing the plan's "strictly implied" and HiGHS's 1e-7; the cap is a binding amendment, lowworth_v2_amendments.txt:17-18; no owner answer in HISTORY up to Entry 46).
- **Where:** stage1c_design/lowworth_v2_design.txt:24, :81, :223, :239; lowworth_v2_amendments.txt:17-18; lowworth_v2_reviews.txt:22-25.
- **Switch:** no switch yet: needs one; the designed statistic `ie_blocked_bound` counts what the test refuses.
- **Alternatives:** HiGHS's 1e-7 for singleton columns; the plan's strict margin for columns with more than one entry; the first design's band of about 2e-7, which lost an optimum in the critic's model (-0.99999985 became -0.95).
- **May change:** On instances with non-integer data the engine can make fewer implied equations and fewer dominated-column fixings than HiGHS.
- **Test at the end:** First a size table (milestones of the implied equation [L1] and of the dominated column [L5] report `ie_blocked_bound` per instance on the 240). A cluster pair only after a looser test is built behind a switch.
- **State:** designed (milestones of the implied equation [L1] and of the dominated column [L5]).

### AS-091 Simple probing uses the engine's forcing-row test, not HiGHS's absolute 1e-9
- **Assumption:** Simple probing on an equation [R1] accepts a row when both conditional rows are forcing by the engine's test (AS-083): within tolerance over dynamism, and with a continuous column in the row only a gap of rounding size.
- **Set by:** Design (v2_design.txt:27-28: HiGHS's value "is not copied").
- **Where:** stage1c_design/v2_design.txt:27-28; gpu_presolve/src/rules_rows.cpp:38, 153-179.
- **Switch:** no switch yet: needs one.
- **Alternatives:** HiGHS's absolute 1e-9; the feasibility tolerance for continuous columns as well.
- **May change:** Fewer aliases on equations with continuous columns whose data are off by more than rounding.
- **Test at the end:** First a count of the rows refused by the rounding restriction on highschool1-aigio, neos-1445765 and brazil3 at the milestone of simple probing [M2]; a cluster pair needs the switch.
- **State:** designed (milestone of simple probing [M2]).

### AS-092 A column is replaced by a binary only when its range is at most 1e6
- **Assumption:** Simple probing [R1] and the single-lock alias [R8] replace a column by a binary only when its range upper minus lower is at most `alias_scale_limit` = 1e6; the range multiplies every coefficient of the column into the rows of the binary. The design names this the only guard of those substitutions, and the free singleton rules move cost with only the cancellation rule: there is no limit on the coefficient range of a resulting row or of the objective, unlike the plan's veto for aggregator substitutions (a ratio above the larger of the old one and 1e6, AS-110).
- **Set by:** Design (v2_design.txt:27, :76, :115, :269; a named constant, not an option).
- **Where:** stage1c_design/v2_design.txt:27, :76, :115, :269; gpu_presolve/include/gprs/rules.hpp:391-394; DESIGN.md section 4 (lines 540-541).
- **Switch:** no switch yet: needs one (named constant `alias_scale_limit`).
- **Alternatives:** A smaller limit such as 1e3 or 1e4; a veto when a row's ratio of largest to smallest coefficient exceeds the larger of its old value and 1e6; no limit.
- **May change:** Coefficients of up to 1e6 times the old ones enter the rows of a binary: a weaker and numerically harder formulation of big-M kind, against one column less; badly scaled rows or costs for cuOpt's LP, more so with scaling off.
- **Test at the end:** Standard cluster comparison after the milestones of simple probing [M2] and of dual bound strengthening and the single-lock alias [M4]: 1e6 against 1e3 (two benchmark commits that differ in the constant), plus the count of aliases with a range above 1e3 and the coefficient range of the reduced rows and costs per instance on the 240. A second pair once a veto on the coefficient range exists with a switch.
- **State:** designed (milestones of simple probing [M2] and of dual bound strengthening and the single-lock alias [M4]); the constant is in the engine and no rule reads it yet.

### AS-093 One shared integrality test: ratios within 1e-9 and a drift guard of 0.1 tolerance
- **Assumption:** A row is "integral for" a column when every other free column is integer, each ratio and the side over the coefficient is within 1e-9 of an integer, and the drift of the rounded ratios over the bounds of the other columns is at most 0.1 times the feasibility tolerance. For a column that stays in the model (conversion, scaling) the drift is measured in the units of that column; row units are enough where the column leaves or moves only in its free direction (free singleton rules for an integer column [R2, R5], integer source of the implied equation). Otherwise the column is left.
- **Set by:** Design (shared function `integral_for`, binding by v2_amendments.txt:97-101 and lowworth_v2_amendments.txt:11-14, after the validity review found that the first form loses an optimum for a coefficient below 0.1: -9989.999999995 became -9989.5).
- **Where:** stage1c_design/v2_design.txt:34; v2_amendments.txt:97-101; lowworth_v2_design.txt:24, :81, :152; lowworth_v2_amendments.txt:11-14; the guard to copy is gpu_presolve/src/rules_tightening.cpp:35, 126-145.
- **Switch:** no switch yet: needs one; named constant `coefficient_rounding_share` = 0.1 with `options_t::tol.zero` and `options_t::tol.feasibility`; the designed statistic `ii_refused_drift` counts refusals.
- **Alternatives:** No drift guard (a ratio test only, which loses an optimum in a fixed test of the design: -1499999 becomes about -2000); another share; the integrality tolerance 1e-5; leave every integer singleton column.
- **May change:** How many integer columns are removed by substitution and how many columns are converted, scaled or accepted as integer sources on models with non-integer coefficients.
- **Test at the end:** Not a quality pair: a correctness guard, tested by two fixed models of the design in both modes and by two wrong engines. Quality is covered by the pairs of the rules that use it.
- **State:** designed (to be built with the milestone of the singleton-column rules and the forcing column [M1]; binding for the implied equation [L1] and the implied-integer conversion [L2]).

### AS-094 A chain of free singleton substitutions may grow a deviation by at most 1000
- **Assumption:** A free singleton column is substituted only when the composed growth factor of its chain (the largest coefficient of its pivot row over its own, times the factor earlier substitutions gave the column) is at most `pivot_growth_limit` = 1000. The undo never clamps, so a restored value can pass its bound by up to 1000 times what the reduced point is off: 1e-3 at a point that uses the full tolerance.
- **Set by:** Design (repair after the validity critique, v2_answers.txt item 6).
- **Where:** stage1c_design/v2_design.txt:34-36, :269; v2_answers.txt:23-26; gpu_presolve/include/gprs/presolve.hpp:134-136; gpu_presolve/include/gprs/rules.hpp:73-75.
- **Switch:** `options_t::pivot_growth_limit` (default 1000).
- **Alternatives:** 10 or 100 (the plan's general pivot rule is 0.01 of the row maximum); 1e6; clamping in the undo.
- **May change:** A lower limit leaves more columns and rows; a higher one lets the postsolve multiply a deviation of the reduced solution by more, so restored values can violate the original model, which cuOpt tests (AS-168).
- **Test at the end:** Standard cluster comparison after the milestone of the singleton-column rules and the forcing column [M1]: 1000 against 10, with the count of postsolved solutions that fail the feasibility check in each arm.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]); the option is in the engine and no rule reads it yet.

### AS-095 A free singleton substitution has no limit on the cost entries it creates
- **Assumption:** A free singleton substitution [R2, R5] deletes its pivot row and gives up to row length minus 1 columns a nonzero cost. There is no fill or row-length limit, only the statistic `cost_fill`. The size gate counts matrix nonzeros (AS-210).
- **Set by:** Design (v2_design.txt:34, :37, :197).
- **Where:** stage1c_design/v2_design.txt:34, :37, :197.
- **Switch:** no switch yet: needs one for a limit; `options_t::sub_enabled[sub_rule::free_singleton]` and `[sub_rule::free_singleton_dual]` turn the rules.
- **Alternatives:** Refuse rows above a length or a cost-fill limit.
- **May change:** A dense objective in place of a long row (in HiGHS, mcsched, swath1, swath3 and neos-3627168-kasai each lose one to three long rows this way): objective-driven heuristics and the LP see many more cost entries.
- **Test at the end:** Standard cluster comparison after the milestone of the singleton-column rules and the forcing column [M1]: both free singleton reductions on against off; a third arm with a row-length limit once one is built.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]).

### AS-096 The zero-cost singleton rule and the forcing column need a cost of exactly zero
- **Assumption:** Both rules [R3, R4] require the cost to be exactly 0; HiGHS allows a small tolerance for the forcing column.
- **Set by:** Design (v2_design.txt:41, :48-49; named among the "weaker forms" in :261).
- **Where:** stage1c_design/v2_design.txt:41, :48-49, :261.
- **Switch:** no switch yet: needs one.
- **Alternatives:** A cost tolerance as in HiGHS.
- **May change:** Columns with a tiny cost stay in the model.
- **Test at the end:** First a count on the 240 of singleton columns with an absolute cost above 0 and below 1e-7. A count of zero closes the entry for the benchmark set; otherwise a cost tolerance behind a switch and a standard cluster comparison, read on the instances whose reduced model differs.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]).

### AS-097 Postsolve records may save rows up to 4 times the nonzeros of the input model
- **Assumption:** The records of the singleton-column rules may save entries of rows that stay in the model up to `saved_row_factor` = 4 times the nonzeros of the input model; beyond that such a row is left and the family reports "limited". The five further rules count their saved rows, candidates, entries and terms against the same limit. Whether any of the 240 reaches it is not measured.
- **Set by:** Design (v2_design.txt:44, :115; reused in lowworth_v2_design.txt:27, :34, :81).
- **Where:** stage1c_design/v2_design.txt:44, :115; lowworth_v2_design.txt:27, :34, :48, :55, :62, :81; gpu_presolve/include/gprs/presolve.hpp:129-132.
- **Switch:** `options_t::saved_row_factor` (default 4).
- **Alternatives:** A larger factor or no limit; a limit per record kind.
- **May change:** Memory of the postsolve stack; on a row visited many times the rule stops, later implied equations and conversions are refused, and the fixed-point flag is lost.
- **Test at the end:** First a count of the instances that reach the limit on the 240 and the 40 (reported at the milestone of the singleton-column rules and the forcing column [M1]). If any does: standard cluster comparison, factor 4 against a larger factor.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1] on); the option is in the engine and no rule reads it yet.

### AS-098 Two columns are parallel within 1e-9 relative and 0.1 of the tolerance over the bounds
- **Assumption:** Two columns are parallel when they have the same live rows and one ratio s such that, in every row, the two coefficients differ from proportionality by at most 1e-9 relative, and that difference times the largest finite bound of the fixed column is at most 0.1 of the feasibility tolerance.
- **Set by:** Design (v2_design.txt:62).
- **Where:** stage1c_design/v2_design.txt:62; gpu_presolve/include/gprs/numerics.hpp:25, :31.
- **Switch:** `options_t::tol.zero`, `options_t::tol.activity_error_share`, `options_t::tol.feasibility` (shared with other tests).
- **Alternatives:** Exact proportionality; HiGHS's tolerances.
- **May change:** The number of pairs found on models with rounded data.
- **Test at the end:** First a count of the pairs that pass only through the tolerance on the 240; no quality pair expected.
- **State:** designed (milestone of dominance between parallel columns [M3]).

### AS-099 Costs of parallel columns count as equal only to 1e-12 relative
- **Assumption:** The cost difference of a parallel pair counts as zero when it is at most 1e-12 times the sum of the two cost terms; the dominance direction uses the exact sign otherwise, and a merge needs a zero difference. HiGHS accepts a difference up to its dual tolerance.
- **Set by:** Design (v2_design.txt:62-63, :83-84; named among the weaker forms in :261; chosen so that the objective identity of the checks holds).
- **Where:** stage1c_design/v2_design.txt:62-63, :83-84, :261.
- **Switch:** `options_t::tol.cancellation` (default 1e-12; shared with the structural-zero test).
- **Alternatives:** The dual tolerance 1e-7 as in HiGHS (the checks then need an exception).
- **May change:** Fewer merges and fewer two-way dominance cases on models whose costs agree only approximately.
- **Test at the end:** First a count on the 240 of parallel pairs with a cost difference between 1e-12 relative and 1e-7. A count of zero closes the entry for the benchmark set; otherwise the looser tolerance behind a switch and a standard cluster comparison, read on the instances whose reduced model differs.
- **State:** designed (milestones of dominance between parallel columns [M3] and of the merge of parallel columns [M5]).

### AS-100 Dual bound strengthening rounds to the safe side and needs an improvement above the tolerance
- **Assumption:** In dual bound strengthening a row's value is moved to the safe side by the candidate's rounding and, for an integer column, rounded as ceil(w - feasibility / max(1, |a|)); the new bound is taken only when it improves the old one by more than the feasibility tolerance. The attempt ends without a result at the first locking row whose rest is infinite or fails a guard.
- **Set by:** Design (v2_design.txt:69-70; the integer rounding follows HiGHS).
- **Where:** stage1c_design/v2_design.txt:69-70.
- **Switch:** no switch yet: needs one of its own (it uses `options_t::tol.feasibility`).
- **Alternatives:** Skip a row without a bound only when it does not lock; a relative improvement threshold.
- **May change:** The number of bound changes.
- **Test at the end:** Covered by the pair `dual_bound` on against off (AS-050).
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4]).

### AS-101 The single-lock alias forces a continuous column only up to rounding
- **Assumption:** For a continuous column the force test of the single-lock alias allows a gap of rounding size only (the form of the forcing rule), not the feasibility tolerance divided by the column's largest coefficient as version 2 of the design wrote. HiGHS is as loose as version 2.
- **Set by:** Design (a binding amendment after the validity review, v2_amendments.txt:61-62, with a model on which two aliases add up and the optimum goes from 9.9999982 to 100).
- **Where:** stage1c_design/v2_amendments.txt:61-62; v2_reviews.txt:38-41.
- **Switch:** no switch yet: needs one.
- **Alternatives:** The feasibility tolerance, as HiGHS.
- **May change:** Fewer aliases of continuous columns on rows that are tight only within tolerance.
- **Test at the end:** A correctness choice (a fixed oracle case with a wrong engine); a count of the aliases lost on the 240.
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4]).

### AS-102 The single-lock alias has no limit on the length of its lock row
- **Assumption:** `lock_alias_row_limit` = 0 means no limit on the free entries of the row through which a column is aliased. HiGHS leaves rows with more than 1000 entries; the design says that cap is not needed for the cost.
- **Set by:** Design (v2_design.txt:79, :115).
- **Where:** stage1c_design/v2_design.txt:79, :115; gpu_presolve/include/gprs/presolve.hpp:138-139.
- **Switch:** `options_t::lock_alias_row_limit` (default 0, no limit).
- **Alternatives:** 1000, as HiGHS.
- **May change:** Aliases through long rows: more columns removed, many entries put into the rows of one binary.
- **Test at the end:** Standard cluster comparison after the milestone of dual bound strengthening and the single-lock alias [M4]: 0 against 1000, after a size run has shown that reduced models differ.
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4]); the option is in the engine and no rule reads it yet.

### AS-103 An implied-equation source needs a single lock and a cost beyond the dual tolerance
- **Assumption:** A column is a source only if its lock count in the cost direction is exactly 1, the lock row is not already an equation, and its absolute cost exceeds the dual tolerance 1e-7 times its coefficient. A zero cost, a cost inside the tolerance or two locks are not taken.
- **Set by:** Design (lowworth_v2_design.txt:24, HiGHS's strict test taken over).
- **Where:** stage1c_design/lowworth_v2_design.txt:24; gpu_presolve/include/gprs/numerics.hpp:24.
- **Switch:** `options_t::tol.dual` (default 1e-7); no switch yet for the lock count.
- **Alternatives:** Accept zero-cost columns (weak form); accept two locks with further tests; another dual tolerance.
- **May change:** How many rows become equations.
- **Test at the end:** Covered by the pair of AS-063; a tolerance sweep would be a size run on the 240 first.
- **State:** designed (milestone of the implied equation [L1]).

### AS-104 A conversion rounds bounds inward and is refused where rounding would widen a bound
- **Assumption:** At an implied-integer conversion both bounds are set to their integer-rounded values. If a finite bound lies within the tolerance inside an integer without being equal to it, the rounded value would be wider and the conversion is refused. The weak test requires bounds that are exactly integers or infinite.
- **Set by:** Design (lowworth_v2_answers.txt:82 and :22, the worth critic's second option).
- **Where:** stage1c_design/lowworth_v2_design.txt:31, :130; lowworth_v2_answers.txt:22, :82; gpu_presolve/include/gprs/numerics.hpp:129-137.
- **Switch:** no switch yet: needs one; the designed statistic `ii_refused_bound` counts refusals.
- **Alternatives:** Widen the bound to the rounded value; tighten to the next integer inside.
- **May change:** Columns with bounds such as [1e-7, 6] are not converted; a few conversions that HiGHS makes may be missing.
- **Test at the end:** Not a cluster pair; the milestone of the implied-integer conversion [L2] reports `ii_refused_bound` per instance on the 240 and the 40.
- **State:** designed (milestone of the implied-integer conversion [L2]).

### AS-105 The column scaling works inside fixed limits
- **Assumption:** The integral scale of the row is at most 1000; the column factor must lie between 1/1024 and 1024 (`col_scale_limit`); every scaled coefficient and the scaled cost must be finite, nonzero and at most 1e12; a scaled bound within 4 rounding units of an integer is snapped to it (the only place where a bound may widen); any other bound that integer rounding would widen refuses the scaling. The proof is made on exactly the numbers the state will store.
- **Set by:** Design (lowworth_v2_design.txt:38, :70; the last sentence is a binding amendment, lowworth_v2_amendments.txt:19).
- **Where:** stage1c_design/lowworth_v2_design.txt:38-39, :70; lowworth_v2_amendments.txt:19; gpu_presolve/src/rules_tightening.cpp:29-32.
- **Switch:** no switch yet: needs one (named constants `col_scale_limit` = 1024 and `max_integral_scale` = 1000).
- **Alternatives:** Other limits; HiGHS's own limits (not quoted in the design).
- **May change:** Which columns are scaled, and how far the coefficient range of a scaled column moves (up to a factor of 1024).
- **Test at the end:** Part of the pair of AS-066; the milestone of the implied-integer conversion [L2] reports scalings per instance.
- **State:** designed (milestone of the implied-integer conversion [L2], second step).

### AS-106 The sums of column stuffing and the single-equation rule allow rounding only
- **Assumption:** The tests that decide a stuffing fixing and a single-equation fixing accept an excess of the activity over the side only up to the error bound of the sum plus 8 rounding units of the terms, and at most feasibility times the smallest coefficient over the larger of 1 and the largest. HiGHS accepts 1e-6 in both rules.
- **Set by:** Design (part of decision 8 of lowworth_v2_design.txt:245; the cap is a binding amendment, lowworth_v2_amendments.txt:17-18; no owner answer in HISTORY up to Entry 46).
- **Where:** stage1c_design/lowworth_v2_design.txt:45-46, :52-53, :223, :245; lowworth_v2_amendments.txt:17-18; gpu_presolve/src/rules_rows.cpp:38, 175-179.
- **Switch:** no switch yet: needs one (named constant `forcing_rounding_units` = 8).
- **Alternatives:** HiGHS's 1e-6; the feasibility tolerance.
- **May change:** Fewer fixings than HiGHS on rows whose slack lies between rounding and 1e-6; the reduced model never needs the tolerance to stay feasible.
- **Test at the end:** First a size table (milestones of column stuffing [L3] and of the single-equation extension [L4] report fixings next to HiGHS's counters). A cluster pair needs a looser test behind a switch.
- **State:** designed (milestones of column stuffing [L3] and of the single-equation extension [L4]).

### AS-107 The dominated-column rule limits prices to 1e7 and coefficient ratios to 1000
- **Assumption:** A compensator's price (cost over coefficient) is admissible only when finite and at most `price_limit` = 1e7 in absolute value, which is HiGHS's limit, and the ratio of the candidate's coefficient to the compensator's must be at most `comp_ratio_limit` = 1000.
- **Set by:** Design (lowworth_v2_design.txt:59, :70).
- **Where:** stage1c_design/lowworth_v2_design.txt:59, :70.
- **Switch:** no switch yet: needs one (named constants).
- **Alternatives:** Other limits; no ratio limit.
- **May change:** Fixings with large prices or badly scaled compensations are not made; the ratio limit bounds how far a compensator moves in postsolve and crush.
- **Test at the end:** No cluster pair; the milestone of the dominated column [L5] reports fixings per instance against HiGHS's counters on the 240 and the 40.
- **State:** designed (milestone of the dominated column [L5]).

### AS-108 The sign of the dominated-column rate is tested to 1e-12 relative, with a capped rise
- **Assumption:** The rate of a column is the change of the objective per unit that the column moves towards the bound: its own cost in that direction less the price of the compensating columns in its locking rows. The column is fixed when its rate is at least -1e-12 times the size of its cost terms (HiGHS: 1e-7). A negative rate is accepted only when the other bound of the column is finite and the rise of the objective is within the identity tolerance of the checks; with an infinite other bound the rate must be at least 0.
- **Set by:** Design (lowworth_v2_design.txt:59-60; part of decision 8; the cap is a binding amendment, lowworth_v2_amendments.txt:21-23).
- **Where:** stage1c_design/lowworth_v2_design.txt:59-60, :245; lowworth_v2_amendments.txt:21-23; lowworth_v2_reviews.txt:37-40; gpu_presolve/include/gprs/numerics.hpp:38.
- **Switch:** `options_t::tol.cancellation` (default 1e-12); no switch yet for the cap.
- **Alternatives:** HiGHS's 1e-7; an exact sign test.
- **May change:** Columns whose cost equals the compensation cost up to rounding are fixed; a looser tolerance would fix more and could raise the objective of a feasible point.
- **Test at the end:** No cluster pair; the size table of the milestone of the dominated column [L5]. Loosening needs its own fixed test first.
- **State:** designed (milestone of the dominated column [L5]).

### AS-109 The five rules' dual tolerances stay stricter than HiGHS until their price is measured
- **Assumption:** The design keeps tolerances "so that the checks need no exception and a reduced model never needs the tolerance to stay feasible": rounding only in the implied test and in the sums of stuffing and the single equation (HiGHS 1e-7 and 1e-6), the sign of the reduced-cost bound exact to 1e-12 relative (HiGHS 1e-7), no status from an infinite target. They stay until the table of the milestone of the dominated column [L5] shows per instance what they cost against HiGHS.
- **Set by:** Design (decision 8 of lowworth_v2_design.txt:245; no owner answer in HISTORY up to Entry 46).
- **Where:** stage1c_design/lowworth_v2_design.txt:223, :245.
- **Switch:** no switch yet: needs one per tolerance (the design: each "can then be loosened with its own fixed test").
- **Alternatives:** HiGHS's tolerances, paid with exceptions in the oracle and the crush check.
- **May change:** Fewer reductions than HiGHS on instances with non-integer data, against a reduced model that is feasible without leaning on a tolerance.
- **Test at the end:** First the per-instance table of the milestone of the dominated column [L5] on the 240 and the 40. A standard cluster comparison only after a looser variant is built behind a switch.
- **State:** designed (milestones of the five further rules [L1 to L5]).

### AS-110 Aggregator substitutions: pivot at least 0.01, fill-in at most 10, coefficient ratio at most 1e6
- **Assumption:** For substitutions a row or column of size 2 is always taken; otherwise the pivot must be at least 0.01 of the largest entry of its row or column and the net fill-in at most 10; all candidates are tested; a pivot is vetoed when a resulting row would have a ratio of largest to smallest coefficient above the larger of its old value and 1e6. The numbers are not tied to a measurement in the plan.
- **Set by:** Design (guards from the reviews, DESIGN.md section 4).
- **Where:** DESIGN.md section 4 (lines 535-541).
- **Switch:** no switch yet: needs one (the aggregator is not built).
- **Alternatives:** HiGHS's limits (its aggregator pass ends after three fill-in failures); other thresholds.
- **May change:** Fill-in trades rows and columns for nonzeros and denser rows.
- **Test at the end:** Standard cluster comparison when built: aggregator off against on, and fill-in limit 0 against 10.
- **State:** planned (step 2 of the build order).

### AS-111 Probing is limited by work in entries, a stall stop and caps per pass
- **Assumption:** The probing work budget counts matrix entries touched per nonzero of the model, not probes; a pass ends when work passes without a finding; new cliques and new implications are capped per pass, and reaching a cap does not end probing. The sizes are open. Basis on the 31: HiGHS probes 2.0% of its first-round candidates; probing all leaves 1.8% fewer nonzeros in the geometric mean at 15 times the probing time.
- **Set by:** The owner for the principle (change 5 of the Stage 0 report, accepted with decision 6, HISTORY Entry 13); the sizes are open.
- **Where:** DESIGN.md section 7 (lines 634-648) and section 13 (lines 809-811); gpu_presolve/results/stage0/STAGE0_REPORT.md section 13 (lines 626-656).
- **Switch:** no switch yet: needs one; on the HiGHS side `HIGHS_DIAG_PROBING_NOSTOP=1`.
- **Alternatives:** HiGHS's limits counted in probes (it ends probing on its clique limit on two of the 31).
- **May change:** How many candidates are probed, and with them the fixings, implications and cliques the solver gets, whose value is not in the nonzero count.
- **Test at the end:** Before it is built: cuOpt with its presolver set to none on the 240 as presolved by HiGHS, `probing` on against off, and HiGHS's probing limits on against off (`HIGHS_DIAG_PROBING_NOSTOP=1` against unset; the family pairs of AS-113). When built: standard cluster comparison, the default budget against one large enough for every candidate.
- **State:** planned (step 4 of the build order).

## 4. Order of the rules, rounds, budgets, stopping

### AS-112 At run time all rules are phases of one loop with fixed alternations
- **Assumption:** "Stage" is a development stage. At run time the cheap cascade and the pairwise and structural phases run until they stall (less than 0.05% of rows and of columns removed), then the wide phases (probing, cliques, cuOpt's special reductions) run, and when they changed the model the loop returns to the cascade, for a fixed number of alternations at most. The plan gives no number of alternations.
- **Set by:** Design (DESIGN.md amendment A7), confirmed to the owner, who asked on 2026-11-27 (HISTORY Entry 26): "The stages run in a loop, right?"
- **Where:** DESIGN.md amendment A7 (lines 87-91) and section 3 (lines 492-495).
- **Switch:** no switch yet: needs one for the stall share and the alternations; today the loop is bounded by `options_t::max_rounds` and `options_t::max_cascade_passes` (AS-118).
- **Alternatives:** Run to a fixed point without a cap; wide phases once (HiGHS runs several stages once); another stall share.
- **May change:** How far the presolve goes once wide phases exist: a low cap leaves reductions undone, a high one costs time.
- **Test at the end:** When the wide phases exist: alternations 1 against the default, first as a size run, then as a standard cluster comparison.
- **State:** planned (the Stage 1 rounds are in the engine).

### AS-113 After Stage 1 rules are built in order of nonzero worth on the 240
- **Assumption:** The order is: cheap column rules, then sparsify and the aggregator, then pairwise dominated columns, then enumeration, probing and cliques. It replaces the earlier order that put the GPU-wide phases (probing, cliques) before the parity rules, which the owner had approved on the assistant's recommendation (HISTORY Entries 6 and 13). The basis is one measurement: on the 240, probing and enumeration take 65% of HiGHS's time for under 2% of the nonzeros in the mean. Inside the cheap column rules the build order is by cost and by what later rules need, not by measured worth: the groundwork [M0], the singleton-column rules and the forcing column [M1], simple probing [M2], dominance between parallel columns [M3], dual bound strengthening and then the single-lock alias [M4], the merge of parallel columns [M5]; then the five further rules in the order of build cost: implied equation [L1], implied-integer conversion [L2], column stuffing [L3], single-equation extension [L4], dominated column [L5]. By worth the dominated column would follow the conversion; the order of the five is the design's, and the owner has not answered on it. Nothing of the five is built before the nine are finished. If the stage stops early, the rules built so far are the ones that can be tested.
- **Set by:** The owner (HISTORY Entry 29, 2026-11-27: "Yes, cheap rules first").
- **Where:** DESIGN.md amendment A8 (lines 93-112), replacing the stage order of section 0 (line 200) and section 13; HISTORY Entries 28 and 29. Build order inside the stage: DESIGN.md amendment A8 (lines 120-124 against 133-138); stage1c_design/v2_amendments.txt:19-23; v2_reviews.txt:124; lowworth_v2_design.txt:17, :194-218; lowworth_v2_amendments.txt:40-41; lowworth_v2_reviews.txt:99-102; HISTORY Entry 38 (lines 2147-2150).
- **Switch:** no switch of its own: a build order. At run time each family has its switch (`options_t::enabled`, `options_t::sub_enabled`), and on the HiGHS side each family has a name in `HIGHS_DIAG_OFF`, which gives a test before the family is built.
- **Alternatives:** Probing and cliques first (plan version 4): the GPU-wide work and the rules most likely to strengthen without removing nonzeros; an order by worth on the 40, which ranks the rules differently (AS-239).
- **May change:** Which rules exist when the end test runs. Until the last step exists the new presolve has no probing, no clique merging, no pairwise dominated columns and no aggregator, which PaPILO in cuOpt has. A worth measured in nonzeros understates rules whose effect is on strength (implications, clique rows, fixings).
- **Test at the end:** A test that can be run now, before the families are built: one pair per HiGHS family, each two arms of cuOpt with its presolver set to none (`presolve` = 0, `mip_scaling` 0) on the 240 as presolved by HiGHS with the family on and with it off: `probing`, `enumeration`, `clique_merging`, `clique_substitutions`, `sparsify`, `aggregator`, `dominated_columns`, `strengthen_inequalities`, `parallel_columns` in `HIGHS_DIAG_OFF`, and for the probing budget `HIGHS_DIAG_PROBING_NOSTOP=1` against unset. The two arms of a pair differ in one HiGHS switch and are scaled alike by HiGHS. The result is a ranking of the families by solution quality next to the ranking by nonzeros of HISTORY Entry 28. Its limit: with the presolver set to none cuOpt's own presolve (bound update, probing cache, block BVE) does not run either (AS-142). When the families exist in the engine: one-family-off arms of the standard cluster comparison, and a pair "wide phases off against on".
- **State:** in force; the cheap column rules are in work.

### AS-114 A round runs the cascade, parallel rows, doubleton equations, then the column families
- **Assumption:** A round is: the cascade (row rules, bound propagation, row tightening, then dual fixing on the locks they leave, repeated until their lists are empty), then parallel rows, then doubleton equations. Since the groundwork four column families follow as phases of their own, still empty: simple probing, singleton columns, dual columns, parallel columns. They come last "so that the first round does on the input model what it did without them", and dual fixing stays in the cascade and so runs before every column rule. Inside a family rows are served in the order of their first event.
- **Set by:** Design (Stage 1 code structure, HISTORY Entries 14 and 15; for the column families v2_design.txt:123).
- **Where:** DESIGN.md section 3 (lines 479-495); gpu_presolve/src/presolve.cpp:287-303, 334-363; gpu_presolve/include/gprs/presolve.hpp:264-270 (the four column phases are presolve.cpp:349-355).
- **Switch:** no switch yet: needs one for the order; a family can only be switched off (`options_t::enabled`).
- **Alternatives:** HiGHS's order (singleton rows, doubleton equations and column singletons inside one fast loop); doubleton equations before parallel rows; column rules inside the cascade or before the parallel rows; dual fixing after the column rules.
- **May change:** Where two reductions exclude each other the order decides which is applied: a column that dual fixing fixes is no longer there for an alias or a merge, and a substitution changes which rows are parallel. The engine is ahead of the reference on nine instances through the row addition, which reads singleton columns that the free singleton rules remove from the second round on.
- **Test at the end:** Needs the switch. Proposed pair once the column rules exist: dual fixing in the cascade against after the column rules; a size run first (no instance worse than the Stage 1 engine), then a standard cluster comparison if reduced models differ.
- **State:** in the engine (the row phases and four column families that are still empty).

### AS-115 The two families of the five rules run last in a round, the conversion last
- **Assumption:** With the five rules a round ends with the exchange family (single equation, dominated column) and then the implied-integer family. The conversion is last "so that in every round the parallel columns, the zero-cost singleton rule and the dual rules see a column while it is still continuous". Stuffing is a step of the singleton-column family and the implied equation a test of the dual-column family.
- **Set by:** Design (lowworth_v2_design.txt:91; the conversion's place is part of its decision 7).
- **Where:** stage1c_design/lowworth_v2_design.txt:91, :187, :227.
- **Switch:** no switch yet: needs one for the order.
- **Alternatives:** Conversion earlier (as HiGHS, where it blocks parallel-column reductions on blp-ic98 and blp-ar98); exchange family before the dual columns.
- **May change:** Which of two competing reductions wins on a column (merge of parallel columns or conversion; zero-cost singleton or conversion). Predicted: the engine keeps the smaller size on blp-ic98 and blp-ar98.
- **Test at the end:** First a size run (milestone of the implied-integer conversion [L2]: the two blp instances not above their size before it). A cluster pair would need a switch for the phase order.
- **State:** designed.

### AS-116 Inside a column family the reductions are tried in a fixed order
- **Assumption:** Singleton family: per row visit the free singleton with an equation pivot, else with an inequality pivot, else the zero-cost singleton rule; the forcing column for listed columns that are no singletons. Dual family: dual bound strengthening for all candidates, then the single-lock alias, which collects its rows after the column pass. Parallel family: dominance, then merge. Candidates are taken in index order.
- **Set by:** Design (v2_design.txt:117-121; v2_amendments.txt:113-114).
- **Where:** stage1c_design/v2_design.txt:117-121; v2_amendments.txt:113-114.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Other orders, for example the zero-cost rule before the free singleton.
- **May change:** Which reduction takes a column that several could take.
- **Test at the end:** No pair planned; covered by the on-against-off pairs of the reductions (AS-050).
- **State:** designed (milestones of the nine column rules [M1 to M5]).

### AS-117 The column rules run both before and after the switch to explicit bounds
- **Assumption:** The four column families run in the rounds before the implied bounds become explicit, where a continuous column still has its input bounds, and in the rounds after. The loss is assumed to be limited to columns that become singletons after the switch.
- **Set by:** Design (v2_answers.txt item 32).
- **Where:** stage1c_design/v2_design.txt:123; v2_answers.txt:127-130.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Column rules only before the switch; the switch at the very end.
- **May change:** Singleton columns whose bound propagation tightened fail the implied-free test; the counter `free_singleton_rejected_explicit` counts them.
- **Test at the end:** First a size run when the singleton-column rules are built: the counter `free_singleton_rejected_explicit` on the 240 and the 40. A counter of zero on the 240 closes the entry for the benchmark set. Otherwise an order switch has to be built, and a size run with the hash of the reduced model at both values decides: identical hashes on all 240 close the entry, any difference sends it to the standard cluster comparison, read on the instances that differ.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]).

### AS-118 The presolve has at most 20 rounds in all and 100 cascade passes per round
- **Assumption:** `max_rounds` = 20 counts the rounds of the whole presolve, those after the switch to explicit bounds included; `max_cascade_passes` = 100 limits the passes over the four cascade families within one round. The round limit is not raised for the column rules, although each alternation such as singleton column, lock rule, redundant rows costs one round. Today no instance of the 240 ends on it (the largest count is 13).
- **Set by:** Code for the values; taken as recommended for not raising it (v2_amendments.txt:90: "max_rounds is not raised"; decision 6 of v2_design.txt:281 says raising it needs the owner's word).
- **Where:** gpu_presolve/include/gprs/presolve.hpp:141-145; gpu_presolve/src/presolve.cpp:291, :337; stage1c_design/v2_design.txt:125, :208, :267, :281.
- **Switch:** `options_t::max_rounds` (default 20) and `options_t::max_cascade_passes` (default 100).
- **Alternatives:** Higher limits; a separate limit for the rounds after the switch to explicit bounds.
- **May change:** An instance that ends on a limit keeps reductions undone and loses the fixed-point flag.
- **Test at the end:** First a count of the instances whose result has 20 rounds, after each milestone, and a size run 20 against 100. A standard cluster comparison 20 against a small value such as 2, and 20 against 100 if any instance ends on it.
- **State:** in the engine.

### AS-119 A new equation or converted column reaches the other rules only in the next round
- **Assumption:** An implied equation reaches the doubleton equations and the free singleton rule only in the next round, and a converted column reaches dual bound strengthening only in the next round; likewise the findings of the column families reach the cascade one round later. Chains through the new rules can therefore end on the round limit.
- **Set by:** Design (a consequence named as a risk, lowworth_v2_design.txt:232).
- **Where:** stage1c_design/lowworth_v2_design.txt:232; v2_design.txt:125.
- **Switch:** `options_t::max_rounds` bounds it; no switch yet for applying follow-up rules inside the same round.
- **Alternatives:** Follow-up rules inside the same round; a higher round limit.
- **May change:** Reductions lost on the round limit; the milestones of the implied equation [L1] and of the implied-integer conversion [L2] report the rounds (map10 is named).
- **Test at the end:** First a size run (rounds and fixed-point counts per instance on the 240 at the milestones of the implied equation [L1] and of the implied-integer conversion [L2]); then the pair of AS-118 if instances end on the limit.
- **State:** designed.

### AS-120 The presolve ends at the first round without a finding, even after a budget stop
- **Assumption:** Rounds repeat until one reports no finding; a recorded implied bound is not a finding. If a call of that round stopped on its budget or stall limit, the presolve still ends, the rows left on the lists are not served, and such rounds are not taken up again after the switch to explicit bounds.
- **Set by:** Design (the engine driver; written into the contract of `presolve()`).
- **Where:** gpu_presolve/src/presolve.cpp:331-363, 411-435; gpu_presolve/include/gprs/presolve.hpp:296-297, 327-331.
- **Switch:** no switch yet: needs one; reachable only through the budgets (AS-121).
- **Alternatives:** Go on while a list is not empty; start the stopped family again with a new budget.
- **May change:** A family that stalled leaves its list unserved; the model keeps what that list held.
- **Test at the end:** Covered by the budget test of AS-121: identical reduced models on the 240 with every budget times 10 close it.
- **State:** in the engine.

### AS-121 Every rule family has a work budget and a stall stop; the values are placeholders
- **Assumption:** A call of a family may spend floor + passes * pass work units and may go floor + stall_passes * pass units without a finding, where pass = free nonzeros + 8 * (live rows + live columns). Defaults {passes, stall passes, floor}: row rules, bound propagation and row tightening {16, 2, 100000}; doubleton equations {16, 4, 100000}; parallel rows {8, 8, 100000}; dual fixing {4, 2, 100000}; for the four column families, which are still empty, simple probing {4, 2, 100000}, singleton columns {8, 4, 100000}, dual columns {8, 4, 100000}, parallel columns {8, 8, 100000}; designed for the five rules: implied integers {4, 2, 100000}, exchange {8, 4, 100000}. The motive: 25.1% of HiGHS's presolve time on the 31 is stage runs that remove nothing.
- **Set by:** The owner for the principle (change 4 of the Stage 0 report, accepted with decision 6, HISTORY Entry 13); code for the values (HISTORY Entry 14: "all budget defaults are placeholders"). The principle entered the plan as part of the block of thirteen changes, not one by one; the values were fixed during the six hours of independent work, without a check-in.
- **Where:** DESIGN.md section 2.1 (lines 317-320) and section 3 (line 495); gpu_presolve/include/gprs/presolve.hpp:80-99, 113-122; stage1c_design/v2_design.txt:101; lowworth_v2_design.txt:70.
- **Switch:** `options_t::budget[rule]` = {passes, stall_passes, floor}; not settable from the tool or the adapter.
- **Alternatives:** Unlimited (run to a fixed point); values tuned per family; a larger stall limit.
- **May change:** A call that stops on its budget leaves reductions for the next round or undone and the fixed-point flag is lost, so the reduced model can be larger; the result stays deterministic. On an instance where most rows hold a slack with a cost the exchange family rescans many rows every round.
- **Test at the end:** First a size run with every budget times 10 on the 240 and the 9 instances above 1e7 nonzeros (remaining nonzeros, presolve seconds, count of calls stopped on a budget). If reduced models differ: standard cluster comparison, default budgets against times 10; the primal integral is the measure that sees the presolve time.
- **State:** in the engine (ten families, four of them still empty); designed (two families).

### AS-122 Work counts one unit per matrix entry and eight per row or column visited
- **Assumption:** Budgets are in work units of the Stage 0 cost model: one unit per entry read or written and `line_visit_work` = 8 units per row or column visited; they never depend on time. The pass that scales a budget counts rows and columns with weight 8 because, with nonzeros alone, a first scan of a model with short rows would stall before every row was seen once.
- **Set by:** Design (the Stage 0 cost model measured a work unit at 10 ns and a row visit at 8 units; the correction of the pass is a finding of the review of the Stage 1 code structure, HISTORY Entry 14).
- **Where:** gpu_presolve/include/gprs/rules.hpp:21-23; gpu_presolve/include/gprs/presolve.hpp:80-84, 92-99; DESIGN.md section 2.1 (lines 322-338).
- **Switch:** no switch yet: needs one (named constant `line_visit_work`).
- **Alternatives:** Other weights; a budget in seconds (not deterministic).
- **May change:** How the budgets of AS-121 cut on models with many short rows.
- **Test at the end:** Covered by the budget test of AS-121.
- **State:** in the engine.

### AS-123 Work ceilings of the column families are reports only until measured
- **Assumption:** Proposed for the four new families together: at most 2 passes of the input model in the median and 6 at the 90% quantile, no instance on the round limit, at most 5 instances losing the fixed-point flag. The review showed that 2 passes is about the cost of the first scan (median 1.70 to 2.03 passes on the 240 input models), so ceilings are stated per family and only reported until the milestone of the singleton-column rules and the forcing column [M1] has measured them. Time and work of the two families of the five rules were not measured either.
- **Set by:** Taken as recommended (HISTORY Entry 40: "work ceilings are reports until measured"; v2_amendments.txt:74-76, :90).
- **Where:** stage1c_design/v2_design.txt:207, :281; v2_amendments.txt:74-76, :90; v2_reviews.txt:91-94; lowworth_v2_design.txt:103, :231.
- **Switch:** no switch yet: needs one if a ceiling is to become a gate (a limit in a gate script); today the figures are only reported.
- **Alternatives:** Hard ceilings as gates.
- **May change:** Presolve time, not the reduced model.
- **Test at the end:** Not a quality test: work in passes per family on the 240 and the 40 at each milestone.
- **State:** measurement only.

### AS-124 The time limit is only an abort; it never changes which reductions are tried
- **Assumption:** A wall-clock limit only ends the presolve with the status "limit" and a valid partial result, and it is tested before every call of a family; budgets and stall stops count work, not time. The default is no limit.
- **Set by:** Design (for determinism, DESIGN.md section 2.5).
- **Where:** DESIGN.md section 2.5 (lines 467-469); gpu_presolve/include/gprs/presolve.hpp:147-149; gpu_presolve/src/presolve.cpp:237-243, 481-483.
- **Switch:** `options_t::time_limit` (default infinite); tool `--time-limit <seconds>`. Inside cuOpt it is the remaining solve time (AS-155).
- **Alternatives:** A time budget per phase (not deterministic); a work limit in place of a time limit; a cap on presolve time as a share of the solver's limit.
- **May change:** With a finite limit the outcome depends on the speed of the machine, and inside cuOpt a presolve that hits the limit hands the solver the unreduced problem (AS-156).
- **Test at the end:** Not a pair: report presolve seconds and the count of "limit" results per arm.
- **State:** in the engine.

### AS-125 The dual column rules run once per round, not inside the cascade
- **Assumption:** Dual bound strengthening and the single-lock alias are called once per round on the listed columns and the columns of listed rows. HiGHS reruns its dual-fixing extensions after every change (301.6 s against 157.3 s with them off on the 240).
- **Set by:** Design (v2_design.txt:72, :120).
- **Where:** stage1c_design/v2_design.txt:72, :120.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Inside the cascade, as HiGHS.
- **May change:** Findings reach the row rules one round later; more rounds against the round limit.
- **Test at the end:** A size run at the milestone of dual bound strengthening and the single-lock alias [M4] (size and rounds against the HiGHS reference with bound strengthening alone); no quality pair unless an instance ends on the round limit.
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4]).

### AS-126 Simple probing applies the aliases of a call as one batch through the alias forest
- **Assumption:** The aliases of all rows of a call are edges of the doubleton family's alias forest. A row is taken only when no two of its free columns are already connected, a row with a cut edge stays (an integer column is never mapped to a continuous root), and the forest picks the root, which need not be the binary.
- **Set by:** Design (repair of a work bound, v2_answers.txt item 24).
- **Where:** stage1c_design/v2_design.txt:27; v2_answers.txt:95-98; v2_reviews.txt:33-37.
- **Switch:** no switch yet: needs one.
- **Alternatives:** One row at a time with the binary as root.
- **May change:** Which column represents a group; rows deferred to the next round.
- **Test at the end:** No quality pair; a work and determinism choice. Alias depth and entries moved are reported at the milestone of simple probing [M2].
- **State:** designed (milestone of simple probing [M2]).

### AS-127 The free singleton rule takes the smallest column index and finishes chains inside one call
- **Assumption:** Of several singleton columns of a row that pass, the smallest column index is substituted; a column that the deletion makes a singleton is visited in the same call; new costs follow the cancellation rule (zero within 1e-12 relative) and must be finite.
- **Set by:** Design (chains: v2_answers.txt item 22).
- **Where:** stage1c_design/v2_design.txt:34; v2_answers.txt:87-90.
- **Switch:** no switch yet: needs one.
- **Alternatives:** The column with the largest pivot; one link of a chain per round.
- **May change:** Which column of a row is removed and how large the growth factor gets (AS-094).
- **Test at the end:** No quality pair; covered by the pair of the free singleton reductions (AS-050).
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]).

### AS-128 Dominance between parallel columns follows fixed type rules and a fixed order of cases
- **Assumption:** An integer absorber needs an integer moved column and an integral ratio; a continuous absorber takes either type; a continuous column is never moved onto an integer. The four cases are tried in a fixed order, one pair at a time, ties to the smaller column index. HiGHS has the same four cases.
- **Set by:** Design (v2_design.txt:62-63, :129).
- **Where:** stage1c_design/v2_design.txt:62-63, :129.
- **Switch:** `options_t::sub_enabled[sub_rule::parallel_dominance]` turns the rule; no switch yet for the order.
- **Alternatives:** Prefer fixing the integer column; another case order.
- **May change:** Which of two parallel columns is fixed and at which bound.
- **Test at the end:** Standard cluster comparison after the milestone of dominance between parallel columns [M3]: `parallel_dominance` on against off.
- **State:** designed (milestone of dominance between parallel columns [M3]).

### AS-129 A parallel pair is merged only when no dominance case applies
- **Assumption:** The parallel-column family tries the dominance fixing first and the merge only for pairs with equal cost that no dominance case takes; in primal-only mode pairs go straight to the merge. In a bucket of singleton columns only neighbours by cost over coefficient are tested for a merge.
- **Set by:** Design (v2_design.txt:83, :121).
- **Where:** stage1c_design/v2_design.txt:83, :121.
- **Switch:** `options_t::sub_enabled[sub_rule::parallel_dominance]` and `[sub_rule::parallel_merge]`; no switch for the order.
- **Alternatives:** Merge first.
- **May change:** With equal costs a column is fixed at a bound instead of merged whenever the other can absorb it.
- **Test at the end:** Standard cluster comparison after the milestone of the merge of parallel columns [M5]: `parallel_dominance` off with `parallel_merge` on, against both on.
- **State:** designed (milestones of dominance between parallel columns [M3] and of the merge of parallel columns [M5]).

### AS-130 The single-lock alias applies all candidates of one binary in one transaction
- **Assumption:** All candidates of one side that pass are aliased to the binary in one transaction; the candidate that is the binary itself is tested afterwards on the new state, and a column that became the target of an alias in a call is not aliased in that call. For two binaries that could each be aliased to the other, exactly one alias is expected.
- **Set by:** Design (a binding amendment after the validity review, v2_amendments.txt:63-64).
- **Where:** stage1c_design/v2_amendments.txt:63-64; v2_reviews.txt:43-46; v2_design.txt:79.
- **Switch:** no switch yet: needs one.
- **Alternatives:** One alias per call.
- **May change:** Which of two mutually aliasable binaries stays.
- **Test at the end:** A unit test; no quality pair.
- **State:** designed (milestone of dual bound strengthening and the single-lock alias [M4]).

### AS-131 The implied equation runs between dual bound strengthening and the single-lock alias
- **Assumption:** Inside the dual column family every candidate column is tested first by dual bound strengthening, then by the implied equation; the single-lock alias collects its rows afterwards. The implied equation therefore takes two-entry lock rows that the single-lock alias would otherwise alias.
- **Set by:** Design (lowworth_v2_design.txt:86, :156; the effect on the alias counts was pointed out by the worth verifier).
- **Where:** stage1c_design/lowworth_v2_design.txt:86, :156, :228; lowworth_v2_reviews.txt:111-112.
- **Switch:** no switch yet: needs one for the order; the designed `implied_equations` off removes the effect.
- **Alternatives:** Single-lock alias first; a family of its own for the implied equation (the first design).
- **May change:** The verifier expects "the same alias with the row gone" as outcome, but the alias counts measured at the milestone of dual bound strengthening and the single-lock alias [M4] move when the implied equation [L1] is built (1,908 integer sources of this kind on bab6, 206 on netdiversion, input models).
- **Test at the end:** No pair of its own; seen in the pair of AS-063 and in the per-instance size table of the milestone of the implied equation [L1].
- **State:** designed (milestone of the implied equation [L1]).

### AS-132 Doubleton equations form a spanning forest; the kept column is chosen by type, then entries
- **Assumption:** The doubleton equations of a batch form a graph; a deterministic spanning forest is chosen and the substitutions are composed on the frozen tree. An edge whose orientation is not admissible is cut and its row stays, and edges outside the tree stay as rows. The kept column of a tree is an integer one if there is any, then the one that makes the most equations usable, then the one with the most entries, then the smallest index; a binary has no priority over a general integer.
- **Set by:** Design (guards from the reviews, DESIGN.md section 4; the root by integrality after the review of HISTORY Entry 14).
- **Where:** DESIGN.md section 4 (lines 532-534); gpu_presolve/src/alias_forest.cpp:139-172; gpu_presolve/include/gprs/postsolve.hpp:168-178.
- **Switch:** no switch yet: needs one for the choice of the root; `options_t::enabled[rule::doubleton]` (tool `--off doubleton`) turns the family.
- **Alternatives:** Sequential substitution in rule order, as HiGHS; keep the column with the fewest entries; prefer a binary, or the column with the smaller range.
- **May change:** Which variables the solver sees and with which ranges; bounds, cost and coefficients of the rows that receive the kept column.
- **Test at the end:** Standard cluster comparison: doubleton equations on against off; with a switch, most-entries against a binary-first rule.
- **State:** in the engine.

## 5. Use inside cuOpt

### AS-133 Inside cuOpt the new presolve is off by default; its mode is no runtime parameter
- **Assumption:** cuOpt's setting `presolve_gprs` has the default 0, so a normal run uses PaPILO only. The setting is not registered as a parameter ("the default here is the switch"), so a run with the new presolve needs a build in which this default is changed. The adapter runs the engine at every size; there is no size threshold.
- **Set by:** Design (the adapter, HISTORY Entry 17: "behind one switch that is off by default").
- **Where:** cpp/include/cuopt/mathematical_optimization/mip/mip_hyper_params.hpp:42-46; cpp/src/mip_heuristics/solve.cu:737-745; cpp/src/presolve/gprs_presolve.hpp:18-25; cpp/src/presolve/third_party_presolve.cpp:1090-1106.
- **Switch:** `mip_hyper_params_t::presolve_gprs` = 0, 1, 2, 3 (default 0); a benchmark arm is a commit that edits this line (the local commit `fdbd9f8a6` sets 2; no commit for 3 exists).
- **Alternatives:** Register it as a runtime parameter; make 3 (later 2) the default at productization; keep PaPILO below 50,000 nonzeros.
- **May change:** Nothing in a default build. In a build with another value the whole reduced model that the solver works on changes.
- **Test at the end:** The master switch of the end test: `presolve_gprs` 0 against 3, both with scaling off, reported for all 240, for the 147 at or above 50,000 nonzeros and for the 93 below.
- **State:** in the adapter.

### AS-134 Mode 3 is the comparison arm: the new presolve, then cuOpt's four own reductions
- **Assumption:** For cluster comparisons the new presolve is followed by the PaPILO framework with its reduction allowlist set to indicator strengthening, GF2, BHW coefficient reduction and single-lock dual aggregation, so that a difference to cuOpt as is comes from PaPILO's rules against ours and not from these four. The comparison is PaPILO with scaling off against mode 3 with scaling off. It is assumed that with the allowlist no PaPILO rule runs. A mode 2 arm mixes two questions, because indicator strengthening, GF2, BHW coefficient reduction and single-lock dual aggregation live inside the PaPILO call and do not run there.
- **Set by:** Design (DESIGN.md amendment A6 and HISTORY Entry 27, after the owner's request of 2026-11-27, Entry 26: "cuopt has other special reductions in presolve. bhw ,bve, indicator, single_lock_dual_aggregation. try to find a way for them in the plan."). The owner did not confirm mode 3 as the comparison arm in words. That mode 2 drops the four was recognised in HISTORY Entry 26, after the owner pointed to GF2: "enlight_hard has a special presolve routine: gf2".
- **Where:** DESIGN.md amendment A6 (lines 66-85); cpp/src/presolve/third_party_presolve.cpp:1122-1133 (names at 1126-1129); cpp/src/presolve/gprs_presolve.hpp:22-24; HISTORY Entry 27 (lines 1634-1637). Mode 2: cpp/src/presolve/third_party_presolve.cpp:1153-1169, :727-728, :759, :918-922; HISTORY Entries 23 and 26.
- **Switch:** `presolve_gprs` = 3; the allowlist names `indicatorstrengthening`, `gf2presolve`, `bhwcoeffreduce`, `single_lock_dual_aggregation` are hard-coded.
- **Alternatives:** Mode 2; mode 1; native ports of the four; cuOpt as shipped (PaPILO, scaling on) as a third arm.
- **May change:** Which reductions the solver's model has had. Not verified: whether the PaPILO framework still runs a built-in clean-up of trivial rows and columns in mode 3 (the pinned PaPILO commit `815a2cc77597d9a07d6d6aadf0ba23fb52ae5117` is not on this machine; another commit does). If it does, part of the 39 of the 240 instances that mode 3 changes against mode 2 may come from PaPILO. Measured without a solver on the 240 (HISTORY Entry 37): enlight_hard is solved only with the four (in the 30 s pre-flight it found no solution in mode 2), neos-787933 goes from 125,652 to 61,944 nonzeros, var-smallemery-m6j6 loses 13%.
- **Test at the end:** Standard cluster comparison: `presolve_gprs` 0 against 3 (the main comparison); 3 against 2 (what the four are worth after ours); 3 against 1 (whether PaPILO's rules still add).
- **State:** in the adapter (27 unit tests pass in modes 1 to 3; no solver run on the cluster exists).

### AS-135 Mode 1, all of PaPILO after the new presolve, has no number on the 240
- **Assumption:** In mode 1 PaPILO with all its rules presolves the model the new presolve left, and two postsolves are chained. It was run only in the unit tests and in the pre-flight (40 models, 30 s); the presolve-only report on the 240 ran modes 0, 2 and 3.
- **Set by:** Design (HISTORY Entry 17; the report's default mode list is the assistant's).
- **Where:** cpp/src/presolve/third_party_presolve.cpp:1170-1189; cpp/tests/mip/gprs_presolve_test.cu:220-229.
- **Switch:** `presolve_gprs` = 1; for the report the environment variable `GPRS_REDUCTIONS_MODES` (default "023").
- **Alternatives:** PaPILO first and the new presolve after it (not built).
- **May change:** The reduced model is PaPILO's result on a pre-reduced input.
- **Test at the end:** Standard cluster comparison: `presolve_gprs` 0 against 1 (whether the new presolve adds anything when PaPILO still runs).
- **State:** in the adapter; measured in the pre-flight only.

### AS-136 cuOpt's four reductions run once after the new presolve, hosted by PaPILO, until ported
- **Assumption:** GF2, BHW coefficient reduction and single-lock dual aggregation are PaPILO methods, and indicator strengthening runs inside the PaPILO call. In mode 3 they run once after the new presolve and never inside its loop, and PaPILO stays in the build as their host, so the goal "no PaPILO inside" is postponed. Ports are planned next to the rule family each resembles: BHW and single-lock dual aggregation with the cheap column rules (BHW after their gate), GF2 and indicator strengthening with sparsify and the aggregator (DESIGN.md:111-112, :147). HISTORY Entry 26 and the table of DESIGN.md amendment A6 still say "Stage 3" for GF2, single-lock and indicator strengthening, in the old stage numbers.
- **Set by:** Design (DESIGN.md amendments A6 and A8), after the owner's request of HISTORY Entry 26.
- **Where:** DESIGN.md amendment A6 (lines 66-82) and amendment A8 (lines 111-112, 147); cpp/src/presolve/third_party_presolve.cpp:727-728, :759, :1126-1129; HISTORY Entry 26 (lines 1595-1604).
- **Switch:** no switch yet: needs one per reduction; today only the hard-coded allowlist names of mode 3 (no hyper-parameter per reduction except for indicator strengthening).
- **Alternatives:** Port now; drop one on evidence (the wording of plan version 4); run them inside the loop.
- **May change:** What the four enable for our rules is not picked up (smaller BHW coefficients for row tightening, GF2 fixings for propagation). GF2 solves enlight_hard in presolve; BHW rewrites coefficients without a size effect.
- **Test at the end:** Standard cluster comparison: mode 3 with each of the names removed from the allowlist in turn, against mode 3 complete; after each port, the ported rule inside the loop against the hosted one after it.
- **State:** in the adapter through the PaPILO framework; ports planned.

### AS-137 cuOpt's single-lock rule keeps running in mode 3 after the engine has its own
- **Assumption:** When the engine gets its single-lock alias (milestone of dual bound strengthening and the single-lock alias [M4]), mode 3 still runs cuOpt's single-lock dual aggregation after it, until a pre-flight shows that it finds nothing more. The same kind of alias is then looked for twice.
- **Set by:** Design (v2_design.txt:248: "outside the engine and only noted").
- **Where:** stage1c_design/v2_design.txt:75, :248; cpp/src/presolve/third_party_presolve.cpp:1126-1129.
- **Switch:** no switch yet: needs one (the name `single_lock_dual_aggregation` is in the hard-coded list of mode 3).
- **Alternatives:** Take the name out of the list once the engine's rule exists.
- **May change:** A difference between two arms may come from either of the two rules.
- **Test at the end:** First a count of what cuOpt's rule still finds after the engine on the 240; then a standard cluster comparison: mode 3 with against without the name in the list. The four arms of AS-051 contain this pair.
- **State:** planned (milestone of dual bound strengthening and the single-lock alias [M4]); the list is in the adapter today.

### AS-138 Two PaPILO methods of the baseline arm have no rule in the plan
- **Assumption:** cuOpt runs PaPILO with eighteen of PaPILO's methods and three of its own (GF2, BHW coefficient reduction, single-lock dual aggregation). Matched by name only, sixteen of the eighteen have a rule that is built or planned here; FixContinuous and SimplifyInequalities have none. The plan's coverage table compares with HiGHS only. What the two methods do in the pinned PaPILO was not read (AS-287), and none of the eighteen was measured apart.
- **Set by:** Code (cuOpt's existing method list); design for the coverage table against HiGHS only (DESIGN.md section 5).
- **Where:** cpp/src/presolve/third_party_presolve.cpp:725-759 (the method list), :925 (dual postsolve is off for a MIP, so all eighteen run); DESIGN.md section 5 (lines 550-589).
- **Switch:** no switch yet: needs one. The list of mode 3 is hard-coded (cpp/src/presolve/third_party_presolve.cpp:1126-1129); `set_reduction_allowlist` exists on the presolver object (cpp/src/presolve/third_party_presolve.hpp:114), is called by unit tests only, and can only narrow that list.
- **Alternatives:** A coverage table against PaPILO's methods as the plan has for HiGHS; port the methods that pay.
- **May change:** Which reductions the mode 0 arm has and the mode 3 arm lacks beyond the planned rules. By their names SimplifyInequalities and FixContinuous change coefficients and bounds, not only sizes.
- **Test at the end:** Standard cluster comparison, one pair per PaPILO method: mode 3 against mode 3 with that one method added to the list (the method names are to be read from the pinned PaPILO; `probing` and `cliquemerging` are the two the code already uses, third_party_presolve.cpp:810 and :815). Each pair gives the worth in solution quality of one PaPILO method, and so an order of what to build that does not rest on nonzeros. Components and dependent rows are the pairs of AS-015.
- **State:** in cuOpt as is; not planned.

### AS-139 Indicator strengthening sees the model after the new presolve instead of before
- **Assumption:** In cuOpt as is, indicator strengthening works on the problem before PaPILO's rules. In modes 1 and 3 it works on the model the new presolve left, so the rows it matches (two-entry rows x - z <= 0 between binaries, with side exactly 0 and coefficients exactly +1 and -1) may no longer be found where our rules removed or changed them: the single-lock alias takes exactly such a row as its lock row, and a strengthened coefficient that is no longer exactly 1 breaks the match. The plan says the count of appended and lifted rows in both orders is measured in the pre-flight; no such count was found in HISTORY.
- **Set by:** Design (DESIGN.md amendment A6, marked open there).
- **Where:** DESIGN.md amendment A6 (lines 83-85, table line 72); cpp/src/presolve/third_party_presolve.cpp:918-922, :1171; cpp/src/presolve/indicator_strengthening.cpp:84-110.
- **Switch:** no switch yet: needs one for the order.
- **Alternatives:** Indicator strengthening on the original model before the new presolve (an adapter change, or the planned port as a first phase that appends rows).
- **May change:** The number of appended and lifted rows, hence LP strength and size on models with indicator structure.
- **Test at the end:** First the count of appended and lifted rows in both orders on the 240. Then a standard cluster comparison: strengthening before the new presolve against after it.
- **State:** in the adapter as "after"; the other order is planned with the port.

### AS-140 Block BVE is taken to be unaffected by replacing PaPILO
- **Assumption:** cuOpt's elimination of blocks of zero-objective binaries runs in cuOpt's own presolve after the third-party presolve, when the probing cache is on, and reads its implications. It is assumed to be unaffected, although the model and the implications it reads differ; when the new probing replaces the probing cache, its implication export must have the form block BVE reads.
- **Set by:** Design (DESIGN.md amendment A6: "Not affected by replacing PaPILO"; read from code, not measured), after the owner's request that names "bve".
- **Where:** DESIGN.md amendment A6 (table line 76); cpp/src/mip_heuristics/diversity/diversity_manager.cu:381-387; cpp/include/cuopt/mathematical_optimization/mip/mip_hyper_params.hpp:34-36.
- **Switch:** `mip_hyper_params_t::presolve_block_bve` (default true; parameter `mip_hyper_block_bve`); it needs `mip_probing` true.
- **Alternatives:** Off; later a wide phase of the new presolve.
- **May change:** Block BVE works on the binaries and implications that the presolve leaves (binaries aliased or fixed by our rules, rows merged), so it finds other blocks in the two arms.
- **Test at the end:** First a log count of block-BVE eliminations in modes 0 and 3 on the 240. Standard cluster comparison: `presolve_block_bve` true against false in mode 3, next to the same pair in mode 0; again when the new probing replaces the probing cache.
- **State:** in cuOpt as is; the export is planned with the probing stage.

### AS-141 cuOpt's probing cache stays, with a work budget fitted on PaPILO-reduced models
- **Assumption:** cuOpt's probing cache runs after the third-party step in every mode. Its work limit is a fixed scale (`probing_work_scale` = 1.5e8) divided by a size measure of the reduced model, and that scale was chosen from runs on 240 instances after PaPILO; a larger reduced model gets a smaller limit.
- **Set by:** Code (existing cuOpt constants, kept).
- **Where:** cpp/src/presolve/presolve_budget_policy.hpp:22-43, 66-70, 136-137; cpp/src/mip_heuristics/diversity/diversity_manager.cu:343-377.
- **Switch:** `mip_probing` (default true) for on and off; the scale is a compile-time constant.
- **Alternatives:** Refit the scale on models reduced by the new presolve; replace the probing cache (planned).
- **May change:** How much probing the solver does after the presolve: with 0.7862 of the nonzeros left in mode 3 against 0.6905 for PaPILO (237 of the 240, HISTORY Entry 37), probing is cut earlier in the new arm.
- **Test at the end:** Standard cluster comparison: `mip_probing` true against false in mode 3; and the probing work each arm used, read from the logs, next to the main comparison.
- **State:** in cuOpt as is; replacement planned.

### AS-142 cuOpt's own presolve runs after every mode and repeats part of a switched-off rule
- **Assumption:** After the third-party step cuOpt runs, in every mode, its own bound propagation on the GPU, which writes the tightened bounds into the problem; the probing cache, which also finds singletons and substitutions and changes the problem; a trivial presolve; and block BVE. An arm in which an engine rule is switched off therefore still gets part of that rule's effect from cuOpt, and how much is not known.
- **Set by:** Code (the existing cuOpt pipeline; named in the first code survey of HISTORY, never as a condition of the tests).
- **Where:** cpp/src/mip_heuristics/diversity/diversity_manager.cu:320-410 (bound update :334-340, probing cache :345-374, trivial presolve :379, block BVE :381-387, the log line "After cuOpt presolve" :407); cpp/src/mip_heuristics/solver.cu:254-262 (not run when the presolver is set to none); HISTORY.md line 27.
- **Switch:** `mip_probing` (default true) for the probing cache and `mip_hyper_block_bve` for block BVE; no switch yet for the bound update and the trivial presolve: needs one.
- **Alternatives:** Skip cuOpt's bound update when the engine ended at a fixed point; keep it as a safety net.
- **May change:** What an off-against-on pair can show. For bound propagation (AS-043), explicit bounds of continuous columns (AS-047), the improvement step (AS-077) and the step limit (AS-079) the off arm is partly repaired by cuOpt, so an equal result says nothing about the rule.
- **Test at the end:** In every pair report per arm the number of bounds that cuOpt's bound update changes (a counter to be added) and the sizes of the log line "After cuOpt presolve" against the sizes the third-party step returned. With a switch for the bound update: standard cluster comparison, bound update on against off, and the pairs of AS-043 and AS-047 repeated with it off in both arms.
- **State:** in cuOpt as is.

### AS-143 cuOpt looks for symmetry on the reduced model, so the presolve decides what is left
- **Assumption:** cuOpt detects symmetry after all presolve steps, on the final reduced problem, and uses it in branch and bound; detection is on by default. The engine's choices change what it finds: a merge of parallel columns removes interchangeable columns, the settings `never` and `range_kept` leave them to the symmetry handling, and dual fixing, alias choices and ties by the smaller index (AS-289) can break a symmetry or keep it. No design weighs a rule against this; the plan lists symmetry only as a reduction it leaves out.
- **Set by:** Code (existing cuOpt behaviour; not noted in DESIGN.md or HISTORY).
- **Where:** cpp/src/mip_heuristics/solver.cu:322-334; cpp/src/mip_heuristics/solve.cu:50-55; cpp/include/cuopt/mathematical_optimization/mip/solver_settings.hpp:126; cpp/src/math_optimization/solver_settings.cu:226; stage1c_design/v2_design.txt:277; DESIGN.md section 5 (lines 588-589), the only mention of symmetry in the notes.
- **Switch:** cuOpt parameter `mip_symmetry` (default -1; 0 switches detection off; values up to 2).
- **Alternatives:** Report the symmetry cuOpt finds per arm; keep interchangeable columns apart on purpose where cuOpt's symmetry handling is on.
- **May change:** How much symmetry cuOpt finds in each arm, and with it the tree on symmetric models. It bears on the default of `merge_integers` (AS-001) and on the merge of continuous pairs (AS-058).
- **Test at the end:** The three arms of AS-001 run twice, with `mip_symmetry` at its default and at 0; within each symmetry setting the arms differ in one switch. For the main comparison, the number of instances on which cuOpt reports symmetry per arm (which log line gives it was not looked up). Read on node count, final bound and optimal count.
- **State:** in cuOpt as is.

### AS-144 Probing and cliques enter cuOpt first as replacements, with PaPILO unchanged
- **Assumption:** When the new probing and clique pipeline exist, they first replace cuOpt's probing cache and clique table while PaPILO stays as it is, which isolates their effect; from then on the probing cache is off in the arm with the new code.
- **Set by:** Design (DESIGN.md section 10).
- **Where:** DESIGN.md section 10 (lines 728-729) and section 13 (line 798).
- **Switch:** no switch yet: needs one.
- **Alternatives:** Run the new probing next to the probing cache.
- **May change:** The probing cache also feeds block BVE and the heuristics; switching it off moves those to the new by-products.
- **Test at the end:** The plan's second cluster run: cuOpt as is against cuOpt with the new probing and cliques, PaPILO unchanged.
- **State:** planned.

### AS-145 PaPILO's limits come from the original model even when it runs on the reduced one
- **Assumption:** The cap on PaPILO's probing batch is derived from the problem as the user handed it in, before the new presolve. In mode 1 PaPILO then probes the smaller model with a cap chosen for the larger one; in mode 3 probing is not in the list and the cap has no effect.
- **Set by:** Code.
- **Where:** cpp/src/mip_heuristics/solve.cu:710-711, 755-756; cpp/src/presolve/presolve_budget_policy.hpp:62-64, 116-123.
- **Switch:** `mip_hyper_params_t::presolve_papilo_probing_max_badgesize` and `presolve_max_rounds` (both default -1).
- **Alternatives:** Compute the features from the model the engine left.
- **May change:** PaPILO's probing effort in mode 1 only.
- **Test at the end:** Matters only if mode 1 becomes an arm: mode 1 with the cap from the original against from the reduced model.
- **State:** in cuOpt and the adapter.

### AS-146 Early heuristics run as long as the presolve lasts; four threads stay reserved for PaPILO
- **Assumption:** cuOpt starts heuristics on the original model when the presolve starts and stops them when it ends; a presolve that is many times faster shortens that window. The thread plan reserves up to four threads for PaPILO in every mode, although the engine uses one.
- **Set by:** Code (existing cuOpt design; not noted in DESIGN.md or HISTORY).
- **Where:** cpp/src/mip_heuristics/solve.cu:557-559, 613-652, 805-828; cpp/src/mip_heuristics/lns/thread_budget.hpp:21-44; cpp/src/mip_heuristics/mip_constants.hpp:43.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Give the free threads to the early heuristics when PaPILO does not run; keep the heuristics running into cuOpt's own presolve.
- **May change:** The first incumbents and so the primal integral, independent of the quality of the reduced model: a confound of every mode 0 against mode 3 comparison (AS-220).
- **Test at the end:** Standard cluster comparison: mode 3 against mode 3 with the early-heuristic window held to PaPILO's duration; or read the primal gap at the end apart from the primal integral.
- **State:** in cuOpt as is.

### AS-147 The adapter keeps a second model copy; the plan's memory rule is not in code
- **Assumption:** The adapter keeps the original model in the engine's format and the postsolve stack until the solve ends. The plan's rule that cuOpt's early GPU heuristics, which hold about 120 bytes per nonzero during presolve, start only if they and the presolve both fit is not in the code: the early GPU heuristic starts without such a test.
- **Set by:** Code for the copy; design for the rule (DESIGN.md section 10; the 120 bytes were read from source).
- **Where:** cpp/src/presolve/gprs_presolve.cpp:210-217, :235; DESIGN.md section 10 (line 732); cpp/src/mip_heuristics/solve.cu:697-704.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Keep only what the feasibility check and the postsolve need; add the memory test; presolve first and heuristics after.
- **May change:** Host memory on the largest instances (2 of the 240 are above 1e7 nonzeros); with the rule, the early heuristics may not start on large models, which delays the first incumbent.
- **Test at the end:** A memory measurement on the instances above 1e7 nonzeros. When the device engine is in cuOpt: standard cluster comparison, early heuristics allowed against not allowed during presolve.
- **State:** in the adapter (copy); the memory rule is planned.

### AS-148 cuOpt's scaling is off in all development and in every benchmark arm
- **Assumption:** cuOpt's own scaling is switched off in every run of the project, standalone and on the cluster, in the baseline arm as well; whether scaling comes back, and where, is decided at the end. cuOpt's shipped default has scaling on, so the baseline "PaPILO with scaling off" is not the product. DESIGN.md amendment A3 still announces one measurement of cuOpt as is with scaling on against off; the owner ruled it out in Entry 27, so that sentence is stale. The adapter's unit tests use default settings and so run the engine on scaled models, which no benchmark does. The plan says the standalone harness keeps a switch that applies cuOpt's scaling to reproduce a cluster case; no such switch was found in the baseline tool or the runners.
- **Set by:** The owner, until the end (HISTORY Entry 26, 2026-11-27: "let's keep scaling off for all develoment and decide if we want to add scaling and debug scaling issues at the end"; Entry 27: "scaling doesn't matter much and we want to eliminate the effects of possible bad scaling on our presolve. at first we should compare reductions, the cuopt quality tuning comes later. ... Do not also use scaling in the benchmark."). It replaces the owner's decision 4 of 2026-11-26 (Entry 13) to run with scaling on and off.
- **Where:** DESIGN.md amendment A3 (lines 41-46), replacing section 0 decision 4 (line 191) and section 10 (lines 719-726); cpp/include/cuopt/mathematical_optimization/mip/solver_settings.hpp:160; cpp/include/cuopt/mathematical_optimization/constants.h:76, 295-297; cpp/src/mip_heuristics/solve.cu:687-689.
- **Switch:** cuOpt setting `mip_scaling` = 0 off, 1 on, 2 on without the objective (the code default is 2); a benchmark arm needs a commit that sets the default to 0 (the local commits `384524b34` and `c0b04112c` do).
- **Alternatives:** cuOpt's default; scaling with the objective; scaling after the presolve (AS-149).
- **May change:** The presolve sees unscaled coefficients, so its absolute tolerance tests fall differently than on a scaled model; the solver's LP and heuristics run on unscaled reduced models in both arms. A gain measured with scaling off need not hold with scaling on. The belief behind the choice, that scaling does not matter much, is the owner's and is not measured: the assistant had proposed one measurement of what scaling is worth for cuOpt as is, the owner removed that arm, so the size of the postponed question is unknown. If scaling matters, every quality number measured with scaling off is off by an unknown amount against the product, in both arms and not necessarily by the same amount.
- **Test at the end:** Standard cluster comparison: mode 3 with `mip_scaling` 0 against 2, and mode 0 with 0 against 2, so that it is seen whether the new presolve changes what scaling is worth.
- **State:** in force as a convention.

### AS-149 cuOpt scales before its presolve; "scale after the presolve" is the expected question
- **Assumption:** When scaling is on, cuOpt scales the problem first and the presolve receives the scaled rows. The assistant expects that the question at the end is not only on or off but scaling after the presolve; no code scales after the presolve today.
- **Set by:** Code for the order; design for the expectation (DESIGN.md amendment A3; the owner did not comment on it beyond "decide ... at the end").
- **Where:** cpp/src/mip_heuristics/solve.cu:687-689 (scaling) before :707-746 (presolve); DESIGN.md amendment A3 (lines 44-46); HISTORY Entry 26 (lines 1584-1587).
- **Switch:** no switch yet: needs one for the position; `mip_scaling` for on and off.
- **Alternatives:** Scale the reduced model after the presolve; scale inside the engine; no scaling.
- **May change:** With scaling before the presolve the engine's absolute tolerances (feasibility 1e-6, zero 1e-9) act on scaled rows and the reductions can differ; with scaling after it the solver gets a scaled reduced model.
- **Test at the end:** Three arms in mode 3: scaling off; scaling before the presolve (`mip_scaling` 2, today's order); scaling after the presolve (to be built).
- **State:** in cuOpt as is; the other position is planned as a question for the end.

### AS-150 cuOpt's sub-MIPs keep PaPILO
- **Assumption:** The new presolve replaces PaPILO only for the main problem. The sub-MIPs that cuOpt solves inside branch and bound are presolved by a separate PaPILO call with the full method list, one thread and a time limit of the smaller of a tenth of the sub-MIP's time and 60 s; that call never reads the mode.
- **Set by:** The owner, for now (HISTORY Entry 45, 2026-11-27: "Yes submips can use papilo for now."); the plan leaves the decision to productization.
- **Where:** DESIGN.md amendment A10 (line 179) and section 10 (line 727); cpp/src/branch_and_bound/branch_and_bound.cpp:2486-2489; cpp/src/presolve/third_party_presolve.hpp:269; cpp/src/presolve/third_party_presolve.cpp:1314-1402.
- **Switch:** no switch yet: needs one (`presolve_gprs` acts on the main MIP only).
- **Alternatives:** The new presolve for sub-MIPs (needs an adapter for the simplex problem type); no presolve for sub-MIPs.
- **May change:** The sub-MIP heuristics work on PaPILO-reduced models in both arms, so they are held equal in a mode 0 against mode 3 comparison; PaPILO stays a build dependency.
- **Test at the end:** Standard cluster comparison: sub-MIPs with PaPILO against sub-MIPs with the new presolve, main presolve the same in both.
- **State:** in cuOpt as is; decided for now.

### AS-151 The new presolve runs only on the main MIP, and only when presolve is on
- **Assumption:** The adapter is skipped for LPs and when presolve is off (`presolve` = 0); those paths keep PaPILO, PSLP or no presolve. For a MIP the values -1, 1 and 2 of `presolve` all end at PaPILO, because the solver replaces PSLP by PaPILO before the presolve, so the new presolve runs with each of them whenever `presolve_gprs` is not 0.
- **Set by:** Design (comment in the header: "gprs runs on MIPs presolved with Papilo only").
- **Where:** cpp/src/presolve/third_party_presolve.cpp:1090-1102; cpp/src/presolve/third_party_presolve.hpp:122-124; cpp/src/mip_heuristics/solve.cu:440-446, :516, :707.
- **Switch:** `presolve_gprs` together with the parameter `presolve` (-1 default, 0 off, 1 PaPILO, 2 PSLP); `presolve` = 0 switches both presolves off.
- **Alternatives:** Use the engine for LPs (needs a dual postsolve) and for sub-MIPs.
- **May change:** Nothing on the benchmark set, where every instance is a MIP with the default presolver.
- **Test at the end:** None over MIP quality; it bounds where a result applies.
- **State:** in the adapter.

### AS-152 The adapter passes two options; every other engine setting keeps its engine default
- **Assumption:** Only primal-only mode and the time limit are passed to the engine. Budgets, round limits, rebuild fractions, the parallel-row settings, family and sub-rule switches, the merge of integer columns and the limits of the column rules cannot be set from cuOpt, and the design expects the standalone defaults inside cuOpt. No switch of this register other than those two can be turned in a cluster run without a change of the adapter or a benchmark commit that edits a default in `presolve.hpp`.
- **Set by:** Code.
- **Where:** cpp/src/presolve/gprs_presolve.cpp:242-244; cpp/src/presolve/third_party_presolve.cpp:1106; stage1c_design/v2_design.txt:277. By HISTORY Entry 46 cuOpt builds against the engine with the groundwork and the adapter's 27 tests pass; the adapter still passes only `primal_only` and `time_limit`.
- **Switch:** no switch yet: needs one (a pass-through of engine options by hyper-parameters or environment variables).
- **Alternatives:** A pass-through; one benchmark commit per arm that flips one engine default.
- **May change:** Nothing in the model; it decides which end tests can be run at all.
- **Test at the end:** Not a test: a precondition of every cluster pair that names an engine option.
- **State:** in the adapter.

### AS-153 cuOpt's tolerances are not passed to the engine
- **Assumption:** The solver hands its absolute and relative tolerance to the presolve call, but the engine works with its own fixed tolerances (AS-069), although the plan says tolerances are inputs. The engine's defaults equal cuOpt's defaults (absolute 1e-6, integrality 1e-5); cuOpt also has a relative tolerance of 1e-12, and PaPILO in the same place uses a feasibility tolerance of 1e-5. A user who changes cuOpt's tolerances does not change the engine's.
- **Set by:** Code.
- **Where:** cpp/src/mip_heuristics/solve.cu:751-752; cpp/src/presolve/third_party_presolve.cpp:1106, :779; cpp/src/presolve/gprs_presolve.cpp:242-245; cpp/include/cuopt/mathematical_optimization/mip/solver_settings.hpp:102-106; DESIGN.md section 6 (lines 593-594).
- **Switch:** no switch yet: needs one in the adapter (the engine side is `options_t::tol`).
- **Alternatives:** Pass cuOpt's `absolute_tolerance`, `relative_tolerance` and `integrality_tolerance`; use 1e-5 as PaPILO does.
- **May change:** Which rows count as redundant, forcing or infeasible, and how far a postsolved solution may violate the original model.
- **Test at the end:** Standard cluster comparison: engine feasibility tolerance at its default against cuOpt's tolerances passed through, with the count of "not feasible after post solve" warnings next to the quality measures.
- **State:** in the adapter.

### AS-154 Inside cuOpt the engine runs primal-only only when a set-solution callback is registered
- **Assumption:** The engine runs with its dual reductions on unless the user registered a callback that injects solutions; then it runs primal-only and PaPILO gets `dualreds = 0`. Every benchmark arm and the presolve-only report therefore measure the mode with dual reductions, and no default-mode run exercises a primal-only choice.
- **Set by:** Code (cuOpt's existing rule for PaPILO, taken over by the adapter).
- **Where:** cpp/src/mip_heuristics/solve.cu:518-522, 734-736; cpp/src/presolve/third_party_presolve.cpp:1106, :936; cpp/tests/mip/gprs_presolve_test.cu:248.
- **Switch:** no switch yet: needs one for the user (`set_dual_reductions(bool)` on the presolver object follows the callback).
- **Alternatives:** Always primal-only, so that every solution found on the original model can be carried into the reduced model; dual reductions always on.
- **May change:** In primal-only mode fewer reductions; in the default mode, solutions of the early heuristics and user starts can fail to carry over (AS-169).
- **Test at the end:** Standard cluster comparison: mode 3 with dual reductions against mode 3 forced primal-only. The callback path itself is not in the benchmark and needs a functional test.
- **State:** in the adapter.

### AS-155 The presolve's time limit inside cuOpt is the whole remaining solve time
- **Assumption:** The engine gets the remaining time of the solve as its limit, and no limit at all in cuOpt's deterministic mode; there is no budget of its own for the presolve other than the work budgets. PaPILO afterwards gets that limit minus the seconds the adapter used.
- **Set by:** Code (cuOpt's existing rule for PaPILO, taken over).
- **Where:** cpp/src/mip_heuristics/solve.cu:714-716; cpp/src/presolve/third_party_presolve.cpp:1106, :1116.
- **Switch:** the solver's `time_limit` and `mip_determinism_mode`; no presolve-specific switch yet.
- **Alternatives:** A cap as a share of the time limit; a work limit instead of seconds.
- **May change:** Nothing while the engine is fast (18 s summed over the 240 standalone against 94 s for the call inside cuOpt, HISTORY Entry 37, not timed in the same run; 16.7 s standalone with the engine now on the branch, Entry 46). It matters when later stages (probing, enumeration) make the engine slow; and because the limit is wall-clock, the same instance can fall on either side of it in two runs.
- **Test at the end:** Not a pair today: check in each arm's logs that no instance has the engine status "limit".
- **State:** in the adapter.

### AS-156 A refused model, or a presolve that hits its limit, gets the full PaPILO presolve
- **Assumption:** When the adapter refuses a model (a quadratic objective or quadratic constraints, semi-continuous variables, variable bounds not set for every variable, a variable named twice in a row, a value the engine does not accept, a single-precision build) or the engine ends on its time limit, the original problem is kept and PaPILO presolves it with all its rules, in every mode, modes 2 and 3 included. The engine's valid partial result on "limit" is thrown away, and PaPILO then gets the time that is left, which is zero or less. The run logs one line; no result table counts such instances. The plan had said that a failure is an assertion in benchmark commits and that a logged fallback exists only in the product default.
- **Set by:** Code (the question was put to the owner as open in HISTORY Entry 22; the owner's answer of Entry 45 covers sub-MIPs only).
- **Where:** cpp/src/presolve/gprs_presolve.cpp:55-58, 80-82, 101-103, :118, 121-124 (the refusals), 183-185, 236-240, :276; cpp/src/presolve/third_party_presolve.cpp:1116-1150; DESIGN.md section 8 (lines 701-703) and section 0 (lines 213-214).
- **Switch:** no switch yet: needs one.
- **Alternatives:** Abort in benchmark builds; solve the original problem without a presolve; use the partial result as a reduced model; no PaPILO in the build.
- **May change:** A mode 2 or mode 3 arm can contain instances that were presolved by PaPILO and are then counted for the wrong arm. Expected zero on the 240 (the standalone engine took all 240: 205 reduced and 35 unchanged, recounted for the engine tags v4pre and s1c_m0; 204 and 36 with the earlier engine of HISTORY Entry 23), not verified inside cuOpt as a count; what PaPILO does with a time limit of zero or less was not checked.
- **Test at the end:** Not a pair: in every arm count the log lines "GPRS presolve skipped" and "the original problem is kept" and report those instances apart. A later pair: partial result on the limit against fallback to PaPILO.
- **State:** in the adapter.

### AS-157 The adapter knows two variable types: integer and continuous
- **Assumption:** The adapter marks a column as integer when cuOpt's type is integer (codes 'I' and 'B') and as continuous otherwise; a model with semi-continuous variables is refused. Binary is not a type of its own, and the reduced model comes back with 'I' and 'C' only.
- **Set by:** Design (the adapter).
- **Where:** cpp/src/presolve/gprs_presolve.cpp:115-120, 153-171; cpp/include/cuopt/mathematical_optimization/optimization_problem_utils.hpp:30-35; cpp/src/mip_heuristics/solve.cu:467-477.
- **Switch:** no switch yet: needs one only with a third column type (AS-165); the two types are fixed in the adapter.
- **Alternatives:** A third type for implied integers (AS-165).
- **May change:** Nothing against PaPILO's path, which uses the same two types plus its implied-integer flag (AS-164).
- **Test at the end:** None on its own; covered by the unit tests in three modes.
- **State:** in the adapter.

### AS-158 A maximisation is negated, objective scale and offset are carried, names are dropped
- **Assumption:** The engine minimises; for a maximisation the adapter negates costs and offset on the way in and back on the way out. The reduced problem keeps the sense, the objective scaling factor and the problem name, and has no row or variable names and no row types.
- **Set by:** Design (following what the PaPILO path returns).
- **Where:** cpp/src/presolve/gprs_presolve.cpp:69-70, 110-114, 128-143; cpp/src/presolve/gprs_presolve.hpp:46-47.
- **Switch:** no switch yet: needs one only if names are to be kept in a written reduced model; the conversion is fixed in the adapter.
- **Alternatives:** None considered.
- **May change:** Nothing in the solver; a written reduced model has no names.
- **Test at the end:** None over quality; covered by the unit test that runs minimise and maximise in modes 1, 2 and 3.
- **State:** in the adapter.

### AS-159 The adapter drops stored zeros, sorts rows, refuses duplicate entries and needs double precision
- **Assumption:** The adapter leaves out stored zeros, sorts a row by column when it is not sorted, and refuses a model in which a row names a variable twice. A single-precision build never runs the engine.
- **Set by:** Design (the adapter).
- **Where:** cpp/src/presolve/gprs_presolve.cpp:31-48, :55, 92-105; cpp/src/mip_heuristics/solve.cu:708 (`sort_csr(op_problem)` before the call).
- **Switch:** no switch yet: needs one only if duplicate entries are to be summed instead of refused; the conversion is fixed in the adapter.
- **Alternatives:** Sum duplicate entries instead of refusing.
- **May change:** The nonzero count the engine logs as its input excludes stored zeros, so it can differ from the count PaPILO's line reports for the same instance; the engine's result depends on the order of the entries of a row (AS-281).
- **Test at the end:** None over quality.
- **State:** in the adapter.

### AS-160 The engine's answer "infeasible" ends the solve without a second check
- **Assumption:** When the engine reports the model infeasible, cuOpt returns Infeasible at once, also when an early heuristic already holds a feasible solution. PSLP's answer is distrusted on badly scaled data; the engine's is not.
- **Set by:** Code (cuOpt's existing rule for PaPILO, taken over).
- **Where:** cpp/src/presolve/third_party_presolve.cpp:1107-1110, 1243-1254; cpp/src/mip_heuristics/solve.cu:758-762.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Fall back to PaPILO or to the solver when an incumbent exists or the data are extreme.
- **May change:** A wrong answer becomes the user's result. Known and open: on generated models with extreme constants the base engine returned "infeasible" for a feasible model (AS-258).
- **Test at the end:** A status test, not a quality measure: the final status of every instance in the new arm against the known status of the 240 (233 with a best known objective, 7 infeasible).
- **State:** in the adapter.

### AS-161 The status "unbounded or infeasible" is passed on as it is
- **Assumption:** The engine has no status "unbounded"; when it reports "unbounded or infeasible" cuOpt returns UnboundedOrInfeasible without trying to tell the two apart, as it does with PaPILO's answer.
- **Set by:** Design (DESIGN.md section 8 lists the statuses).
- **Where:** cpp/src/presolve/third_party_presolve.cpp:1111-1114; cpp/src/mip_heuristics/solve.cu:763-767; DESIGN.md section 8 (lines 701-703).
- **Switch:** no switch yet: needs one if the original problem is to be solved without presolve to tell the two cases apart; today the status is passed on as it is.
- **Alternatives:** Solve the original problem without presolve to decide.
- **May change:** The reported status on such instances.
- **Test at the end:** A status test on the status instances of the testbed (neos-3230511-yuna); not a quality measure.
- **State:** in the adapter.

### AS-162 A model reduced to nothing counts as solved by the presolve
- **Assumption:** When the engine removes every row and column, the adapter reports the problem as solved in presolve in every mode, and neither PaPILO nor cuOpt's four reductions run. The optimum is then the engine's alone, without a solver in between.
- **Set by:** Design (the adapter).
- **Where:** cpp/src/presolve/third_party_presolve.cpp:1153-1164; cpp/src/presolve/gprs_presolve.cpp:135-151.
- **Switch:** no switch yet: needs one only if an emptied model is to be confirmed by a solve; `presolve_gprs` decides whether the engine runs at all.
- **Alternatives:** None considered.
- **May change:** The result on such instances rests on the engine's postsolve alone.
- **Test at the end:** Status and objective against the best known value on every instance that ends this way; not a quality pair.
- **State:** in the adapter.

### AS-163 An unchanged model goes on to the third-party step only in modes 1 and 3
- **Assumption:** When the engine finds nothing, modes 1 and 3 hand the original problem to PaPILO (mode 3 with the four own reductions only), and mode 2 passes a copy of the problem through with no reduction at all.
- **Set by:** Design (the adapter).
- **Where:** cpp/src/presolve/third_party_presolve.cpp:1117-1119, 1122-1133, 1139-1150, 1165-1168.
- **Switch:** `presolve_gprs`.
- **Alternatives:** In mode 2 fall back to PaPILO when nothing was found.
- **May change:** The standalone engine leaves 35 of the 240 unchanged (recounted for the engine tag s1c_m0; 36 with the earlier engine of HISTORY Entry 23); in mode 2 those reach the solver unpresolved except for cuOpt's own later presolve.
- **Test at the end:** Report the unchanged instances apart in the mode 0 against mode 3 comparison.
- **State:** in the adapter.

### AS-164 No implied-integer flags reach the solver in modes 2 and 3
- **Assumption:** In cuOpt as is, PaPILO marks continuous columns as implied integers and cuOpt uses the marks to decide whether the objective is integral, which lets branch and bound round its bound and tighten its cutoff. In mode 2 the adapter returns an empty list, and in mode 3 PaPILO's implied-integer method is not in the allowlist. No place in DESIGN.md or HISTORY decides or notes this.
- **Set by:** Code.
- **Where:** cpp/src/presolve/third_party_presolve.cpp:1152-1169, 1298-1304, :745; cpp/src/mip_heuristics/solve.cu:780; cpp/src/mip_heuristics/problem/problem.cu:1266-1296; cpp/src/branch_and_bound/branch_and_bound.cpp:506, :520, :541.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Add PaPILO's implied-integer method to the mode 3 list (its name was not verified for the pinned PaPILO commit); pass the engine's own flag once the conversion rule exists.
- **May change:** Whether cuOpt treats the objective as integral: node bounds rounded up, cutoff one unit below the incumbent. That acts on the optimal count and the gap, not on the nonzero count. The engine's own reductions can cost the same property when they move cost onto other columns (AS-267).
- **Test at the end:** Standard cluster comparison: mode 3 as is against mode 3 with the implied-integer detection added to the list.
- **State:** in the adapter (a blind spot).

### AS-165 A converted column reaches cuOpt as a true integer; there is no third column type
- **Assumption:** After the implied-integer conversion the column is integer in the state and in the reduced model. A flag per reduced column that was continuous in the input is designed, as "information for the adapter, not a licence to drop integrality"; postsolve rounds the value of such a column. The plan's interface names a separate implied-integer flag and the type order integer, implied integer, continuous; the adapter has no place for the flag.
- **Set by:** Design (decision 5 of lowworth_v2_design.txt:242, recommended "no third type": "Say if cuOpt should get such columns as continuous; that is then a rule of its own, not a flag."; no owner answer in HISTORY up to Entry 46).
- **Where:** stage1c_design/lowworth_v2_design.txt:32-33, :242; DESIGN.md:521 and section 9 (lines 705-711); cpp/src/presolve/gprs_presolve.cpp:157.
- **Switch:** no switch yet: needs one (the flag `result_t::implied_integer` is designed; nothing in the adapter reads it).
- **Alternatives:** Hand flagged columns to cuOpt as continuous (safe only while the row that proves integrality is still in the reduced model and no later reduction used the integrality); hand them as integers that the solver does not branch on.
- **May change:** cuOpt sees more integer variables: more branching candidates, rounding and local-search heuristics treat them as integers, integrality is checked on them. The LP relaxation is unchanged except for rounded bounds.
- **Test at the end:** Standard cluster comparison when the conversion exists: converted columns handed over as integers against as continuous with the flag (needs adapter code); and the pair of AS-065.
- **State:** designed (milestone of the implied-integer conversion [L2]); the adapter part is not designed in detail.

### AS-166 A scaled column keeps its index with a new meaning; cuOpt's "transformed" marker is missing
- **Assumption:** After a column scaling the column index stands for z with x = scale * z; later records and every by-product exported in reduced indices are in the new units, and postsolve multiplies the value back. The plan's interface has a "transformed" marker in the column map for this; the adapter has only index maps. How cuOpt's own scaling interacts is not stated in the design.
- **Set by:** Design (lowworth_v2_design.txt:40, :233, following DESIGN.md section 9).
- **Where:** stage1c_design/lowworth_v2_design.txt:40, :233; DESIGN.md section 9 (line 708); cpp/src/presolve/gprs_presolve.cpp:252-257.
- **Switch:** no switch yet: needs one; the designed `implied_integer_scaling` off avoids the case.
- **Alternatives:** A new column instead of a reused index; no scaling.
- **May change:** Anything in cuOpt that maps reduced columns to original columns one to one (warm starts, incumbents crushed into the reduced model, exported bounds or implications) must use the scale.
- **Test at the end:** First a correctness test of the adapter on the instances that scale (postsolved solutions feasible on the original model); then the pair of AS-066.
- **State:** designed in the engine (milestone of the implied-integer conversion [L2]); the adapter part is not built.

### AS-167 The by-products of the plan's interface are not exported
- **Assumption:** The plan's interface gives cuOpt by-products (fixings, implications, clique rows, variable bounds) in reduced indices. The adapter passes none of them, so cuOpt's probing cache and clique table have to find again what the presolve knew.
- **Set by:** Design (DESIGN.md section 9; the missing part is code).
- **Where:** DESIGN.md section 9 (lines 705-711) and section 12 (lines 757-758); cpp/src/presolve/gprs_presolve.cpp:153-171.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Export implications and cliques to cuOpt's probing cache and clique table.
- **May change:** A strengthening without a size effect that cuOpt does not get today.
- **Test at the end:** With the probing and clique stage: standard cluster comparison, export off against on.
- **State:** planned (probing and clique stage).

### AS-168 A postsolved solution that violates the original model gives a warning only
- **Assumption:** After the undo cuOpt recomputes feasibility on the original problem; when it fails, the log gets a warning and the solution is still returned as feasible. A wrong postsolve of the new presolve would therefore show only in the log.
- **Set by:** Code (existing cuOpt behaviour).
- **Where:** cpp/src/mip_heuristics/solve.cu:902-919, :928; DESIGN.md section 12 (line 750) and section 13 (line 797: "every postsolved solution feasible").
- **Switch:** no switch yet: needs one (a setting that returns such a solution as infeasible or falls back to the early incumbent).
- **Alternatives:** Return the solution as infeasible, or fall back to the early-heuristic incumbent.
- **May change:** A quality number can contain objective values of solutions that are not feasible in the original model.
- **Test at the end:** In both arms count the lines "The solution is not feasible after post solve" and "The solution violates the variable bounds after post solve"; an arm with more of them is not comparable on the primal gap.
- **State:** in cuOpt as is.

### AS-169 Solutions found on the original model are carried over by crush and may be lost
- **Assumption:** User start solutions and the solutions of the early heuristics are carried into the reduced model by the engine's crush, then PaPILO's, then a clamp to the reduced bounds that was written for PaPILO's dual fixing. With dual reductions on, the engine guarantees a feasible crushed point of equal objective only in primal-only mode, so a good early solution can fail to carry over; cuOpt keeps it in original space as a fallback for the end.
- **Set by:** Code (existing cuOpt design, extended to the engine by the adapter).
- **Where:** cpp/src/presolve/third_party_presolve.cpp:1610-1654; cpp/src/presolve/gprs_presolve.hpp:48-49; cpp/src/mip_heuristics/solve.cu:542-547, 932-955; cpp/src/mip_heuristics/diversity/diversity_manager.cu:217-222.
- **Switch:** no switch yet: needs one (see AS-154).
- **Alternatives:** Primal-only mode for the engine in every run; a repair step after the crush; the forward replay designed for the column rules (AS-299).
- **May change:** Whether early incumbents enter the population and branch and bound after the presolve, so the early part of the primal integral.
- **Test at the end:** The pair of AS-154 (dual reductions against forced primal-only), with the count of crushed solutions that are infeasible in the reduced model.
- **State:** in the adapter.

### AS-170 Injected solutions are checked against the adapter's own copy of the original model
- **Assumption:** With the new presolve active, a solution that a callback injects is checked on the engine's copy of the original model: bounds and integrality within cuOpt's integrality tolerance, rows within the absolute tolerance plus the relative tolerance times the largest finite side, with compensated sums.
- **Set by:** Design (mirroring the rule of the PaPILO path).
- **Where:** cpp/src/presolve/gprs_presolve.cpp:191-201, 322-355; cpp/src/presolve/third_party_presolve.cpp:1557-1608; cpp/src/mip_heuristics/diversity/population.cu:364-378.
- **Switch:** cuOpt's tolerances `absolute_tolerance`, `relative_tolerance`, `integrality_tolerance`.
- **Alternatives:** None considered.
- **May change:** Which callback solutions are accepted; not reached in the benchmark, which has no callbacks.
- **Test at the end:** A functional test with a set-solution callback; not a quality pair.
- **State:** in the adapter.

## 6. Measurement conventions and references

### AS-171 Until the end every gate measures size against HiGHS presolve only
- **Assumption:** Every stage gate measures the reduction in nonzeros (with rows and columns next to it) against HiGHS presolve alone, without a solve. Whether the reduced model is a stronger formulation for cuOpt is measured only at the end, over solution quality; on the way a smaller model is taken as a usable stand-in for a stronger one. What "quality of reductions" means was left to the assistant ("Find the balance yourself", HISTORY Entry 1).
- **Set by:** The owner (HISTORY Entry 27, 2026-11-27: "at first we should compare reductions, the cuopt quality tuning comes later"; Entry 45: "first goal is the nnz reduction", "nnz reduction is not the main goal is to strengthen the formulation ... we will test all of them at the end over the solution quality").
- **Where:** DESIGN.md amendment A10 (lines 159-180); HISTORY Entries 1, 27 and 45; stage1c_design/lowworth_v2_design.txt:234 ("the milestones measure sizes only"). From the audit: HISTORY Entry 17 (item 4); DESIGN.md section 12; gpu_presolve/analysis/bench240_stage1c.py:77; gpu_presolve/check/gprs_check.cpp:906-908.
- **Switch:** no switch yet: needs one (a strength measure in the gate scripts); the end comparison is switched by `presolve_gprs`.
- **Alternatives:** A cluster run after each stage (the plan had cluster runs 1 to 3); a strength measure in the gates that needs no cluster: the LP relaxation value of the reduced model against the original and against the HiGHS-presolved model, counts of tightened bounds and strengthened coefficients, a model diff by original row and column.
- **May change:** Rules are ranked, ordered and tuned by size. A rule that shrinks the model but weakens the LP relaxation or the heuristics passes every gate; a rule that strengthens without shrinking gets no credit before the end. Coefficients, bounds, integrality and the LP bound of the reduced model are not compared with the reference, although the audit found equal sizes with different models: on rd-rplusc-21 the first engine kept about 1e7 coefficients that HiGHS strengthens, and its output was no fixed point of its own rules on 5 of 15 instances. The check writes the LP values of the original and the reduced model per instance, but no script collects them.
- **Test at the end:** The frame of the end test, not a pair. Next to the main comparison, per instance, the nonzero ratio against the change in primal gap and primal integral, to see whether size predicted quality. The root relaxation objective and the final bound of every arm are read from the logs next to it (AS-217).
- **State:** measurement only.

### AS-172 A gain is claimed only from 50,000 nonzeros up; results are reported per size class
- **Assumption:** The goal is stated for MIPs from 50,000 nonzeros up; instances above 1e7 nonzeros are stress tests. Below 50,000 nonzeros (93 of the 240) no gain is claimed and nothing is gated. Results are given per size class (below 5e4, to 5e5, to 1e7, above for the 240; sub-bands A1 to D for the testbed), and the time rule is a gate only in classes A and B (5e4 to 1e7 nonzeros).
- **Set by:** The owner (HISTORY Entry 2, 2026-11-26: "It should already perform better with more than 50,000 non-zeros."); the class boundaries are design.
- **Where:** DESIGN.md section 1 (lines 237-238) and section 10 (lines 715-718); gpu_presolve/testbed/make_benchmark_testbed.py:104-112; gpu_presolve/analysis/stage1_time_gate.py:88; the counts 147 and 93 were recomputed from gpu_presolve/testbed/benchmark240.json.
- **Switch:** no switch of its own: a reporting boundary, held in the band field of the testbed files (gpu_presolve/testbed/make_benchmark_testbed.py:104-112).
- **Alternatives:** Gate every size; other boundaries; keep PaPILO below a threshold.
- **May change:** Design choices are made for the upper class; on the 93 small instances the new presolve may be weaker than PaPILO, and that is hidden if only the upper class is read.
- **Test at the end:** Every cluster comparison reported for all 240, for the 147 and for the 93.
- **State:** measurement only.

### AS-173 Presolve is measured on the dev node, solver quality only on the cluster
- **Assumption:** Sizes, times and correctness are measured on the dev node without a cuOpt solve (the 40, the 9 above 1e7 nonzeros in the background with a time limit, and the 240). No solver quality test runs on the dev node. The cluster runs the regular cuOpt benchmark and may be used without asking when a larger test is needed; a standalone cluster job is set up only for a wide set (the 147 at or above 50,000 nonzeros plus the testbed), never for a handful.
- **Set by:** The owner (HISTORY Entry 2: "Do not do solver quality tests locally. We have benchmark clusters for large-scale tests."; Entry 7: "Either run those 9 instances locally or increase the number of instances on the cluster."; Entry 13: "if you need to run a benchmark on the cluster to do larger scale test, do it.").
- **Where:** DESIGN.md section 0 (lines 194, 201) and section 12 (lines 747-753).
- **Switch:** no switch yet: needs one only if short solver runs on the dev node are wanted; today it is a rule of working, not a setting.
- **Alternatives:** Short solver runs on the dev node for a first look at quality (the LP bound of the reduced model, short solves against HiGHS-presolved models).
- **May change:** No reduction has been seen by the solver at scale until the first cluster run; the effect of every rule on the solve is unknown until then.
- **Test at the end:** This is the place of the end test, not a pair.
- **State:** measurement only.

### AS-174 Decisions rest on the 240 benchmark instances, and every figure names its instance set
- **Assumption:** A decision about scope or order is supported by a number on the 240 MIPLIB 2017 benchmark instances as cuOpt runs them (233 with a best known value and 7 listed as infeasible; 93 below 5e4 nonzeros, 93 up to 5e5, 52 up to 1e7, 2 above), and every figure says which set it was measured on. Order decisions need both the 240 and the 40, because the two sets rank the same rules differently (AS-239).
- **Set by:** The owner (HISTORY Entry 36, 2026-11-27: "wait a sec. this is on testbed instances. i want to see the impact on all instance set"); the rule for order decisions is the assistant's lesson of Entry 41.
- **Where:** gpu_presolve/testbed/make_benchmark_testbed.py:104-158; gpu_presolve/testbed/benchmark240.json; HISTORY Entries 23, 36 and 41.
- **Switch:** runner option `--testbed gpu_presolve/testbed/benchmark240.json --set all`.
- **Alternatives:** Only the 147 at or above 50,000 nonzeros; the 40 alone; the whole MIPLIB collection.
- **May change:** Which rules look worth building, their order, and the gate values. Every threshold is then chosen on the set the end test runs on (AS-274).
- **Test at the end:** The cluster run uses the same 240; not a pair.
- **State:** measurement only.

### AS-175 The testbed is 40 instances drawn by a feature cover of the MIPLIB 2017 collection
- **Assumption:** The 40 were drawn by a greedy cover of 85 structure features with a quota per size band and one instance per model family, benchmark instances with a known optimum preferred. 29 of the 40 are in the 240; 9 are above 1e7 nonzeros. A mean over 40 is carried by a few instances: four hold 73% of the nonzeros that full HiGHS removes on the 31.
- **Set by:** The owner for diversity (HISTORY Entry 2: "I wouldn't just do the largest 10 in MIPLIB, but also consider the differences in the problem to keep the diversity"); the cover rule and the 40 names are design, not reviewed by the owner.
- **Where:** DESIGN.md section 14 (lines 861-919); gpu_presolve/testbed/testbed.json (list `main`, 40 entries).
- **Switch:** runner option `--testbed gpu_presolve/testbed/testbed.json` with `--set dev`, `large`, `status` or `all`.
- **Alternatives:** Another cover rule; more instances per band; the 240 alone.
- **May change:** The testbed gate value of every stage and the ranking of rules.
- **Test at the end:** Not a quality test by itself: 11 of the 40 are not in the cluster benchmark set, so quality on them would need an extra cluster arm on the testbed.
- **State:** measurement only.

### AS-176 Evidence above 1e7 nonzeros rests on nine testbed instances
- **Assumption:** The large end is measured on the nine testbed instances above 1e7 nonzeros, on the dev node; the 240 hold only two such instances (square41, square47). On the 40, two of the nine (neos-4332810-sesia and dlr2) carry the mean against HiGHS with all cheap rules: 0.8008 in nonzeros over 40 and 0.9357 over the other 38.
- **Set by:** The owner for where they run (HISTORY Entry 7); the choice of the nine is design.
- **Where:** gpu_presolve/testbed/testbed.json (`where` = dev-large); gpu_presolve/results/stage1c/gate/today_testbed_cheap_all.txt; HISTORY Entry 41.
- **Switch:** runner option `--set large`; `--without neos-4332810-sesia,dlr2` in gpu_presolve/analysis/bench240_stage1c.py.
- **Alternatives:** Leave the largest band out of the mean (its time is already reported only); add more large instances.
- **May change:** Whether a stage passes its testbed gate, and the build order (the singleton column rules were confirmed first because of these two instances).
- **Test at the end:** Not a quality test: these instances are not in the cluster set, so only size and time can be compared unless a cluster arm on the testbed is added.
- **State:** measurement only.

### AS-177 Status instances have their own rule, and the rule behind neos-3230511-yuna is unknown
- **Assumption:** Four testbed instances test statuses only: bnatt500, fhnw-binpack4-4 and neos-2075418-temuka (infeasible) and neos-3230511-yuna (unbounded; HiGHS reports "unbounded or infeasible"). They have no size gate. On temuka, where HiGHS returns the model unreduced after 15.7 s, detecting infeasibility counts as better and returning a model within 18.8 s as equal. The rule by which HiGHS detects yuna is unknown; if it belongs to a later stage, the Stage 1 check needs an exception. The 240 hold seven instances listed as infeasible.
- **Set by:** Taken as recommended (proposed in section 7 of the Stage 0 report and "entered as the report proposes them", DESIGN.md section 0, open item 4).
- **Where:** DESIGN.md section 12 (line 788) and section 14 (lines 921-922); gpu_presolve/testbed/testbed.json (list `status`); gpu_presolve/tools/run_engine.py:222-228.
- **Switch:** runner option `--set status`.
- **Alternatives:** Require the engine to prove infeasibility where a full presolve does; find yuna's rule with HiGHS's switches; leave status instances out.
- **May change:** Nothing in the reduced model; whether a missed detection counts as a failure. A presolve that misses an infeasibility hands the solver a model it must prove infeasible itself.
- **Test at the end:** Not a quality test: the status cuOpt returns on these instances in each arm (count of wrong statuses).
- **State:** measurement only.

### AS-178 Seventeen local model dumps stand in for the node
- **Assumption:** Work on the Mac was verified on 17 model dumps (15 testbed instances with a solution and the infeasible bnatt500 and fhnw-binpack4-4), the largest with 3.0 million nonzeros: identity checks, wrong-engine tests, timing previews. Since HISTORY Entry 45 local dumps of the 240 exist too, and the groundwork engine gives the stored hashes on 240 of 240 there; the node has since confirmed the groundwork on the 240, the 31 and the 9 (HISTORY Entry 46).
- **Set by:** Design (implicit: these are the dumps that fit and existed on the Mac; no decision is recorded).
- **Where:** gpu_presolve/local_dumps (17 files); HISTORY Entries 17, 20, 24, 31, 44 and 45.
- **Switch:** the directory passed to the runners with `--dumps`.
- **Alternatives:** Every verification on the node on the 240 and the 40.
- **May change:** A defect or a slowdown that shows only above 3 million nonzeros or on other structures is not seen before the node run.
- **Test at the end:** Not a quality test: every local result is repeated on the node on the 240 and the 40 (gpu_presolve/tools/stage1c_engine_runs.sh).
- **State:** measurement only.

### AS-179 The standalone engine reads each model as HiGHS's MPS reader built it
- **Assumption:** Every standalone number of the engine (sizes, seconds, hashes, the checks on real instances) is made on a binary dump that the baseline tool writes after HiGHS has read the MPS file: column indices sorted inside each row, a maximisation written as the equivalent minimisation, column types continuous or integer only, no names. Inside cuOpt the engine gets the model from cuOpt's own reader through the adapter. The two inputs are taken to be the same model.
- **Set by:** Design (the Stage 0 tools; no decision is recorded).
- **Where:** gpu_presolve/baseline/README.md:124-143; gpu_presolve/baseline/baseline_tool.cpp:510-545; gpu_presolve/tools/gprs_presolve.cpp:8-11; gpu_presolve/testbed/make_benchmark_testbed.py:5-8; HISTORY Entry 22 (lines 1329-1331), Entry 30 (lines 1817-1819), Entry 37 (lines 2124-2126).
- **Switch:** no switch yet: needs one (a dump of the model the adapter builds; gpu_presolve/cuopt_runs/reduce_to_mps.cpp writes the reduced model only).
- **Alternatives:** Dump the model the adapter hands to the engine and run gates and checks on that; read the MPS files with cuOpt's reader in the standalone tools.
- **May change:** Whatever HiGHS's reader changes on the way in is in the dumps and not in what cuOpt hands over, and the reverse; which entries or bounds either reader alters was not read for this register. Inside cuOpt the sizes equal the standalone ones on 238 of the 240 (AS-281); bounds, coefficients and hashes were not compared. Published solution points are judged on the model HiGHS reads as well (hawaiiv10-130, AS-276).
- **Test at the end:** Not a quality test: on the 240, the hash of the adapter's input model against the hash of the dump, and the list of instances where they differ.
- **State:** measurement only.

### AS-180 The diagnostic patch was checked in the full setting only on 204 of the 240
- **Assumption:** Checks of the split dual-fixing patch in the full setting were made on the 204 of the 240 whose full HiGHS presolve took at most 10 s in the stored node run; 36 instances, among them rd-rplusc-21, square41, square47 and var-smallemery-m6j6, are left out.
- **Set by:** Design (the agent that split the dual-fixing switch, HISTORY Entry 42).
- **Where:** gpu_presolve/baseline/HIGHS_DIAG.md:1044-1052, 1218-1219.
- **Switch:** no switch yet: needs one (the 204 names are a list inside that check; a run on all 240 needs the node).
- **Alternatives:** Run the full setting on all 240 on the node.
- **May change:** Nothing in the engine; a difference between the old and the new patch on the 36 slow instances in the full setting would not have been seen.
- **Test at the end:** Not a quality test: repeat the comparison of the two patch builds on all 240 on the node.
- **State:** measurement only.

### AS-181 HiGHS is the only standalone baseline
- **Assumption:** Reductions and presolve time are judged standalone against HiGHS alone. PaPILO, which the new presolve is to replace, is never run standalone; it is measured only through cuOpt (the presolve-only report and the cluster), where it runs with cuOpt's four reductions and with 4 threads against 1 thread for HiGHS.
- **Set by:** The owner (HISTORY Entry 1: "The goal is to be better than highs in terms of problem reductions and quality of reductions."; Entry 2: "If testing both PaPILO and HiGHS is too cumbersome, we can only test on HiGHS to reduce the total effort.").
- **Where:** DESIGN.md section 0 (line 205); gpu_presolve/analysis/bench240_reductions.py:7-15; HISTORY Entry 27.
- **Switch:** no switch yet: needs one for a second baseline (a tool for PaPILO or SCIP like the HiGHS baseline tool); the HiGHS variant is chosen by the runner option `--variant` and the environment variable `HIGHS_DIAG_OFF`.
- **Alternatives:** PaPILO or SCIP as a second standalone baseline.
- **May change:** The target is HiGHS's rule set and counts; a reduction where PaPILO is stronger than HiGHS stays unseen until a run inside cuOpt.
- **Test at the end:** The baseline arm of the end test is cuOpt with PaPILO, so PaPILO is compared there on solution quality.
- **State:** measurement only.

### AS-182 The reference is HiGHS 1.16.0 at a pinned commit: 64-bit integers, one thread, diagnostic patch
- **Assumption:** Every HiGHS number comes from HiGHS 1.16.0 at commit 283bfc859, built Release with 64-bit integers and run with one thread, patched with the project's diagnostic patch (rule switches and counters). The 64-bit build is needed because HiGHS's clique limit of 10 times the nonzeros overflows 32 bits on neos-4535459-waipa; by the audit it makes the reference slower than a default build would be.
- **Set by:** Design (DESIGN.md section 0: "Baseline: HiGHS only, pinned commit, 64-bit integers"); the one thread is code.
- **Where:** DESIGN.md header (lines 20-23), section 0 (line 205), section 14 (lines 933-935); gpu_presolve/baseline/build_highs_diag.sh:17, 52-53; gpu_presolve/baseline/baseline_tool.cpp:644; gpu_presolve/baseline/HIGHS_DIAG.md.
- **Switch:** `COMMIT` and the cmake options in build_highs_diag.sh (`-DHIGHSINT64=ON`); `threads` is fixed at 1 in the tool.
- **Alternatives:** A newer HiGHS; the 32-bit build where it can run; several threads.
- **May change:** The reference sizes (another version has other rules and limits) and the time allowance; parity is defined against one version of one solver.
- **Test at the end:** Not a quality test; a timing pass against a stock 32-bit build on the instances it can run would size the effect on the time rule.
- **State:** measurement only.

### AS-183 A reference is one presolve-only call of HiGHS with a time limit
- **Assumption:** The reference is one call of HiGHS's presolve with a time limit (1800 s by default; 600 s for the cheap-rule references on the 240; 3600 s above 1e7 nonzeros; full HiGHS on the 240 ended its last instance, ns1760995, at a 30-minute limit), with no solve. HiGHS tests its limit only every 1024 reductions and can overrun. Presolve calls of restarts and of sub-MIPs are not part of any reference.
- **Set by:** Design (the baseline tool; the limits are script defaults).
- **Where:** gpu_presolve/baseline/baseline_tool.cpp:639-647, 701-719; gpu_presolve/baseline/run_baseline.py:48; gpu_presolve/baseline/stage1c_reference_runs.sh:128, 136-147; HISTORY Entry 27 (line 1678).
- **Switch:** `--time-limit` of baseline_tool and run_baseline.py.
- **Alternatives:** The highs binary with a written presolved model; a full solve with restarts.
- **May change:** Nothing in the engine.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-184 The diagnostic patch is taken not to change HiGHS when no switch is set
- **Assumption:** With every switch unset the patched HiGHS is taken to give the stock presolved model (byte-identical on 32 instances; `diag3`, the build of the patch with the split dual-fixing switch, gives the sizes of the build before it, `diag2`, on all 240 with all cheap rules on). The cost of its counters was not measured for the current patch (stock and patched times were within 0.95 to 1.05 on the 31 for its first version). HIGHS_DIAG.md still says the current patch was not built on the node, while HISTORY Entry 42 says it was built there as `diag3`.
- **Set by:** Design (DESIGN.md section 0: "a small patch that adds rule switches and per-rule counters; harness only").
- **Where:** gpu_presolve/baseline/HIGHS_DIAG.md:3-7, 56-57, 631-635, 649-660, 1079-1088; HISTORY Entry 42 (lines 2320-2325).
- **Switch:** environment variables `HIGHS_DIAG_OFF=<names>`, `HIGHS_DIAG_PROBING_CONTINGENT`, `HIGHS_DIAG_PROBING_NOSTOP`, `HIGHS_DIAG_PROBE_ROWS`, `HIGHS_DIAG_REDUNDANCY_EXPLICIT`; all unset gives stock.
- **Alternatives:** Time against the stock build wherever no switch is needed.
- **May change:** The reference time by the counters' cost.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-185 The Stage 1 reference is HiGHS restricted to our rules, with its lock rule on
- **Assumption:** The Stage 1 gate compares with the patched HiGHS in which everything outside our Stage 1 rule set is switched off: the ten expensive stages, parallel columns, simple probing, the dual-fixing substitution, implied equations, implied-integer conversion and six column rules (rule bits 5, 7, 8, 18, 19, 23). A second run with HiGHS's whole dual fixing off (bit 17 as well) brackets the lock rule; the gate uses the first run. Rules without a switch stay in the reference (the binary row test, implied bounds written to integer column bounds, integer row scaling, coefficient strengthening, gcd substitution), so it is not exactly our rule set.
- **Set by:** The owner for the closer reference (HISTORY Entry 13: "1: Option A"); taken as recommended for the choice of the bracket run (a reading listed in DESIGN.md section 0, open item 4, for the owner to overrule).
- **Where:** DESIGN.md section 0 (lines 188, 220-221) and section 12 (lines 772-774); gpu_presolve/baseline/stage1_reference_runs.sh:19-26, 79-84; gpu_presolve/analysis/stage1_compare.py:38; STAGE0_REPORT.md lines 559-563.
- **Switch:** `HIGHS_DIAG_OFF=probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns,parallel_columns,simple_probing,dual_fixing_substitution,implied_equations,implied_integer_conversion` with `--presolve-rule-off 9175456`, a bit mask of HiGHS rule numbers (the sum of 2^5, 2^7, 2^8, 2^18, 2^19 and 2^23: dominated column, forcing column, free column substitution, zero-cost singleton, column stuffing and weakly dominated column off); the bracket run uses 9306528, which adds 2^17, HiGHS's whole dual fixing.
- **Alternatives:** The second bracket run as the gate (without the lock rule the reference leaves 1.427 instead of 1.297 times full HiGHS's nonzeros on the 31); full HiGHS as the reference; more switches.
- **May change:** What parity at Stage 1 means (ratio 1.002 on the 240, 1.0028 on the 40).
- **Test at the end:** Not a quality test.
- **State:** measurement only; done.

### AS-186 The reference of the cheap column rules is HiGHS with ten expensive stages off
- **Assumption:** The reference for Stage 1c is HiGHS with probing, enumeration, clique substitutions, clique merging, the implied-integer sweep, the aggregator, sparsify, Fourier-Motzkin, inequality strengthening and pairwise dominated columns off and every other rule on ("all cheap rules"), presolve only, on unscaled models. The whole parallel rows and columns stage and all rules inside row and column presolve count as cheap. The split is the assistant's.
- **Set by:** Design for the split (HISTORY Entry 28); the owner agreed to the order built on it ("Yes, cheap rules first", Entry 29).
- **Where:** DESIGN.md amendment A8 (lines 104-106); gpu_presolve/baseline/stage1c_reference_runs.sh:5-8, :62, :128, 136-140; stage1c_design/v2_design.txt:195; gpu_presolve/baseline/HIGHS_DIAG.md:1054-1056.
- **Switch:** `HIGHS_DIAG_OFF=probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns` with no rule bit off.
- **Alternatives:** Move sparsify or the aggregator (0.4% and 0.1% of HiGHS's time on the 240) into the cheap set; move parallel columns out; full HiGHS as the reference already now.
- **May change:** The gate target of the stage (0.753 remaining nonzeros for all cheap rules against 0.676 for full HiGHS on 237 of the 240) and what the stage has to build.
- **Test at the end:** Not a quality test. Its counterpart: standard cluster comparison, the four column families off against on.
- **State:** measurement only; references run on the 240 and the 40.

### AS-187 The first three milestones are accepted against a HiGHS ladder that has no forcing column
- **Assumption:** What each group of cheap rules is worth was measured by a ladder of HiGHS runs that add one group at a time. The milestones of the singleton-column rules, simple probing and dominance between parallel columns [M1 to M3] are accepted against the ladder with rule bits 5, 7, 19 and 23 off. HiGHS reaches the forcing column only inside its weakly dominated column test, so this ladder has no forcing column: the engine may be ahead where that rule fires (neos-2746589-doon, neos-3656078-kumeu, neos-3402454-bohle), and every "left-out rules off together" figure also contains the loss of the forcing column, which is among the nine required rules of the engine and is not built yet. The ladder with only bit 19 off is reported next to it.
- **Set by:** Design (a binding amendment after the worth review, v2_amendments.txt:34-42; HISTORY Entries 36 and 40).
- **Where:** stage1c_design/v2_amendments.txt:34-42; v2_reviews.txt:106-109; gpu_presolve/baseline/stage1c_reference_runs.sh:16-27, 62-72; gpu_presolve/results/bench240/stage1c_refs/.
- **Switch:** `--presolve-rule-off 8913056` for the acceptance ladder (a bit mask: the sum of 2^5, 2^7, 2^19 and 2^23, which switches off the dominated column, the forcing column, column stuffing and the weakly dominated column) against `--presolve-rule-off 524288` for the second ladder (2^19: column stuffing only), with the switch sets `M1_OFF`, `M2_OFF` and `M3_OFF` of gpu_presolve/baseline/stage1c_reference_runs.sh.
- **Alternatives:** The second ladder as the acceptance reference; worth measured with the engine's own switches once the rules exist.
- **May change:** Nothing in the model; which reference number a milestone is held against, and what counts as a defect of a milestone.
- **Test at the end:** Not a quality test.
- **State:** measurement only; the runs exist.

### AS-188 The worth of a rule is measured by switching it off in HiGHS, in nonzeros
- **Assumption:** All worth figures are losses of HiGHS (presolve only, unscaled, all cheap rules on) with one rule or a set of rules switched off, ranked by the geometric mean of remaining nonzeros. A one-rule-off run says what HiGHS loses, not what the engine's form of the rule gains, and rules stand in for each other. Known caveats of the switches: HiGHS's conversion switch removes the scaling together with the conversion; its implied-equation switch also disables its integrality proof for substituting a converted column; rule bit 5 guards the whole dominated test; the split of worth between dual bound strengthening and the single-lock alias and between dominance and merge was by counters until the build of the patch with the split dual-fixing switch (`diag3`).
- **Set by:** Design (a convention of the two rule designs and of HISTORY Entries 29 and 36).
- **Where:** stage1c_design/v2_design.txt:211, :261, :273; lowworth_v2_design.txt:19, :28, :35, :42, :63, :161-171; gpu_presolve/analysis/bench240_cheap_ablation.py; gpu_presolve/baseline/stage1c_reference_runs.sh:36-69, 158-167.
- **Switch:** on the HiGHS side: the rule bits of `--presolve-rule-off` (5 dominated column, 7 forcing column, 8 free column substitution, 17 dual fixing, 18 zero-cost singleton, 19 column stuffing, 23 weakly dominated column) and the names `parallel_columns`, `simple_probing`, `dual_fixing_substitution`, `implied_equations`, `implied_integer_conversion`, `dual_fixing_single_equation`, `implied_row_dual_bounds` in `HIGHS_DIAG_OFF`.
- **Alternatives:** Rank by rows and columns (printed), or by solution quality; measure in the engine with its own switches once the rules exist.
- **May change:** The build order and which rules were called low-worth; a rule with no worth in nonzeros (implied-integer conversion, for example) may still change branching and cuts.
- **Test at the end:** Replaced at each milestone by the engine's own on-against-off size runs, and at the end by the cluster pairs per rule.
- **State:** measurement only.

### AS-189 A rule's predicted gain is the smaller of the engine's gap and HiGHS's loss
- **Assumption:** For each instance the gain the engine can get from a rule is predicted as the smaller of the engine's present gap to HiGHS and HiGHS's loss without the rule; these are summed as mean logs over 237 of the 240. The 40 have a reference but no prediction.
- **Set by:** Design (lowworth_v2_design.txt:28, :175; v2_answers.txt item 29).
- **Where:** stage1c_design/lowworth_v2_design.txt:28, :175; v2_design.txt:195; v2_amendments.txt:42-44; v2_answers.txt:115-118.
- **Switch:** no switch yet: needs one only if the prediction is to be made by another rule; it is a formula of the two designs (lowworth_v2_design.txt:28, :175), in no script that was read.
- **Alternatives:** No prediction; a run of the engine with the rule; predictions for both sets.
- **May change:** The prediction is an upper bound per rule only if the engine's form reaches what HiGHS's reaches, and it ignores cases where the engine could gain more than HiGHS.
- **Test at the end:** Not a quality test: checked by each milestone's size run on the 240 and the 40.
- **State:** measurement only.

### AS-190 The reductions measure is a geometric mean of the ratio of remaining nonzeros
- **Assumption:** A presolve is judged by the geometric mean over the instances of the nonzeros the reference leaves divided by the nonzeros the engine leaves (below 1 the engine leaves more). The ratio is of remaining, not of removed, nonzeros. The arithmetic mean and the median are printed next to it.
- **Set by:** The owner (HISTORY Entry 13: "3. do the recommendations", on the proposal of section 7 of the Stage 0 report); reading it as remaining nonzeros is taken as recommended (DESIGN.md section 0, open item 4).
- **Where:** DESIGN.md section 0 decision 3 (line 190); gpu_presolve/analysis/stage1_compare.py:11-14, 314-323; gpu_presolve/analysis/bench240_stage1c.py:34-43, 1104-1108.
- **Switch:** no switch yet: needs one for another mean; the geometric mean is fixed in gpu_presolve/analysis/stage1_compare.py and bench240_stage1c.py, which print the arithmetic mean and the median next to it.
- **Alternatives:** Ratio of removed nonzeros; arithmetic mean; median; sum over instances; a shifted geometric mean.
- **May change:** Which instances carry the figure: a geometric mean of a ratio is carried by the instances that a presolve reduces to almost nothing, and over 40 instances a mean of 0.99 can hide one instance at 67% (Stage 0 report, section 7).
- **Test at the end:** Not a quality test by itself: after the end test, check on the 240 whether the per-instance size ratio between two arms goes with the change in primal gap and primal integral.
- **State:** measurement only.

### AS-191 Means over the 240 leave out ex9, ex10 and enlight_hard
- **Assumption:** The headline geometric mean on the 240 is taken over 237 instances: ex9, ex10 and enlight_hard are left out because a full presolve reduces them to almost nothing and they would carry the mean (HiGHS full is 0.612 with them and 0.676 without, HISTORY Entry 28). The mean over all 240 is printed on a second line, and solved instances are counted apart. The Stage 1c gate script leaves the same three out.
- **Set by:** Taken as recommended (HISTORY Entry 28: "Reported from now on without them"; no owner statement).
- **Where:** DESIGN.md amendment A8 (lines 94-96); gpu_presolve/analysis/bench240_stage1c.py:78, 613-638; gpu_presolve/analysis/bench240_cheap_ablation.py:29.
- **Switch:** the constant `LEFT_OUT = ("ex9", "ex10", "enlight_hard")` in the two scripts; `--without NAME[,NAME]` adds a further line.
- **Alternatives:** Keep all 240; cap each ratio; report the median.
- **May change:** The reported distance to HiGHS. The three are the instances that special reductions decide (enumeration on ex9 and ex10, GF2 on enlight_hard); a gate without them cannot see whether the engine reaches them.
- **Test at the end:** Not a switch. In the end test the three stay in the set and their outcome is reported apart per arm.
- **State:** measurement only.

### AS-192 A size of zero counts as one; instances without a model are handled two ways
- **Assumption:** In every ratio a size of zero is counted as one. An instance that a presolver solves or proves infeasible counts as one remaining nonzero in bench240_reductions.py but is left out of the comparison in bench240_stage1c.py; in stage1_compare.py an engine status without a reduced model fails the gate.
- **Set by:** Code.
- **Where:** gpu_presolve/analysis/bench240_reductions.py:17-19, 125-127; gpu_presolve/analysis/bench240_stage1c.py:28-33, 483-505, 1105-1108; gpu_presolve/analysis/stage1_compare.py:203, :323.
- **Switch:** no switch yet: needs one (one rule for all three scripts; today bench240_reductions.py, bench240_stage1c.py and stage1_compare.py each have their own).
- **Alternatives:** One rule in all scripts; count a solved instance as a full win with a cap.
- **May change:** The mean when an instance is reduced to empty or found infeasible by one side only.
- **Test at the end:** Not a quality test: recompute the stored comparisons with one rule and see whether any gate value moves.
- **State:** measurement only.

### AS-193 An instance is left out of a comparison when one side has no result
- **Assumption:** An instance is left out when a side hit its time limit, was not run, or the two sides read input models of different size. For full HiGHS on the 240 this concerns ns1760995, which reaches the limit; it is left out of the stage shares and of the one-rule-off ranking, but bench240_reductions.py keeps it with the sizes reached at the limit. On the testbed hawaiiv10-130 for full HiGHS is in the same state.
- **Set by:** Code.
- **Where:** gpu_presolve/analysis/bench240_stage1c.py:508-533; bench240_highs_stages.py:46-51; bench240_cheap_ablation.py:51-52; bench240_reductions.py:64-86.
- **Switch:** `--time-limit` of run_baseline.py.
- **Alternatives:** A longer limit; count a limit as the original size.
- **May change:** The means by one instance.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-194 The reductions gate is 0.99 in the mean, below 0.95 explained, a lower bound only
- **Assumption:** A stage passes on size when the geometric mean of reference over engine is at least 0.99 and every instance below 0.95 is explained. An instance where the engine removes much more is listed, never failed; only the correctness checks decide whether such a result is right. The scripts enforce less: stage1_compare.py fails only on the mean and lists instances below 0.95, and bench240_stage1c.py prints the numbers and always returns 0, so "explained" is a judgement of the assistant made by reading.
- **Set by:** The owner (the two numbers: HISTORY Entry 13, "3. do the recommendations"; the lower bound: Entry 22, 2026-11-27, "our final aim is to remove more than highs, so no it shouldn't fail the gate.").
- **Where:** DESIGN.md amendment A1 (lines 32-34), section 0 decision 3 (line 190), section 12 (line 785); gpu_presolve/analysis/stage1_compare.py:314-349; gpu_presolve/analysis/bench240_stage1c.py:79, 1075-1120.
- **Switch:** the constant 0.99 in gpu_presolve/analysis/stage1_compare.py:323 and `BELOW = 0.95` in gpu_presolve/analysis/bench240_stage1c.py:79; no option.
- **Alternatives:** Parity (1.0); a gate per size class; a hard failure for every instance below 0.95; a two-sided gate that fails a large surplus until it is explained.
- **May change:** When a stage is called done. A wrong reduction that removes too much is not flagged by the gate; only the correctness checks stand against it.
- **Test at the end:** Not a quality test: the gate is a proxy. The end test shows whether a version that passes it is also not worse in solution quality; in every arm wrong statuses and infeasible postsolved solutions must be zero, and the instances with a ratio above 1.05 are looked at one by one.
- **State:** measurement only.

### AS-195 The cheap-rules gate has the same two numbers in nonzeros, rows and columns
- **Assumption:** The Stage 1c gate measures rows and columns as well as nonzeros, on the 240 (reported over 237 and over 240) and on the 40, with 0.99 in the mean and nothing below 0.95 unexplained in each measure. "Explained" means a run that names the rule; an instance may also stay behind when it is explained by a named form that was not built (an integer compensator, a compensation with a negative price in a non-locking row, a bound made explicit from a deleted row, a tolerance HiGHS allows and the engine does not). Today the engine stands at 0.9563 / 0.9653 / 0.9424 over 237 of the 240 and 0.8008 / 0.8238 / 0.7654 over the 40. The gate is taken after the fifth of the five further rules: nothing of the five is built before the nine rules are finished, and the run at the end of the merge of parallel columns [M5] is a measurement of the nine rules only.
- **Set by:** Taken as recommended (HISTORY Entry 40: "the gate has the same two numbers in all three measures"; v2_amendments.txt:87; DESIGN.md amendment A8). The list of allowed explanations is design (lowworth_v2_design.txt:218).
- **Where:** DESIGN.md amendment A8 (lines 145-147); stage1c_design/v2_design.txt:201-203, :278; lowworth_v2_design.txt:218; gpu_presolve/analysis/bench240_stage1c.py:34-43; gpu_presolve/results/stage1c/gate/. For the place of the gate: stage1c_design/v2_amendments.txt:19-21; lowworth_v2_amendments.txt:7-8; lowworth_v2_design.txt:197, :217-218.
- **Switch:** `BELOW = 0.95` in gpu_presolve/analysis/bench240_stage1c.py:79 and the 0.99 of the design; the script prints the three measures and enforces nothing.
- **Alternatives:** Nonzeros as the gate and rows and columns as reports (a weak zero-cost singleton rule or merge would pass it); a gate without named exceptions. The gate could also be taken after the nine rules, with the five as a later stage.
- **May change:** Which rules are pursued: by measured single-rule losses the rows gate passes only if neos-3402294-bobin is reached, and eight instances will have to be explained in columns. The exceptions let the held-back forms pass the gate without being built.
- **Test at the end:** Not a quality test. Each named exception is an entry of this register with its own test.
- **State:** measurement only.

### AS-196 In the gate mean the engine's leads offset its lags
- **Assumption:** The gate mean uses each ratio as it is, so an instance where the engine leaves less than the reference raises the mean and can cover instances where it leaves more. The script also prints a second line with every lead counted as zero (0.9552 against 0.9563 in nonzeros on 237 of the 240 today); which line is the gate was not decided.
- **Set by:** Code (the capped line was added by the comparison script's author and checker, HISTORY Entry 42).
- **Where:** gpu_presolve/analysis/bench240_stage1c.py:613-668.
- **Switch:** no switch yet: needs one (which of the two lines is the gate; gpu_presolve/analysis/bench240_stage1c.py always prints both).
- **Alternatives:** Gate on the capped mean; gate on the count of instances below 1.
- **May change:** Whether a stage passes while single instances stay behind HiGHS.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-197 "All instances" of the testbed gate means every instance with a reference, largest band included
- **Assumption:** The testbed gate is taken over every testbed instance that has the reference, 40 at the gate, not per band, including the three instances of the largest band whose time is reported only.
- **Set by:** Taken as recommended (a reading of the owner's decision 3, listed in DESIGN.md section 0, open item 4, for the owner to overrule).
- **Where:** DESIGN.md section 0 (lines 222-223) and section 12 (line 785); STAGE0_REPORT.md section 7 (line 354).
- **Switch:** `--without NAME[,NAME]` in gpu_presolve/analysis/bench240_stage1c.py prints a further line; the gate line itself has no option.
- **Alternatives:** Per band; without the largest band; over the 31.
- **May change:** The gate value: over 31 instances one instance at 73% uses up the 0.99, over 40 one at 67%.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-198 No instance may end larger than with the Stage 1 engine
- **Assumption:** A second lower bound per instance: not more nonzeros, rows or columns than the Stage 1 engine of today, every exception explained; the nine instances where the engine is ahead of the reference today are named first in the report of the milestone of the singleton-column rules and the forcing column [M1] (0.0011 of each end value of the gate is this lead).
- **Set by:** Design (v2_design.txt:204; naming the nine by v2_amendments.txt:72-73).
- **Where:** stage1c_design/v2_design.txt:204, :270; v2_amendments.txt:72-73; gpu_presolve/results/bench240/v4pre_reductions.csv.
- **Switch:** `BEFORE_BENCH` in gpu_presolve/analysis/bench240_stage1c.py:66 names the results of the Stage 1 engine (tag v4pre) to compare with; no option, and the script enforces nothing.
- **Alternatives:** None stated.
- **May change:** It guards the lead of nine instances against the new phase order.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-199 The time rule is 1.2 times HiGHS plus 0.02 s, minimum of three alternating runs
- **Assumption:** An instance passes on time when the engine needs at most 1.2 * t(HiGHS reference) + 0.02 s, both sides taken as the minimum of at least three alternating runs with one process at a time on one machine. It is a gate between 5e4 and 1e7 nonzeros and reported only above. The measured ratios are far from the limit (the design calls the rule slack).
- **Set by:** The owner (HISTORY Entry 13: "3. do the recommendations"; the report offered 0.02 s or 0.05 s).
- **Where:** DESIGN.md section 0 decision 3 (line 190), section 1 (lines 264-267), section 12 (line 783); gpu_presolve/analysis/stage1_time_gate.py:96-98; stage1_compare.py:196; overheads.py:163; gpu_presolve/baseline/bench240_timing_runs.sh:75-86.
- **Switch:** the constants 1.2 and 0.02 in gpu_presolve/analysis/stage1_time_gate.py:96-98, stage1_compare.py:196 and overheads.py:163; no option.
- **Alternatives:** 0.05 s; a tighter factor now that the engine is several times faster; the mean of the runs; a work-unit rule.
- **May change:** Nothing in the reduced model; it decides how much the work budgets may spend and when overheads or a slow rule must be attacked.
- **Test at the end:** Not a quality test: a timing run on one machine; presolve seconds are reported per arm in the end test.
- **State:** measurement only; passes on all 40 and all 240 for Stage 1 (HISTORY Entries 34 and 36).

### AS-200 What the clock covers on each side; warm times; a shifted mean with 0.1 s
- **Assumption:** HiGHS is timed from its "Presolving model" log line to its first "Presolve reductions" line, which contains the build of its data structures, shrink and final scaling but not the copy of the reduced model. The engine is timed from the input model in memory to the finished reduced model, which contains its state build and the hash of the reduced model but not reading the file and not the adapter's conversions. Times are warm (a cold start is reported apart). Per size band the shifted geometric mean (shift 0.1 s) and the median stand next to the per-instance rule.
- **Set by:** Taken as recommended (proposed in section 7 of the Stage 0 report and entered as proposed, DESIGN.md section 0, open item 4; "Time the call, not the process" is the lesson of HISTORY Entry 8).
- **Where:** DESIGN.md section 1 (lines 254-292) and section 12 (line 784); gpu_presolve/baseline/baseline_tool.cpp:128-150, 713-718; gpu_presolve/src/presolve.cpp:500-506; gpu_presolve/analysis/stage1_compare.py:37; stage1_targets.py:20.
- **Switch:** the constant `SHIFT = 0.1` in the two scripts; no switch for what is timed.
- **Alternatives:** Cold timing; the whole call on both sides; the hash left out of the engine's time; HiGHS without its setup, shrink and scaling (a median 22% of its Stage 1 reference time).
- **May change:** Nothing in the reduced model; the factor by which the engine is called faster. Inside cuOpt the solver pays the cold start and the adapter.
- **Test at the end:** Not a quality test; report cold-start presolve seconds inside cuOpt in the end test.
- **State:** measurement only.

### AS-201 The time reference is HiGHS on the same rule set, with its own inefficiencies
- **Assumption:** At each stage the engine is timed against HiGHS restricted to the same rules, not against full HiGHS. That reference contains costs that are HiGHS's own: the 64-bit build, column hashes for a switched-off loop, scaling and the rebuild of the MIP data, which the audit put at 16 to 29% of the Stage 1 reference time; with all cheap rules on, rd-rplusc-21 takes 137 s of 315 s on the 240 because of the marking of one dual-fixing extension.
- **Set by:** Design (DESIGN.md section 1: the Stage 1 reference at Stage 1, full HiGHS once probing and cliques exist); the extra work was named by the audit of HISTORY Entry 17 and restated in Entry 34.
- **Where:** DESIGN.md section 1 (lines 264-269); gpu_presolve/baseline/HIGHS_DIAG.md:1152-1161; gpu_presolve/baseline/stage1_timing_runs.sh:32-34; HISTORY Entries 17 and 34.
- **Switch:** the HiGHS switch set (`HIGHS_DIAG_OFF` and `--presolve-rule-off`) of the timing scripts gpu_presolve/baseline/stage1_timing_runs.sh and bench240_timing_runs.sh.
- **Alternatives:** Full HiGHS as the time reference at every stage; subtract the audited share; a 32-bit build.
- **May change:** How much time allowance the engine seems to have; the speed ratios are somewhat in the engine's favour.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-202 Times come from a shared node and are compared only within one machine
- **Assumption:** Timing runs are three passes, reference and engine in turn, one process at a time. The node is shared (a load of 6 to 9 in one run, 32 to 38 in another) and has changed three times, so only times measured in turn on one machine are compared; single runs of identical work differed by up to 15% in Stage 0. Two early headlines broke the rule ("258 s against 614 s" used a single reference run; "75.0 s against 614.1 s" compared two machines).
- **Set by:** The owner for the minimum of three alternating runs (part of decision 3, HISTORY Entry 13); design for the scripts.
- **Where:** gpu_presolve/baseline/stage1_timing_runs.sh:5-12, 93-100; gpu_presolve/baseline/bench240_timing_runs.sh:75-86; HISTORY Entries 8, 12, 17, 22, 36 and 42.
- **Switch:** `--jobs 1` and `--reps 1` per pass in the timing scripts; three passes.
- **Alternatives:** A reserved quiet machine; more passes; pinned cores.
- **May change:** Nothing in the model; a time difference of a few percent between two engine versions is inside the noise.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-203 The engine is timed in an -O2 build, HiGHS in its CMake Release build
- **Assumption:** In every time comparison the engine is the standalone tool built with `-O2` (the project sets its Release flags to `-O2`), and HiGHS is the patched library built with CMake's Release type, `FAST_BUILD`, 64-bit integers and as a shared library. Neither side is built for the machine, and the two optimisation levels are not set to be equal.
- **Set by:** Code (the build files; no decision is recorded).
- **Where:** gpu_presolve/CMakeLists.txt:17-20; gpu_presolve/baseline/build_highs_diag.sh:50-58; gpu_presolve/baseline/CMakeLists.txt:13-15. The flags that HiGHS's own Release type sets were not read.
- **Switch:** no switch yet: needs one (a build option; today the flag is one line of gpu_presolve/CMakeLists.txt).
- **Alternatives:** The engine with cuOpt's flags or with `-O3`; the same flags on both sides.
- **May change:** The presolve seconds of the engine and with them every ratio to HiGHS (6.9 on the 40 and 8.4 on the 240, AS-249) and the share of the time allowance. Inside cuOpt the engine runs with cuOpt's flags (AS-294), so the standalone seconds are not those of a cluster arm. Nothing in the reduced model if both builds give the same bits.
- **Test at the end:** Not a quality test: one timing run of three alternating passes on the 240 with the engine built with cuOpt's flags against the `-O2` build.
- **State:** measurement only.

### AS-204 A new engine version is timed against the one before, with a mark at 3%
- **Assumption:** Each engine version is timed in three passes in turn with its base version and with its new rules switched off; instances of at least 10 ms that are more than 3% slower are listed. This is a report, not a gate. First values, for the groundwork on the node (minimum of three alternating passes, node load 12 to 40, HISTORY Entry 46): +1.07% on the 240 (16.50 s to 16.68 s), +2.27% on the 31 and +1.91% on the 9 (dlr2 +5.1%, neos-4332810-sesia +3.2%).
- **Set by:** Design (the author of the script, 2026-11-27; the 3% criterion for the lists is fixed by v2_amendments.txt:78-79).
- **Where:** gpu_presolve/analysis/stage1c_time.py:14-19, 135-137, :165; gpu_presolve/tools/stage1c_engine_runs.sh:46-50, 245-275; HISTORY Entry 46 (lines 2537-2541).
- **Switch:** `--limit PERCENT` in stage1c_time.py (default 3); `GPRS_OFF` in stage1c_engine_runs.sh.
- **Alternatives:** A gate on the sum; a tighter mark.
- **May change:** Nothing in the model (AS-269 for the first numbers).
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-205 With its new rules off, a version must give the results of the version before
- **Assumption:** A version with the rules of a milestone switched off must give the same status, sizes, hash of the reduced model, records and fixed-point flag as the version before it on the 240 and the 40. This is what makes two arms that differ in one switch a clean comparison. Each milestone also requires the structure and crush checks clean in three modes and no instance larger than before unexplained.
- **Set by:** Design (v2_design.txt:232-233; lowworth_v2_design.txt:202-218; HISTORY Entry 44).
- **Where:** gpu_presolve/analysis/stage1c_identity.py:14-21, :36; gpu_presolve/tools/stage1c_engine_runs.sh:23-25, 234-243; HISTORY Entry 46 (lines 2526-2530); stage1c_design/v2_design.txt:232-233; lowworth_v2_design.txt:117, 202-218.
- **Switch:** `GPRS_OFF=<families or sub-rules>` for the runs; tool `--off <rule>`; stage1c_identity.py `--sizes-only`.
- **Alternatives:** Acceptance on solver results per milestone.
- **May change:** If it does not hold, an off arm of the end test would not be the earlier engine. "Larger" is judged in rows, columns and nonzeros only.
- **Test at the end:** Not a quality test: the pre-flight of every end-test pair (the off arm equals the previous version's result on all 240).
- **State:** measurement only; shown for the groundwork on the node: with the four new families on and off the engine equals the engine before it on the 240, the 31 and the 9 (HISTORY Entry 46).

### AS-206 Each milestone of the nine rules has named acceptance instances
- **Assumption:** The milestones are accepted on sizes against the cumulative HiGHS reference, not on solver results. Singleton-column rules and forcing column [M1]: at or above 0.99 in the three measures or explained by the pivot kind that is not built; neos-3402294-bobin at or above 0.95 of 381,000 nonzeros, 32,932 rows and 792 columns or traced. Simple probing [M2]: highschool1-aigio, neos-1445765 and brazil3 at or above 0.95 or "explained by explicit bounds", which accepts a known weaker form. Dominance between parallel columns [M3]: nw04, decomp2, neos-3004026-krka, neos-2987310-joes, rail02, rail01, co-100, n3div36 and the two satellites instances at or above 0.95; the crush check passes in default mode on the benchmark instances with points (173 when the design was written, 184 in the newest check, AS-276). Dual bound strengthening and the single-lock alias [M4]: supportcase12 at or above 0.95 except for its merge share; var-smallemery-m6j6, neos859080 and the radiation instances at or above 0.95 or the gap named. Merge of parallel columns [M5]: the table with `range_kept` and `always`, with and without the three largest gains; the expected exception with `range_kept` is neos-4722843-widden.
- **Set by:** Design (v2_design.txt:237-253; neos-3402294-bobin as acceptance instance by v2_amendments.txt:25-30).
- **Where:** stage1c_design/v2_design.txt:205, :237, :241, :245, :249, :252-253; v2_amendments.txt:25-30.
- **Switch:** `options_t::merge_integers` for the two settings of the table of the merge of parallel columns [M5]; the acceptance values themselves are text of the design, in no script.
- **Alternatives:** Acceptance without named exceptions; acceptance on solver results.
- **May change:** When a milestone counts as done.
- **Test at the end:** Not a quality test; the table of the merge of parallel columns [M5] is the size half of the question of AS-001.
- **State:** planned.

### AS-207 The cheap-rule references are size runs, not timing runs
- **Assumption:** The HiGHS references for the cheap column rules were run with 8, 4 or 3 jobs at a time and give sizes and counters only. The time rule of that stage therefore needs a timing run of its own with the cheap switch set; none is stored, and the overheads script without a timing name falls back to the whole-call seconds of a single run on a busy node.
- **Set by:** Design (the reference script: "they are not timing runs (several jobs at a time)").
- **Where:** gpu_presolve/baseline/stage1c_reference_runs.sh:5-8, :128, 136-148; gpu_presolve/analysis/overheads.py:288-292.
- **Switch:** `--jobs` in the reference script; a timing name as the last argument of overheads.py.
- **Alternatives:** A timing script with the cheap switch set, as for Stage 1.
- **May change:** Nothing in the model.
- **Test at the end:** Not a quality test.
- **State:** planned.

### AS-208 Overheads are tracked in every run and optimised only when they block beating HiGHS
- **Assumption:** The time outside the rules (state build, storage, output, hash, adapter, transfers) is reported with every run and is attacked only when it stands between the engine and HiGHS on time. With the engine now on the branch (tag s1c_m0, HISTORY Entry 46) it is 60% of the engine's time on the 240 (16.7 s; state build 42.7%, output 5.5%, hash 6.2%), where the engine uses 8.7% of the time allowance and its overheads alone 5.2% (reference: HiGHS with the Stage 1 rule set, one run on a busy node). On the 40 it is 61% (83.3 s; state build 42.7%), with 11.3% and 6.9% of the allowance against a stored reference from another machine. Timed in turn on one machine (the engine before the groundwork, tag v4pre, HISTORY Entry 36) the engine uses 10.3% of the allowance on the 240 and its overheads alone 6.3%.
- **Set by:** The owner (HISTORY Entry 33, 2026-11-27: "We can do multiple optimization paths for the overheads. But keep track of the overheads during our runs. When they become the bottleneck of beating highs, we can attack them.").
- **Where:** DESIGN.md amendment A9 (lines 154-157); gpu_presolve/analysis/overheads.py; HISTORY Entry 33 (lines 1982-1998), Entry 36 (line 2113) and Entry 46 (lines 2546-2548); gpu_presolve/results/stage1/overheads.csv (the lines "bench240 s1c_m0" and "testbed s1c_m0", which are in the working tree and not yet committed).
- **Switch:** no switch yet: needs one only when an optimisation path is built; the report is gpu_presolve/analysis/overheads.py `<tag>` or `--bench <tag>`.
- **Alternatives:** Optimise the state build, the output and the hash now (device state build, huge pages, a cheaper hash).
- **May change:** Presolve seconds only, and so the time left for the solve in a time-limited benchmark.
- **Test at the end:** Not a quality test by itself: presolve seconds per arm and instance next to the quality numbers.
- **State:** measurement only.

### AS-209 Sizes are compared as rows, columns and nonzeros; added nonzeros count against the measure
- **Assumption:** Both reduced models are compared in one convention (minimisation, implied integers counted separately), as rows, columns and nonzeros together. The plan says nonzeros added by clique extension and by aggregator fill-in are reported apart from the size measure; no script does this yet, so a reduction that grows the model counts as a loss.
- **Set by:** Taken as recommended (proposed in section 7 of the Stage 0 report and entered as proposed; by the owner's message of Entry 45 strengthening may mean added constraints).
- **Where:** DESIGN.md section 12 (lines 757-765); HISTORY Entry 37 (lines 2127-2130); gpu_presolve/analysis/bench240_stage1c.py (no such split).
- **Switch:** no switch yet: needs one (a split of added rows and nonzeros per rule in the scripts).
- **Alternatives:** Count extension and fill-in nonzeros in the size; compare nonzeros only.
- **May change:** Whether strengthening rules that add rows (indicator strengthening, clique extension, lifted rows) are held back because they lower a size figure.
- **Test at the end:** Two arms that differ in the adding rule, compared on solution quality, never on size; the end test reports the true reduced size per arm next to the quality measures.
- **State:** planned.

### AS-210 Objective nonzeros are not part of the size measure
- **Assumption:** The comparison script can take matrix nonzeros plus objective nonzeros as a fourth measure, but neither the engine tool nor the HiGHS tool writes objective nonzeros, so it is never compared. A substitution that moves cost onto other columns is therefore not seen.
- **Set by:** Code (the field names are placeholders of the script's author).
- **Where:** gpu_presolve/analysis/bench240_stage1c.py:97-101, 558-597; gpu_presolve/results/stage1c/gate/today_bench240_cheap_all.txt.
- **Switch:** no switch yet: needs one (the fields do not exist in any result).
- **Alternatives:** Write both fields in the two tools.
- **May change:** Nothing in the model; the measure misses the objective fill-in of free column substitution and aliases (AS-095).
- **Test at the end:** Not a quality test.
- **State:** planned.

### AS-211 HiGHS sizes are read from two sources: the DIAG total line and the returned model
- **Assumption:** The plan takes the sizes of a HiGHS reference from the DIAG total line of the patched HiGHS, because HiGHS returns the original model with "Not reduced" when only clique merging changed it and writes its presolved model power-of-two scaled. The Stage 1 scripts do that; the bench240 scripts read the sizes of the returned model. In the stored runs the two agree on all 240 in the cheap and the Stage 1 setting and on the 40, and differ on three instances of full HiGHS (neos-1171737, neos-1171448, neos-662469).
- **Set by:** Design for the plan's rule (Stage 0 report section 7); code for the other reading.
- **Where:** DESIGN.md section 12 (lines 759-761); gpu_presolve/analysis/stage1_compare.py:146-158; gpu_presolve/analysis/bench240_stage1c.py:496-505; gpu_presolve/baseline/README.md:77-83.
- **Switch:** no switch yet: needs one (one source of sizes in all scripts; today gpu_presolve/analysis/stage1_compare.py reads the DIAG total line and the bench240 scripts read the returned model).
- **Alternatives:** The DIAG total line everywhere.
- **May change:** The distance to full HiGHS on instances where only clique merging or bound tightening changed the model.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-212 The presolve-only report inside cuOpt approximates how the solver calls the presolve
- **Assumption:** Size figures "inside cuOpt" come from a disabled test that calls the presolver without a solve and without scaling, with dual reductions on, 4 threads for PaPILO, a time limit of 600 s and the modes 0, 2 and 3. An instance that a presolver solves or proves infeasible counts as one remaining nonzero. In a real solve the thread count and the time limit come from the solver.
- **Set by:** Design (the assistant's report, HISTORY Entries 27 and 37); "reductions first" and "no scaling" are the owner's (Entry 27).
- **Where:** cpp/tests/mip/gprs_presolve_test.cu:216-282; gpu_presolve/analysis/bench240_reductions.py:1-23; results in gpu_presolve/results/bench240/cuopt_report_v3 and cuopt_report_v4.
- **Switch:** environment variables `GPRS_REDUCTIONS_LIST`, `GPRS_REDUCTIONS_THREADS` (default 4), `GPRS_REDUCTIONS_MODES` (default "023").
- **Alternatives:** Read the sizes from the logs of real solves; include mode 1; primal-only mode.
- **May change:** Nothing in the engine. PaPILO's result depends on its thread budget and time limit, so its figure in the report (0.616 on the 240) can differ from what a cluster run leaves.
- **Test at the end:** Not a quality test; the cluster logs give the sizes of the real arms.
- **State:** measurement only.

### AS-213 "cuOpt as is" means PaPILO as cuOpt configures it, not stock PaPILO
- **Assumption:** The reference arm is PaPILO at a pinned commit with cuOpt's method list (cuOpt's GF2, BHW and single-lock methods, and indicator strengthening before it), a feasibility tolerance of 1e-5, at most four threads, no round cap, clique merging on, and a probing batch capped at 32 or 1024. PaPILO's internal defaults at the pinned commit were not read: the commit is not on this machine.
- **Set by:** Code (existing cuOpt code).
- **Where:** cpp/src/presolve/third_party_presolve.cpp:725-762, 775-784, 810-818, 918-922, :936; cpp/CMakeLists.txt:297-304; cpp/src/presolve/presolve_budget_policy.hpp:116-123.
- **Switch:** `presolve_gprs` = 0 with cuOpt's defaults.
- **Alternatives:** Stock PaPILO; full HiGHS as the reference inside cuOpt (AS-218).
- **May change:** Every "against PaPILO" number is against this configuration.
- **Test at the end:** The baseline arm of the main comparison.
- **State:** in cuOpt as is.

### AS-214 A cluster arm is a flipped-default commit, run once; the baseline has no stored run
- **Assumption:** An arm of the end test is a pushed benchmark commit that changes one default. The plan foresees the regular benchmark set with one run per arm and the stored baseline reused. Since no arm uses scaling any more, the baseline is PaPILO with scaling off, which has no stored run, and no baseline exists at the branch's upstream base `0d0915a8c`. Four benchmark commits were prepared and not submitted: PaPILO with scaling on (`af4a94d3e`) and off (`384524b34`), and the new presolve in mode 2 with scaling on (`fdbd9f8a6`) and off (`c0b04112c`); none exists for mode 3. Whether one run per arm separates two arms that differ in one switch is not established.
- **Set by:** Design (DESIGN.md sections 10 to 12); the owner for "no scaling" (HISTORY Entry 27) and for launching cluster runs without asking (Entry 13).
- **Where:** DESIGN.md section 11 (lines 742-743) and section 12 (line 750); HISTORY Entry 23 (lines 1409-1413) and Entry 27 (lines 1634-1637).
- **Switch:** the edited defaults `presolve_gprs` and `mip_scaling` in a benchmark commit.
- **Alternatives:** Several runs per arm; a fresh baseline in the same submission; a runtime parameter so that one build serves all arms.
- **May change:** What difference can be seen at all. A comment in cuOpt's own budget code reports a run-to-run noise of 0.53 on a mean error near 12 over 240 instances for another change (cpp/src/presolve/presolve_budget_policy.hpp:34), so one run per arm resolves only larger effects; a baseline from another base commit mixes upstream changes into the difference.
- **Test at the end:** Before the first pair: two runs of the same arm give the spread on primal gap, optimal count and primal integral; the number of runs per arm and the baseline commit are fixed from it.
- **State:** planned (two baseline commits and two mode 2 commits prepared, none submitted).

### AS-215 Acceptance is a cluster A/B with a threshold set before the run; none is given
- **Assumption:** The product decision is a cluster A/B as acceptance test with a threshold set before the run. The plan gives no threshold. The plan also expects that with the Stage 1 rules alone the solver outcome is below PaPILO, and requires of the first cluster run only that no status is wrong and every postsolved solution is feasible.
- **Set by:** Design (DESIGN.md section 13).
- **Where:** DESIGN.md section 13 (lines 797, 799-801).
- **Switch:** `presolve_gprs` 0 against 3.
- **Alternatives:** Acceptance on size alone; acceptance without a preset threshold.
- **May change:** What "done" means for the project.
- **Test at the end:** This is the end test; the owner has to set the threshold (for example on mean primal gap and optimal count) before it runs.
- **State:** planned.

### AS-216 The end test turns one switch at a time; interactions between switches are not tested
- **Assumption:** Every pair of this register changes one switch and leaves all others at their defaults. What a switch is worth is therefore measured only next to the default of every other switch; two choices that stand in for each other, or that only work together, are not seen. With 110 entries of this register asking for a standard cluster comparison and one run per arm (AS-214), a small effect of one switch can also lie below what a single run resolves.
- **Set by:** Design (the rule of this register, DESIGN.md amendment A10). The owner's words are "we will test all of them at the end" (HISTORY Entry 45); they do not fix the form of the test.
- **Where:** DESIGN.md amendment A10 (lines 171-176); HISTORY Entry 36 (lines 2082-2083); gpu_presolve/baseline/HIGHS_DIAG.md:508-515; stage1c_design/v2_design.txt:262; cpp/src/presolve/presolve_budget_policy.hpp:34.
- **Switch:** no switch yet: needs one (a way to set several engine options in one arm, AS-152).
- **Alternatives:** Group arms first (every held-back or strengthening choice switched together against the defaults), then single pairs inside a group that moved; several runs per arm.
- **May change:** Which choices the end test confirms. The project's own size runs show rules standing in for each other: in HiGHS on the 240 with all cheap rules on, one part of dual fixing switched off makes another part fix the same columns (irish-electricity 436 to 856 fixings), and on 237 of the 240, 2.2 of the 10.6 summed log gap in nonzeros belong to no single rule (AS-242).
- **Test at the end:** Before the single pairs, one pair with the strengthening candidates switched together (the general integer merge of AS-001, the primal-only exemptions of AS-004, indicator strengthening of AS-003, off against on); the run-to-run spread of one arm is measured first (AS-214).
- **State:** measurement only.

### AS-217 The quality measures named so far are on the primal side only
- **Assumption:** The plan and the owner's words name solution quality as primal gap against the best known value, optimal count and primal integral, which in cuOpt are led by the heuristics. What a stronger formulation changes first, the root LP bound, the final bound and the size of the tree, was not collected in any comparison so far, although the plan's measurement table names the root LP bound. This register adds the dual measures to the standard cluster comparison and reads each pair first on the instances whose reduced model differs, because most of the 240 have the same reduced model in both arms of a one-switch pair.
- **Set by:** Taken as recommended (the three primal measures are the reading of the owner's "solution quality" in the plan, DESIGN.md section 12; the added measures are this register's and were not put to the owner).
- **Where:** the opening of this register; DESIGN.md section 12 (line 750); cpp/src/branch_and_bound/branch_and_bound.cpp:4338 (the log line "Root relaxation objective") and :943 ("Explored ... nodes"); cpp/include/cuopt/mathematical_optimization/mip/solver_stats.hpp:32-39 (solution bound, node count).
- **Switch:** no switch yet: needs one only in the analysis (which measures and which instances a pair is read on); cuOpt already logs the values.
- **Alternatives:** The primal measures alone, over all 240.
- **May change:** Which entries look neutral: a pair that tightens the relaxation (coefficient strengthening, BHW, clique rows, indicator rows, implied-integer flags) can stay inside the spread of the primal measures and still move the bound.
- **Test at the end:** Not a pair: the measures are added to every pair. One check of the convention: for the pairs of AS-003 and AS-048, whether the dual and the primal measures point the same way.
- **State:** measurement only.

### AS-218 Quality against HiGHS is to be measured by cuOpt, presolve off, on HiGHS-presolved models
- **Assumption:** To compare formulation quality with HiGHS inside the solver, the plan adds a cluster arm in which cuOpt with its own presolve off solves the models HiGHS presolved. HiGHS writes its presolved model power-of-two scaled and cannot read indicator constraints, so that arm sees scaled models and misses such instances; with cuOpt's presolve off its probing cache and block BVE do not run either. A tool writes the engine's reduced model as an MPS file for the same kind of run. Nothing of it has been run. Until the HiGHS arm is written before HiGHS's final scaling, the two arms differ in scaling as well as in the presolver.
- **Set by:** Taken as recommended (recommendation of plan version 3, "decided earlier and not reopened").
- **Where:** DESIGN.md section 0 (line 202), section 12 (lines 759-761), section 13 (line 799); gpu_presolve/cuopt_runs/reduce_to_mps.cpp:8-14.
- **Switch:** the parameter `presolve` = 0 on a presolved MPS file.
- **Alternatives:** Compare with HiGHS on size only.
- **May change:** Nothing in the engine.
- **Test at the end:** Itself a pair of arms: cuOpt with its presolver set to none and `mip_scaling` 0 on HiGHS-presolved models against the same on the MPS files the engine's tool writes; solutions are mapped back to the original model for the primal gap. For the two arms to differ in the presolver only, the HiGHS arm has to use the reduced model before HiGHS's final scaling (the baseline tool writes it before the stage `final_scaling`, or undoes the scaling), so that both MPS files are unscaled. Until that tool change exists the pair is not a one-switch test and is to be marked so. The pairs that switch one HiGHS rule (AS-014, AS-113) are not affected, because both of their arms are scaled by HiGHS in the same way.
- **State:** planned.

### AS-219 The probing gate has no threshold; the clique gate rests on eight instances
- **Assumption:** The probing gate compares the model after the whole presolve with and without the new probing (rows, columns, nonzeros, bounds), not the number of fixings; no threshold is set. The clique gate takes the instances where HiGHS's merging leaves at least 1% fewer rows (8 of the 31) and asks for geometric means of rows(reference) / rows(ours) and nnz(reference) / nnz(ours) of at least 0.99 each, extension nonzeros apart. The HiGHS references for both are not built.
- **Set by:** Taken as recommended (proposed in the Stage 0 report; reading "at least 0.99 of the reference" as reference over ours is listed for the owner to overrule); gating cliques apart from probing is the owner's decision 2.
- **Where:** DESIGN.md section 12 (lines 786-787) and section 0 (lines 226-227).
- **Switch:** no switch yet: needs one (a threshold for the probing gate; the instance list and the 0.99 of the clique gate are text of the plan, in no script).
- **Alternatives:** A threshold for probing; all instances for the clique gate; the 240.
- **May change:** When probing and cliques are called done. Probing's value for the solver (implications, tighter bounds) is only partly in the size.
- **Test at the end:** Not quality tests; their counterparts are the probing and clique pairs of the end test.
- **State:** planned.

### AS-220 The two arms of the main comparison differ in presolve seconds as well as reductions
- **Assumption:** Summed presolve seconds on the 240 on a busy node: the new presolve 18 s standalone (the engine before the groundwork, tag v4pre, one run; 16.7 s with the engine now on the branch) and 94 to 128 s for the call inside cuOpt, PaPILO with cuOpt's reductions 1,127 s with 4 threads, HiGHS 4,851 s with 1 thread. A cluster comparison under a time limit therefore gives the arm with the new presolve more solve time, apart from any difference in the model. The consequence for the end test is a reading of this register, not a statement of the log. Presolve seconds are to be read from the log line "GPRS presolve" (adapter seconds and engine seconds), not from "Papilo presolve time", which counts from the start of the solve and includes scaling, conversions and the new presolve.
- **Set by:** Design (a measured finding of HISTORY Entries 27 and 37; the consequence for the end test is this register's reading).
- **Where:** HISTORY Entry 27 (lines 1687-1689) and Entry 37 (lines 2134-2135); cpp/src/presolve/third_party_presolve.cpp:1116. The two log lines: cpp/src/mip_heuristics/solve.cu:781-785, :923; cpp/src/presolve/gprs_presolve.cpp:262-276.
- **Switch:** no switch yet: needs one (a time limit counted from the end of presolve, for a comparison at equal solve time).
- **Alternatives:** Report quality at equal solve time (time limit counted after presolve) next to quality at equal total time.
- **May change:** Primal integral and gap at the time limit move with the seconds left for the solve, so a quality gain can come from speed and not from a stronger model, and a loss from weaker reductions can be hidden by it.
- **Test at the end:** For every pair report presolve seconds per instance, and repeat the main comparison once with the time limit counted from the end of presolve.
- **State:** measurement only.

### AS-221 Correct means structure check, crush check and solver check pass; solver quality never decides
- **Assumption:** A result counts as correct when three checks pass: the structure check (reads the postsolve records and proves that no reduction relaxes), the crush check (known feasible points of the original model are carried into the reduced model and back; a result without a dual finding, which primal-only mode guarantees, must give the point back unchanged) and the solver check (HiGHS on the LP relaxations and the reduced MIP). Whether a reduction is right is decided by these, the unit tests, the random campaign and wrong engines the checks must catch, never by a solver result; a rule that leaves a new postsolve record comes with the check extension that reads it.
- **Set by:** Design (after the audit of HISTORY Entry 17, Entries 20, 21 and 25; item 8 of the gate in v2_design.txt:209). Entries 17 to 21 belong to the six hours of independent work, without a check-in.
- **Where:** gpu_presolve/check/gprs_check.cpp:8-46; gpu_presolve/check/structure_check.hpp:10-16; gpu_presolve/check/crush_check.hpp:72-82; stage1c_design/v2_design.txt:136, :155, :190, :209.
- **Switch:** flags of gpu_presolve/check/run_check.py: `--primal-only`, `--lp-time-limit`, `--mip-time-limit`, `--require-points`, `--cost-variants`, `--push-points`.
- **Alternatives:** Exact solves of both models; a certificate per reduction; the cluster run as a correctness test as well.
- **May change:** Nothing in the model; the end test over solution quality is read as a statement about strength only.
- **Test at the end:** In every cluster arm count wrong statuses and postsolved solutions that fail the original model, apart from the quality measures; both counts must be zero.
- **State:** measurement only.

### AS-222 The correctness checks pass violations up to their own tolerances
- **Assumption:** The checks accept what the table lists. A point of a solution file is used when it meets the original model within 1e-7 with integer columns within 1e-5 of integers; violations are relative to the largest term of the row or 1. The structure check counts a row whose test would cost too much, or that is made of more than 16 equations, as not tested, neither passed nor failed (0.19% of such rows in one campaign of 132 million generated models).

| Name | Value | Place |
|---|---|---|
| `point_tolerance`, `point_integrality` (a stored point is used) | 1e-7, 1e-5 | gpu_presolve/check/crush_check.hpp:76-77 |
| `check_tolerance` (derived points) | 1e-6 | crush_check.hpp:78 |
| `identity_tolerance` (two objective values of one point, relative to the cost terms) | 1e-9 | crush_check.hpp:80 |
| `objective_tolerance` (values of different points; a proved reduced optimum against the known one) | 1e-6 relative | crush_check.hpp:82; gpu_presolve/check/gprs_check.cpp:794-796 |
| `structure_tolerance` (relative to the largest term and to the multiplier of an equation the row is made of) | 1e-5 | gpu_presolve/check/structure_check.hpp:147 |
| `max_kept_row_work`, `max_kept_work`, `max_row_equations` | 5e7, 2e10, 16 | structure_check.hpp:157-158, :161 |
| `solve_tolerance` of the oracle | 1e-9 | gpu_presolve/tests/oracle.hpp:59 |
- **Set by:** Design (the checks' authors and reviewers, HISTORY Entries 20, 21 and 31; not decided by the owner).
- **Where:** the places of the table; HISTORY Entry 31 (lines 1842-1866).
- **Switch:** the constants of the table, such as `structure_tolerance`, `point_tolerance` and `check_tolerance` (check headers and oracle); no option.
- **Alternatives:** The engine's 1e-6 for the structure check; MIPLIB's 1e-6 for published points, judged against the violation the point came with; exact arithmetic on small models.
- **May change:** Nothing in the engine. A relaxation between 1e-6 and 1e-5 relative, or on a row that is not tested, passes the structure check; an objective error below 1e-6 relative passes; published points that violate rows by more than 1e-7 are rejected (AS-276).
- **Test at the end:** Not a quality test: wrong engines and the campaign show what a tolerance lets through; rerun the structure check on the 240 with 1e-6 and list what fails; report the count of rows not tested with every run.
- **State:** measurement only.

### AS-223 The solver-based check uses HiGHS with short limits, below 1e7 nonzeros only
- **Assumption:** Below 1e7 nonzeros the check solves the LP relaxation of the original and the reduced model (120 s limit) and the reduced MIP with gap 0 (60 s by default, 300 s in the node scripts). Above 1e7 nonzeros no solve is made. HiGHS can ignore its limit (it ran 35 minutes past a 120 s limit on var-smallemery-m6j6), so the tool leaves such a solve behind. A cut-off optimum is seen only where HiGHS proves the optimum of the reduced model: on 9 of the 31 with the merged engine (HISTORY Entry 39).
- **Set by:** Design (the check's author; limits are script defaults).
- **Where:** gpu_presolve/check/gprs_check.cpp:44-45, 124-140, 489-499; gpu_presolve/check/run_check.py:57-66; gpu_presolve/tools/stage1_host_runs.sh:179-198; HISTORY Entries 16 and 39.
- **Switch:** `--lp-time-limit`, `--mip-time-limit`, `--threads`, `--solution-time-limit`, `--max-points` of run_check.py.
- **Alternatives:** Longer limits; another solver as the checker.
- **May change:** Nothing in the model; how often a cut-off optimum can be seen.
- **Test at the end:** The end test adds what this check lacks: every postsolved cuOpt solution tested in the original model, and the optimal count against best known values on all 240.
- **State:** measurement only.

### AS-224 On the 240 a known objective value is a bound, not the optimum
- **Assumption:** On the testbed the objective of an instance is the published optimum unless marked with a star. On the 240 every value comes from cuOpt's table of best known values and is passed as an upper bound only, so a reduced optimum below it is not a failure there.
- **Set by:** Design (the testbed script: "the check takes it as a bound, not as the optimum").
- **Where:** gpu_presolve/testbed/make_benchmark_testbed.py:1-16, 145-149; gpu_presolve/check/run_check.py:119-144; gpu_presolve/check/gprs_check.cpp:834-839; the table is benchmarks/linear_programming/cuopt/miplib2017_bks.hpp.
- **Switch:** `--known-is-bound` of gprs_check (set by run_check.py when the testbed value ends with a star).
- **Alternatives:** Mark proved optima in the 240 list as exact; MIPLIB's own solution values.
- **May change:** A reduction that makes the reduced optimum better than the true optimum (a relaxation) is not caught by this check on the 240; the structure check has to catch it. A wrong table value hides a cut-off optimum or raises a false alarm.
- **Test at the end:** The cluster's primal gap uses the same table, so both use one source of best known values.
- **State:** measurement only.

### AS-225 Each stored point gives up to four pushed points; instances get other cost vectors too
- **Assumption:** Each point of a solution file gives up to four more points made by moving its columns as far as bounds and rows allow, and each instance is also presolved with other cost vectors (two by default, one on the 240 and on the large set) to reach the cost paths. On the large set one pushed point and at most four points per instance are used.
- **Set by:** Design (the check's reviewer, HISTORY Entry 20).
- **Where:** gpu_presolve/check/crush_check.hpp:285-340; gpu_presolve/check/gprs_check.cpp:428-445; gpu_presolve/check/run_check.py:64-66; gpu_presolve/tools/stage1c_engine_runs.sh:277-294.
- **Switch:** `--push-points`, `--cost-variants`, `--max-points` of run_check.py.
- **Alternatives:** More variants; random feasible points.
- **May change:** How many dual and bound paths the crush check reaches.
- **Test at the end:** Not a quality test: wrong engines.
- **State:** measurement only.

### AS-226 A check counts only after it failed on a deliberately wrong engine
- **Assumption:** The checks are trusted because deliberately wrong copies of the engine (36 at first, 49 before the groundwork, 51 with it, 58 after its review) fail them on the instances they change. A wrong engine that fits no stored recipe, or an error on a rule path that no local instance uses, is outside this evidence.
- **Set by:** Design (the lesson of HISTORY Entry 17).
- **Where:** gpu_presolve/check/make_mutants.py:3-17; HISTORY Entries 20, 21, 25, 44 and 45.
- **Switch:** make_mutants.py `<out dir> [name ...]`; `-DGPRS_ENGINE_SRC=<mutant>/src` for the check build.
- **Alternatives:** Automatic mutation of the engine sources.
- **May change:** Nothing in the model.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-227 Every rule change needs a random campaign of tens of millions of models
- **Assumption:** Every rule change needs tens of millions of generated tiny models in the release and in the assertion build, with and without added singleton columns, judged by the exact oracle next to the structure check and, on one seed in four, by a second presolve of the reduced model, which must come back unchanged, because the one wrong reduction found so far occurred 30 times in 80 million models (6 under default options). Seeds are consecutive integers from a first seed, each used for 12 generator profiles; the options follow the seed's last digit (three in ten default, one each primal-only, one family or one column reduction off, a random subset of families and reductions with a random mode and a random `merge_integers`, rebuild always, rebuild never, tiny budgets, few rounds). The first seeds of past campaigns are not recorded in the files read.
- **Set by:** Design (a rule the assistant set after the finding of HISTORY Entry 18; written into gpu_presolve/tools/run_campaign.sh).
- **Where:** gpu_presolve/tools/run_campaign.sh:11-14, 35-40; gpu_presolve/tools/gprs_fuzz.cpp:8-36, 62-166, 232-280, :342; HISTORY Entries 18, 19, 21, 25, 31 and 44.
- **Switch:** gprs_fuzz `<first seed> <seeds per profile> [--singletons] [--profile <name>]`.
- **Alternatives:** Larger campaigns; new seed ranges per run; more profiles; a generator for larger constants and mixed rows.
- **May change:** Nothing in the model; the rate of a remaining wrong reduction that the campaign can exclude (about one in the number of models run).
- **Test at the end:** Not a quality test; record the seed range with every campaign so that a later one does not repeat it.
- **State:** measurement only (the tool is in the repository).

### AS-228 The check extensions designed for the nine rules accept what the rules' tolerances allow
- **Assumption:** Two designed check clauses fix what the checks accept. In primal-only mode a column removed by a free singleton record is compared with the original point up to the check tolerance plus the row residual and the difference of the saved columns, divided by the pivot (a pivot growth of up to 1000 is tolerated); columns of the zero-cost and forcing records and merged pairs are exempt. The structure check's own undo asks a saved row to hold only where the rest of the point lies inside the range recomputed from the record, so a wrong undo of the zero-cost rule is caught by the crush check and the oracle, not by this test.
- **Set by:** Design (two binding amendments after the validity review, v2_amendments.txt:48-52 and 57-58).
- **Where:** stage1c_design/v2_amendments.txt:48-52, 57-58; v2_design.txt:155; v2_reviews.txt:13-16, 28-31.
- **Switch:** no switch yet: needs one only if the stricter clause is wanted (exempt the column and require the saved row to hold); the two clauses are designed, not in code.
- **Alternatives:** Exempt the column and require the saved row to hold.
- **May change:** What the checks can detect for the singleton-column rules and the merge.
- **Test at the end:** Not a quality test.
- **State:** designed (milestones of the singleton-column rules and the forcing column [M1] and of the merge of parallel columns [M5]).

### AS-229 Cluster builds carry no engine assertions
- **Assumption:** The engine's assertions are compiled in only in cuOpt's assertion build. Instead of asserting on the cluster, the assertion build of the engine and the structure check run on all 240 on the dev node, which takes minutes (an abort on the cluster loses a batch of 40 instances). It is assumed that this screen is repeated for the exact engine version and options of each cluster arm. Every precondition of the state operations and the consistency check after a rebuild are such assertions, and the status "failed" is never returned, so a broken invariant in a build without them goes on silently.
- **Set by:** Taken as recommended (DESIGN.md amendment A5, HISTORY Entry 23: "So the cluster build needs no engine assertions. This replaces my recommendation of Entry 22"; the owner had asked what the open point meant in Entry 22 and did not decide).
- **Where:** DESIGN.md amendment A5 (lines 62-64); cpp/CMakeLists.txt:642-647; gpu_presolve/src/presolve.cpp:70-74; gpu_presolve/tools/stage1c_engine_runs.sh:219-224; gpu_presolve/include/gprs/assert.hpp:16-23; gpu_presolve/include/gprs/presolve.hpp:179-180; gpu_presolve/CMakeLists.txt:15; DESIGN.md section 8 (a failure is an assertion in the standalone project and in benchmark commits).
- **Switch:** the CMake option `DEFINE_ASSERT` of the cuOpt build; `-DGPRS_ASSERT=ON` for the standalone build.
- **Alternatives:** Assertions on in the cluster build; cheap checks that stay in a release build, with a fallback to the unreduced problem.
- **May change:** Inside cuOpt the engine gets the model as cuOpt hands it over, which is not byte for byte the dump tested on the dev node (AS-281); a broken invariant in a release build goes on silently and can give a wrong reduced model.
- **Test at the end:** Not a quality test: one assertion-build run of the presolve-only report inside cuOpt on the 240 before the end test, and the dev-node screen for every arm's engine commit and options.
- **State:** measurement only; the screen was run on the merged Stage 1 engine and on the engine with the groundwork (HISTORY Entry 46).

### AS-230 Before a cluster run the adapter passes a pre-flight: 40 models, 30 s per mode
- **Assumption:** Before a cluster run the adapter is run on the dev node on 40 models (24 MIPLIB instances and 16 edge-case models of the repository) for 30 s per mode with 8 CPU threads in an assertion build. It was run once, for the Stage 1 engine; of 120 runs five aborted on three assertions in cuOpt code (AS-284).
- **Set by:** Design (HISTORY Entries 22 and 23).
- **Where:** gpu_presolve/cuopt_runs/preflight.sh:19-45; HISTORY Entry 23 (lines 1372-1393).
- **Switch:** the arguments of preflight.sh (output directory, seconds, parallel runs, instances); modes 0 to 3 through `GPRS_PREFLIGHT_MODE`.
- **Alternatives:** A longer limit; a release build; more instances.
- **May change:** What counts as "the adapter works".
- **Test at the end:** Not a quality test: before the cluster arms, the pre-flight again for the engine commit of the arm, in all four modes.
- **State:** measurement only.

### AS-231 The working tree reads the adapter mode from the environment through an uncommitted patch
- **Assumption:** The local tree carries an uncommitted patch that lets the environment variable `GPRS_PREFLIGHT_MODE` override the mode, without a range check (the code comment says "PREFLIGHT ONLY, NOT FOR COMMIT"), next to uncommitted dev-container cuDSS changes. It is assumed that neither reaches a commit or a cluster build.
- **Set by:** Design (for the pre-flight, HISTORY Entry 23: "applied for that build only").
- **Where:** cpp/src/mip_heuristics/solve.cu:740-745 (uncommitted); gpu_presolve/cuopt_runs/preflight_mode.patch; cpp/cmake/thirdparty/FindCUDSS.cmake and cpp/src/barrier/sparse_cholesky.cuh (uncommitted).
- **Switch:** the environment variable `GPRS_PREFLIGHT_MODE` in a build with the patch.
- **Alternatives:** A registered runtime parameter for the mode.
- **May change:** With the patch applied a run's mode depends on the shell environment, so a result can be attributed to the wrong mode.
- **Test at the end:** Not a quality test: before each cluster build, the diff of solve.cu must be empty.
- **State:** measurement only; applied in the working tree today.

### AS-232 The adapter's unit tests cover four small instances and no refusal, limit or unbounded path
- **Assumption:** The adapter is tested on four small instances in three modes (objective within 1e-6 of a run without the new presolve, 20 s limit), on one infeasible model, and on a hand-made model for the undo with minimise and maximise and with dual reductions on and off: 27 tests. There is no test of a refused model, of the status "unbounded or infeasible" or of the time limit.
- **Set by:** Design (the adapter's tests).
- **Where:** cpp/tests/mip/gprs_presolve_test.cu:36-214.
- **Switch:** no switch yet: needs one only if the engine's oracle is to be run through the adapter (a test option); the missing cases are tests to write, not settings.
- **Alternatives:** Tests for the paths named above; the engine's oracle run through the adapter.
- **May change:** Which adapter paths are known to work before a cluster run.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

## 7. Findings that later choices rest on

### AS-233 Finding: the Stage 1 engine is at or above HiGHS with the same rule set
- **Assumption:** Against HiGHS restricted to our Stage 1 rule set the engine leaves fewer nonzeros on 18 of the 240, the same on 220 and more on one (traininstance2, 3 nonzeros), ratio 1.002; on the 40 the ratio is 1.0028 (0.9955 before the row addition and the rounds on explicit bounds), with nothing below 0.95. Stage 1 is taken as passed on that basis and the cheap column rules build on it. The three counts of the log add up to 239; the nine-rule design gives 18, 221 and 1.
- **Set by:** Design (a measured finding, HISTORY Entries 26, 28 and 39; Stage 1 is taken as passed on it).
- **Where:** DESIGN.md amendment A2 (lines 37-39); HISTORY Entry 28 (lines 1716-1718) and Entry 39 (lines 2184-2186); stage1c_design/v2_design.txt:199.
- **Switch:** no switch of its own: the finding is reproduced with gpu_presolve/analysis/stage1_compare.py; in the engine the Stage 1 rule set is `options_t::enabled` with the four column families off.
- **Alternatives:** None; it can be repeated.
- **May change:** If it stopped holding after a later change, the base of the Stage 1c gate and the per-instance lower bound would be wrong.
- **Test at the end:** Not a quality test: rerun the Stage 1 comparison on the 40 and the 240 with the final engine restricted to the Stage 1 families.
- **State:** measurement only.

### AS-234 Finding: what each presolve leaves on the 240; belief that fewer nonzeros mean better quality
- **Assumption:** Presolve only, no scaling, geometric mean of remaining nonzeros on 237 of the 240: our Stage 1 engine 0.788, HiGHS with our Stage 1 rule set 0.789, HiGHS with all cheap rules 0.753, HiGHS full 0.676, PaPILO with cuOpt's reductions 0.689 (Entry 37, merged engine: 0.7885, with cuOpt's four 0.7862, PaPILO 0.6905, HiGHS full 0.6767). Over all 240: engine 0.790, PaPILO 0.616, HiGHS full 0.613. The distance to PaPILO is largest between 5e5 and 1e7 nonzeros (52 instances: 0.723 against 0.385). The order of the remaining work rests on these numbers, and the plan expects the solver outcome with the Stage 1 rules to be below PaPILO; no cuOpt run has tested either.
- **Set by:** Design (a measured finding, HISTORY Entries 27, 28 and 37; the expectation that the solver outcome is below PaPILO is the plan's, DESIGN.md section 13, line 797).
- **Where:** DESIGN.md amendment A8 (lines 93-97); HISTORY Entry 27 (lines 1657-1686), Entry 28 (lines 1724-1730), Entry 37 (lines 2131-2133).
- **Switch:** `presolve_gprs` for the test; the finding itself is reproduced with gpu_presolve/analysis/bench240_reductions.py and bench240_stage1c.py.
- **Alternatives:** A cluster run now, to learn early how reductions and quality relate.
- **May change:** It sets the size of the gap to close and the order of work; if quality follows other properties (added rows, integer merges, coefficient strength), the order may be wrong.
- **Test at the end:** The main comparison, which could be run once early as a calibration: `presolve_gprs` 0 against 3, scaling off, on the 240. Repeated after each stage it shows how quality follows size.
- **State:** measurement only; no cluster run exists.

### AS-235 Finding: HiGHS spends most of its presolve time in probing and enumeration for little size
- **Assumption:** In HiGHS's full presolve on the 240 (mean share of an instance's nonzeros removed; share of time): initial scan 10.9% (3%), parallel rows and columns 4.7% (6%), fast loop 4.2% (9%), sparsify 1.7% (0.4%), enumeration 1.3% (13%), pairwise dominated columns 0.8% (12%), probing 0.6% (52%), aggregator -0.3% of the nonzeros but 1.7% of the rows (0.1%). About a third of the gap from the engine to full HiGHS is cheap rules, which cost HiGHS 146 s more (302 against 156 s). Probing and enumeration decide single instances all the same (enumeration removes 56 to 76% on ex9, ex10, neos8 and sorrell3).
- **Set by:** Design (a measured finding, HISTORY Entry 28, with the instrumented HiGHS; the owner chose the build order on it: "Yes, cheap rules first", Entry 29).
- **Where:** DESIGN.md amendment A8 (lines 97-98); HISTORY Entry 28 (lines 1732-1744); gpu_presolve/analysis/bench240_highs_stages.py.
- **Switch:** no switch of its own: a measurement with the instrumented HiGHS (gpu_presolve/analysis/bench240_highs_stages.py). On the HiGHS side the names `probing` and `enumeration` in `HIGHS_DIAG_OFF` give a pair before the phases are built (AS-113).
- **Alternatives:** None; a measurement of size, not of formulation strength.
- **May change:** The build order rests on it. Probing and cliques give fixings, implications and clique rows whose value for the solver is not in the nonzero count; ranking them last by size may postpone the part that strengthens most.
- **Test at the end:** Before the phases are built: the HiGHS family pairs of AS-113 for `probing` and `enumeration`. Once they exist: standard cluster comparison, wide phases off against on; the cluster run after the probing and clique stage against the run before it.
- **State:** measurement only.

### AS-236 Finding: four cheap rules carry the stage; the textbook column rules are worth nothing measurable
- **Assumption:** With one rule switched off in HiGHS (all cheap rules otherwise on, the 240, nonzeros without over with): parallel columns 1.010 (39 instances change), simple probing on equations 1.010 (8), free column substitution 1.009 (44), dual-fixing substitution 1.006 (40), zero-cost singleton 1.001 (16), implied-integer conversion 1.0003 (7), implied equations 1.0002 (11), dominated, weakly dominated and forcing column 1.0001 each, column stuffing 1.0000. Switching off HiGHS's whole dual fixing loses exactly what its substitution extension loses, so the lock rule the engine has is all of the rest.
- **Set by:** Design (a measured finding, HISTORY Entry 29, table added to the entry).
- **Where:** HISTORY Entry 29 (lines 1789-1807); gpu_presolve/results/bench240/cheap_ablation.
- **Switch:** in the engine `options_t::sub_enabled` and `options_t::enabled` hold the switches of these rules (no rule reads them yet); the finding comes from the HiGHS rule bits and names of AS-188.
- **Alternatives:** None; a measurement.
- **May change:** The build order of the nine rules and the first proposal to leave five rules out rest on it.
- **Test at the end:** Engine pairs with one reduction switched off each, first as size runs, then on the cluster for the rules that change many instances (AS-050).
- **State:** measurement only.

### AS-237 Finding: the five further rules are worth about 0.06% in nonzeros on the 240
- **Assumption:** With all left-out rules switched off together in HiGHS on the 240: remaining nonzeros 1.0006, rows 1.0015, columns 1.0006, 19 instances change (largest: ns1208400 1.110, where the engine already has the smaller size by another route; neos-3656078-kumeu 1.029 and neos-2746589-doon 1.028, mostly the forcing column, which is among the nine required rules and is not built yet; csched007 1.016). In HiGHS's forms the five rules are what is left between the nine rules and all cheap rules: 0.9994 / 0.9985 / 0.9993 on 237 of the 240 and 0.9974 / 0.9998 / 0.9972 on the 40. Per rule, with HiGHS on 237 of the 240:

| Rule | Worth in HiGHS (nonzeros / rows / columns without over with) | Note |
|---|---|---|
| Implied equation | 1.00017 / 1.00079 / 1.00033, 11 instances | csched007 and roi5alpha10n8 are reached by the free singleton with an inequality pivot; without the two the bound is 0.00009 / 0.00013 / 0.00021 in the mean log |
| Conversion with scaling | 1.00030 / 1.00017 / 1.00009, 7 instances | without ns1208400 HiGHS ends smaller in nonzeros and columns without the conversion (blp-ic98 and blp-ar98 end 3% smaller); supportcase42 is the instance that needs it |
| Column scaling alone | no switch in HiGHS; by counters at most physiciansched3-3's 28 rows | built for completeness |
| Column stuffing | one column and one nonzero on sp97ar | seymour1 is not a stuffing instance (refuted by the worth verifier) |
| Single-equation extension | 1 row, 8 columns, 20 nonzeros on two instances | no change on either set when the other dual-fixing parts are on; it has a cost side (rd-rplusc-21) |
| Dominated-column tests beyond the forcing column | 1.00003 / 1.00003 / 1.00005, three instances | no change on the 40 although HiGHS credits 38,984,796 weakly dominated fixings there |
- **Set by:** Design (a finding measured on HiGHS by the design agents and recomputed by the worth verifier, HISTORY Entries 36 and 42; the owner saw the figure on the 240 and decided "Build them too", Entry 38).
- **Where:** HISTORY Entry 36 (lines 2082-2096) and Entry 42 (lines 2326-2347); stage1c_design/lowworth_v2_design.txt:28, :35, :42, :49, :56, :63, :163-183; lowworth_v2_reviews.txt:63, :71, 84-87, :106; lowworth_v2_amendments.txt:31-35, 46-51.
- **Switch:** no switch yet: needs one; the eight designed switches of AS-011 (`implied_equations` to `dominated_columns`) are not in code. The finding comes from HiGHS reference runs with the names and rule bits of AS-188.
- **Alternatives:** None; if the engine's forms of the nine rules reach less than HiGHS's, the five could matter more or less.
- **May change:** This belief is the reason the rules were ranked last and the reason their tolerances and restrictions are chosen for safety, not reach. The size belief says nothing about solution quality.
- **Test at the end:** Engine sizes at the milestones of the five further rules [L1 to L5]; then the cluster pairs of AS-011.
- **State:** measurement only.

### AS-238 Finding: the rule ladder, and where the engine stands before the column rules
- **Assumption:** HiGHS with all cheap rules over HiGHS with a growing rule set, nonzeros / rows / columns, on 237 of the 240 and on the 40: Stage 1 set 0.9544 / 0.9640 / 0.9406 and 0.7986 / 0.8235 / 0.7634; with the singleton column rules 0.9726 / 0.9788 / 0.9562 and 0.9233 / 0.9113 / 0.9081; with simple probing 0.9826 / 0.9892 / 0.9772 and 0.9377 / 0.9355 / 0.9297; with parallel columns 0.9932 / 0.9907 / 0.9919 and 0.9396 / 0.9363 / 0.9340; with dual bound strengthening 0.9958 / 0.9937 / 0.9947 and 0.9529 / 0.9566 / 0.9486; with the binary substitution (the nine rules) 0.9994 / 0.9985 / 0.9993 and 0.9974 / 0.9998 / 0.9972. The engine of today against all cheap rules: 0.9563 / 0.9653 / 0.9424 on the 237 (39 / 19 / 32 instances below 0.95). Predicted after the milestone of the merge of parallel columns [M5] by measured single-rule losses: 0.9918 / 0.9874 / 0.9921 without neos-3402294-bobin and 0.9988 / 0.9994 / 0.9975 with it; the review computes 0.9900 / 0.9864 / 0.9898 with `range_kept` and without the engine's present lead.
- **Set by:** Design (a measured finding, HISTORY Entries 40 and 42, with two builds of the instrumented HiGHS; the end values are a prediction of the design, recomputed by the worth review).
- **Where:** DESIGN.md amendment A8 (lines 138-143); HISTORY Entry 40 (lines 2224-2229) and Entry 42 (lines 2326-2347); stage1c_design/v2_design.txt:199, 219-223; v2_reviews.txt:77, 86-89.
- **Switch:** no switch of its own: a measurement of HiGHS reference runs (gpu_presolve/baseline/stage1c_reference_runs.sh); the engine's counterpart is `options_t::enabled` and `options_t::sub_enabled`.
- **Alternatives:** None.
- **May change:** The milestone order and the acceptance values of each milestone come from it; the predicted margin is smaller than one family.
- **Test at the end:** Size runs per milestone; the cluster pair "four column families off against on" measures what the stage is worth in quality.
- **State:** measurement only.

### AS-239 Finding: two large instances carry the testbed distance, and the two sets rank rules differently
- **Assumption:** On the 40, HiGHS with all cheap rules over the engine of today is 0.8008 / 0.8238 / 0.7654 (nonzeros / rows / columns), carried by neos-4332810-sesia (HiGHS 407,609 nonzeros, the engine 34,283,539) and dlr2 (10,542,701 against 72,603,249); without them 0.9357 / 0.9210 / 0.9184 over 38. Both are reached by the singleton column rules followed by the lock rule. The dual-fixing extensions are worth 6% in nonzeros on the 40 and 0.6% on the 240.
- **Set by:** Design (a measured finding, HISTORY Entry 41).
- **Where:** HISTORY Entry 41 (lines 2247-2265, 2296-2300).
- **Switch:** no switch of its own: a measurement; the two instances are taken out of a mean with `--without neos-4332810-sesia,dlr2` in gpu_presolve/analysis/bench240_stage1c.py.
- **Alternatives:** None.
- **May change:** It confirms the build order and is the reason both sets are asked for in order decisions.
- **Test at the end:** Size runs after the milestones of the singleton-column rules and the forcing column [M1] and of dual bound strengthening and the single-lock alias [M4] on the 40.
- **State:** measurement only.

### AS-240 Finding: the zero-cost singleton rule followed by the lock rule reaches neos-3402294-bobin
- **Assumption:** With HiGHS restricted to our Stage 1 rule set plus only the zero-cost singleton rule (2 instances run), neos-3402294-bobin ends at 32,932 rows, 792 columns and 381,000 nonzeros, its size with all cheap rules on. That instance alone is a third of the summed gap in rows on the 240; the rows gate and the build order (singleton columns first) rest on it. The mechanism in the engine is a hypothesis until the milestone of the singleton-column rules and the forcing column [M1] runs.
- **Set by:** Design (a measured finding, HISTORY Entry 40; v2_amendments.txt:25-32).
- **Where:** HISTORY Entry 40 (lines 2220-2223); stage1c_design/v2_amendments.txt:25-32; v2_design.txt:225, :260; gpu_presolve/baseline/stage1c_reference_runs.sh:72 (`BOBIN=8913312`).
- **Switch:** `options_t::sub_enabled[sub_rule::zero_cost_singleton]` in the engine (no rule reads it yet); the finding is a HiGHS run with `--presolve-rule-off 8913312` (a bit mask of the rule bits 5, 7, 8, 19 and 23; `BOBIN` in gpu_presolve/baseline/stage1c_reference_runs.sh:72).
- **Alternatives:** None.
- **May change:** Without bobin the predicted rows value ends at 0.9874, below the gate.
- **Test at the end:** A size run: the engine's own run on bobin after the milestone of the singleton-column rules and the forcing column [M1].
- **State:** measurement only.

### AS-241 Belief: the nine rules reach the gate if the engine's forms reach what HiGHS's reach
- **Assumption:** In HiGHS's forms the nine rules are within 0.1% of all cheap rules in every measure on the 240 and within 0.3% on the 40, so the gate of 0.99 is taken as reachable with the nine rules. The engine's forms are weaker in six named ways: explicit bounds only in every test, no dual bound from another column of the row, the force test of the single-lock alias on the lock row only, exact zero cost for the zero-cost singleton and the forcing column, costs equal to 1e-12 relative for a merge, no single-equation extension.
- **Set by:** Design (a belief that rests on a measurement on HiGHS, HISTORY Entry 42; the condition is stated there, and the weaker forms are listed in the risks of the design).
- **Where:** HISTORY Entry 42 (lines 2339-2342); stage1c_design/v2_design.txt:211, :261.
- **Switch:** no switch yet: needs one per weaker form; the six forms are entries of their own (AS-022, AS-027, AS-031, AS-096, AS-099, AS-038).
- **Alternatives:** None.
- **May change:** The expectation that the gate is reachable; by measured single-rule losses the margin at the end is 0.002 without bobin and 0.006 to 0.009 with it.
- **Test at the end:** A size run after each milestone: the engine against HiGHS with all cheap rules, with the instances below 0.95 traced and the engine's findings per reduction compared with HiGHS's counters.
- **State:** measurement only.

### AS-242 Finding: part of the gap to HiGHS has no single-rule attribution
- **Assumption:** On 237 of the 240, 2.2 of the 10.6 summed log gap in nonzeros, 3.1 of 8.4 in rows and 2.2 of 14.1 in columns are explained by no single switch (bobin, neos-2746589-doon, neos-827175, unitcal_7, n5-3, icir97_tension and others, where rules stand in for each other). Whether the built rules reach them is unknown.
- **Set by:** Design (a finding recomputed by the worth review of the nine-rule design).
- **Where:** stage1c_design/v2_design.txt:217, :262; v2_reviews.txt:77.
- **Switch:** no switch of its own: recomputed from the HiGHS runs with one rule off (`HIGHS_DIAG_OFF` and `--presolve-rule-off`, with the names and rule bits of AS-188).
- **Alternatives:** None.
- **May change:** These instances are the first candidates for the list below 0.95.
- **Test at the end:** Size runs; each instance traced by a run that names the rule.
- **State:** measurement only.

### AS-243 Finding: in Stage 1 the time is in building the state, not in propagation
- **Assumption:** On the finished host engine (the 40, 75.0 s) building the state is 45.7% of the time, the rules 35.8% (parallel rows 11.9, doubleton equations 9.2, bound propagation 6.9, row tightening 5.4, row rules 2.1, dual fixing 0.3), writing the reduced model 8.1% and its hash 8.0%. The first propagation call holds 92.5% of the propagation work, and none of 392 later calls on the 17 local dumps would pass the rule that selects the GPU. Inside the state build the transpose with cross positions and locks is 86 to 94%.
- **Set by:** Design (a measured finding, HISTORY Entries 24 and 32).
- **Where:** DESIGN.md amendment A4 (lines 48-53); HISTORY Entry 24 (lines 1464-1489).
- **Switch:** no switch of its own: a measurement, repeated on every run by gpu_presolve/analysis/overheads.py; the choice that rests on it is the size switch of the device state build (AS-304), which is not built.
- **Alternatives:** None; the shares may shift once the column rules, probing and cliques exist, and the later calls were measured on instances up to 3 million nonzeros only.
- **May change:** It moved the GPU work of Stage 1 from propagation to the state build (AS-304); if the shares change with later rules, that placement is wrong.
- **Test at the end:** Not a quality test: gpu_presolve/analysis/overheads.py on every run shows whether the shares still hold; repeat the split on the 240 and the 40 after the column rules.
- **State:** measurement only.

### AS-244 Belief: below 500,000 nonzeros the GPU wins only on wide work; cascades stay on host
- **Assumption:** At 5e4 to 5e5 nonzeros a full matrix sweep costs 0.5 to 3.1 ms on one CPU thread and 0.06 to 0.24 ms on the GPU, and a device step costs 77 to 93 microseconds before any kernel work (Stage 0, one H200, 32 matrices from 71,477 nonzeros up). So the cascade below 5e5 nonzeros runs on the sequential host engine and the GPU is aimed at probing, cliques, pairwise tests and aggregator candidates. A batched probing round was not measured.
- **Set by:** Design (a belief that rests on the Stage 0 measurements, HISTORY Entries 8, 11 and 12; the conclusion is the plan's, change 1 of the Stage 0 report, which entered with the block of thirteen changes).
- **Where:** DESIGN.md section 1 (lines 240-252) and section 2.1 (lines 312-314); gpu_presolve/results/stage0/STAGE0_REPORT.md section 6.
- **Switch:** no switch yet: needs one (the size switch between host and device, AS-306); the engine is host only today.
- **Alternatives:** GPU cascades at all sizes; another size boundary.
- **May change:** The division of work between host and device; measured on one GPU type and one CPU. Nothing in the reduced model if both engines give the same bits.
- **Test at the end:** Not a quality test: a time measurement with the device forced on and off once a wide phase exists.
- **State:** measurement only.

### AS-245 Belief: a surplus over HiGHS comes from where HiGHS limits itself, and from cliques
- **Assumption:** The plan expects parity through a complete rule set of its own and a surplus from coverage where HiGHS limits itself (its probing contingent starting at 1,000, the aggregator pass that ends after three fill-in failures, stages it runs once) and from clique strengthening. One source was measured: with HiGHS's probing limits off the 31 end with 1.8% fewer nonzeros in the geometric mean, for 15 times the probing time and 31 times the probes; the gain sits on supportcase7, bab6 and brazil3. The other two are not measured.
- **Set by:** Design (a belief of the plan); the one measurement is from the Stage 0 report, section 13.
- **Where:** DESIGN.md section 1 (lines 294-298) and section 13 (line 818); STAGE0_REPORT.md lines 626-656.
- **Switch:** no switch in the engine yet: needs one when probing is built; on the HiGHS side `HIGHS_DIAG_PROBING_NOSTOP=1`.
- **Alternatives:** No surplus from probing; a surplus only from rules HiGHS lacks.
- **May change:** It sizes the probing work budget and is a reason for building probing at all. It was measured in nonzeros only; what the fixings, implications and cliques are worth to the solver was not.
- **Test at the end:** Standard cluster comparison when probing exists: a budget at HiGHS's coverage against complete coverage (AS-111).
- **State:** measurement only (one of three sources measured).

### AS-246 Finding: what HiGHS's rule families are worth on the 31 (Stage 0)
- **Assumption:** On the 31, summed: HiGHS restricted to our Stage 1 rule set reaches 57.9% of the rows, 95.2% of the columns and 87.5% of the nonzeros that full HiGHS removes (82.8% of the nonzeros with the stricter reference of decision 1); with probing and the clique rules 97.5%, 97.0% and 97.2%. Full HiGHS removes 34.2% of the nonzeros of the 31, and four instances hold 73% of that. Without probing HiGHS keeps 9.3% more nonzeros in the geometric mean. The plan itself says these data "do not contradict the stage order and do not prove it".
- **Set by:** Design (a measured finding of Stage 0, HISTORY Entries 10, 11 and 15).
- **Where:** DESIGN.md section 5 (lines 556-586); HISTORY Entry 11 (lines 692-698) and Entry 15 (lines 923-934).
- **Switch:** on the HiGHS side the names of `HIGHS_DIAG_OFF`; the engine has no switch for the families it has not built.
- **Alternatives:** None. The same question on the 240 gave another ranking (probing 0.6% of the nonzeros in the mean, HISTORY Entry 28).
- **May change:** The first stage order rested on these sums.
- **Test at the end:** Not a quality test.
- **State:** measurement only; superseded for the order by the measurement on the 240.

### AS-247 Finding: what each rule family of the engine is worth in nonzeros
- **Assumption:** With one rule family of the engine off at a time, the nonzeros left against all families on, in the geometric mean over the 40: row rules 1.134, parallel rows 1.128, dual fixing 1.077, doubleton equations 1.054, bound propagation 1.025, row tightening 1.006. Parallel rows cannot be dropped: usafa keeps 5.4 times more nonzeros without them.
- **Set by:** Design (a measured finding, HISTORY Entry 16).
- **Where:** HISTORY Entry 16 (lines 1018-1023); gpu_presolve/tools/stage1_ablation.sh; gpu_presolve/analysis/stage1_ablation.py.
- **Switch:** `options_t::enabled` (one flag per family); tool `--off <family>`.
- **Alternatives:** None.
- **May change:** The worth of a family for the formulation is not what its nonzero number says (AS-043).
- **Test at the end:** The per-family pairs of AS-043.
- **State:** measurement only.

### AS-248 Finding: cuOpt's four own reductions look almost worthless by nonzero count
- **Assumption:** After the merged Stage 1 engine, cuOpt's four reductions change 39 of the 240 and move the geometric mean of remaining nonzeros on 237 of the 240 from 0.7885 to 0.7862. Gains: enlight_hard is solved (GF2), neos-787933 is halved, var-smallemery-m6j6 loses 13%, the two cryptanalysis instances 2%. The mean nets these gains against the rows that indicator strengthening appends, so it understates both. Their worth for the solve was not measured.
- **Set by:** Design (a measured finding, HISTORY Entries 27 and 37, the presolve-only report inside cuOpt).
- **Where:** HISTORY Entry 27 (lines 1675-1677) and Entry 37 (lines 2127-2133).
- **Switch:** `presolve_gprs` 3 against 2.
- **Alternatives:** None.
- **May change:** It supports keeping the four in the comparison arm and porting them.
- **Test at the end:** The pair 3 against 2 of AS-134, and mode 3 with one name removed from its list at a time (AS-136).
- **State:** measurement only.

### AS-249 Finding: the engine is 6.9 (the 40) and 8.4 (the 240) times faster than HiGHS
- **Assumption:** Timed in turn on one machine (viking-prod-215, minimum of three passes): on the 40, HiGHS with the Stage 1 rule set 569.0 s against the merged Stage 1 engine (the engine before the groundwork) 82.5 s; on the 240, 143.0 s against 17.1 s. One instance is slower than HiGHS, neos-1122047 (0.092 s against 0.081 s, inside its allowance of 0.118 s). The node has changed since (viking-cr-194), so these times compare with no later one.
- **Set by:** Design (a measured finding, HISTORY Entries 34, 36 and 39).
- **Where:** HISTORY Entry 34 (lines 2006-2010), Entry 36 (lines 2107-2113), Entry 42 (lines 2310-2311).
- **Switch:** no switch of its own: a measurement of the timing scripts gpu_presolve/baseline/stage1_timing_runs.sh and bench240_timing_runs.sh.
- **Alternatives:** None.
- **May change:** It is the reason the time rule is called slack and the overheads are left alone. The column rules will be compared with a slower reference (HiGHS with all cheap rules: 302 s on the 240).
- **Test at the end:** Not a quality test; to be timed again on one node after each stage.
- **State:** measurement only.

### AS-250 Finding: HiGHS's slow row deletion on rd-rplusc-21 was a hypothesis that a later run weakened
- **Assumption:** The plan requires that deleting rows or columns costs time proportional to what is deleted, because HiGHS needs at least 69 s on rd-rplusc-21 in the call that deletes 71,300 duplicate rows; the cause (walks over long columns) is marked a hypothesis. A later run with the switches of decision 1 takes 0.61 s instead of 126 s on that instance with the same rows removed, so the slow part belongs to the rule families those switches turn off. The plan text was not corrected.
- **Set by:** Design (a measured finding of Stage 0; the cause is a hypothesis of the assistant, corrected in the Stage 0 report, section 11).
- **Where:** DESIGN.md section 2.3 (lines 367-369) and section 1 (lines 268-269); STAGE0_REPORT.md lines 552-558.
- **Switch:** no switch of its own: a finding about HiGHS; the run that weakened it used the HiGHS switch set of the Stage 1 reference (`HIGHS_DIAG_OFF`, AS-185).
- **Alternatives:** None.
- **May change:** It shaped the storage design (tombstones, deferred rebuild).
- **Test at the end:** Not a quality test.
- **State:** measurement only; partly refuted.

### AS-251 Finding: what cuOpt's probing and clique table do today (read from source on 2026-11-26)
- **Assumption:** cuOpt's probing runs one variable per OpenMP task with dense state, its budget allows about 375 variables at 1e7 nonzeros and it does not use cliques; the load-balanced propagation was dropped from the build in September 2025; the clique table is host-only, built inside branch and bound, extends only set-packing rows below 128 entries once each, removes no dominated cliques and is not built in deterministic mode.
- **Set by:** Design (a finding read from cuOpt's source by the assistant on 2026-11-26; not verified again for this register).
- **Where:** DESIGN.md section 7 (lines 606-614); HISTORY Entry 4.
- **Switch:** no switch of its own: a reading of cuOpt's source; cuOpt's probing cache is switched by `mip_probing` (AS-141).
- **Alternatives:** None.
- **May change:** It is the baseline that the new probing and cliques replace; cuOpt has changed since, so the description may be out of date at the end test.
- **Test at the end:** Not a quality test: read the source again before the probing stage.
- **State:** measurement only.

### AS-252 Belief: every row-dual bound HiGHS uses is a function of the present state
- **Assumption:** The design of the five rules rests on the reading that HiGHS's original row-dual bounds are only 0 or infinite, so every implied bound those rules read is cost over coefficient of one non-integer column with at most one lock in a direction. That is why no dual state is kept (AS-301). It was read in the HiGHS source by three designers, two critics and the repair agent, not run.
- **Set by:** Design (a belief from reading the HiGHS source, HISTORY Entry 41, lines 2279-2283; lowworth_v2_design.txt:68, :238).
- **Where:** stage1c_design/lowworth_v2_design.txt:68, :238; HISTORY Entry 41.
- **Switch:** no switch in the engine: no dual state is kept (AS-301); on the HiGHS side `implied_row_dual_bounds` in `HIGHS_DIAG_OFF` freezes its bounds.
- **Alternatives:** If HiGHS derives a dual bound in another way somewhere, the lock form would miss reductions that HiGHS makes.
- **May change:** If wrong, the engine would stay behind HiGHS on some instances after the milestone of the dominated column [L5] without the design predicting it.
- **Test at the end:** Not a quality test: the per-instance comparison with HiGHS's counters on the 240 and the 40 when the dominated column [L5] is built.
- **State:** measurement only.

### AS-253 Belief: the exchange rules are valid; evidence is exact tests of tiny models and reading
- **Assumption:** The proofs of the inequality-pivot singleton, dominance, dual bound strengthening and the single-lock alias were derived on paper; the review enumerated 90,000 tiny all-integer models in exact arithmetic without a failure. For the five further rules, 105,366 tiny models in rational arithmetic gave no changed outcome in 32,067 applications of the implied equation, 34,110 of the dominated column and 2,335 of the single equation; stuffing was tested exactly on 60,000 random rows; the weak test and the scaling rest on reading. None of these tests has tolerances, crush maps, stored activities or the interplay of the rules, and nothing was run in the engine.
- **Set by:** Design (a belief of the two rule designs and their validity reviews; HISTORY Entry 43).
- **Where:** stage1c_design/v2_design.txt:263; v2_reviews.txt:9-11; lowworth_v2_design.txt:10, :221; lowworth_v2_reviews.txt:9, 50-51; HISTORY Entry 43 (lines 2374-2379).
- **Switch:** no switch of its own: evidence, not a setting; each rule it covers has its switch (`options_t::sub_enabled` for the nine, the designed switches of AS-011 for the five).
- **Alternatives:** None before the code exists; the campaign of each milestone is the first test of the engine's form.
- **May change:** An error in the engine's floating-point form of a rule would show as a cut-off optimum or an infeasible postsolved point.
- **Test at the end:** Not a quality test: the campaign of each milestone and the checks on the 240. On the cluster, an arm that ends above the best known objective on an instance the other arm solves is a signal to look at.
- **State:** measurement only.

### AS-254 Belief: a sequential engine reaches more than pure batch application
- **Assumption:** The design reviews established that applying reductions only in batches ends below a sequential presolve on cascades and ordered aggregation, and that PaPILO's substitution is batch-and-reject. The host engine therefore applies one reduction at a time.
- **Set by:** Design (a belief from the design panel, HISTORY Entry 1; not measured on this engine).
- **Where:** HISTORY Entry 1 (lines 58-61, 74); DESIGN.md section 3 (lines 485-491).
- **Switch:** no switch yet: needs one only if a batch engine is built (AS-310); the host engine is sequential.
- **Alternatives:** A batch-only GPU engine.
- **May change:** The number of reductions reached per round.
- **Test at the end:** It would need a batch engine to compare with; none is planned (AS-310 for GPU batches).
- **State:** in the engine (the host engine is sequential).

### AS-255 Finding: HiGHS's own result depends on where it was built
- **Assumption:** The same HiGHS source built on the Mac gives the node's sizes on 239 of the 240; on fiball it ends with 32,397 columns instead of 32,399. References are taken from the node runs, and local previews are not the gate.
- **Set by:** Design (a finding of the agent that split the dual-fixing switch, HISTORY Entry 42).
- **Where:** gpu_presolve/baseline/HIGHS_DIAG.md:1070-1077.
- **Switch:** no switch of its own: a finding about builds; the node build made by gpu_presolve/baseline/build_highs_diag.sh is the reference.
- **Alternatives:** None; the node build is the reference.
- **May change:** A reference number can differ by a few entries between machines.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-256 Finding: a HiGHS counter names the test that took a fixing, not the rule needed
- **Assumption:** HiGHS credits millions of fixings on neos-4332810-sesia, dlr2 and neos-3402294-bobin to its weakly dominated column test, but all three are reached by the singleton column rules followed by the lock rule. Only a run with a rule switched off decides whether the rule is needed.
- **Set by:** Design (a finding, the lesson of HISTORY Entry 41).
- **Where:** HISTORY Entry 41 (lines 2252-2260, 2296-2298); gpu_presolve/baseline/HIGHS_DIAG.md:943-947.
- **Switch:** no switch of its own: a rule for reading HiGHS's counters; what decides is a run with the rule switched off (`HIGHS_DIAG_OFF`, `--presolve-rule-off`), later the engine's own switches.
- **Alternatives:** A HiGHS switch for each outcome; the engine's own switches once the rules exist.
- **May change:** How counters are read when the engine's findings are compared with HiGHS's (the comparison script pairs counters that are not the same rule), and the worth figures that were split by counters.
- **Test at the end:** Not a quality test: size runs with the engine's switches after each milestone.
- **State:** measurement only.

### AS-257 Inferences about reach that no engine run has confirmed, and unknowns left open
- **Assumption:** Three chains are inferred from counters, sizes and source reading: supportcase42 is reached by a weak-test conversion, then dual bound strengthening on the converted column, then redundant rows, then the free singleton rule or the lock rule; the general form of the dominated-column rule covers b1c1s1; the engine's own integrality proof avoids the 173 rows that the conversion costs HiGHS on physiciansched3-3 when implied equations are off. Still unknown: the source types of the implied equations on physiciansched3-3 and physiciansched6-2, and what the engine reaches on instances whose reductions depend on the running state (the map10 pair, rocII-5-11, snp-02-004-104, seymour1, neos-5093327-huahum, sp97ar, the last 30 rows of neos-3656078-kumeu). None is credited in a gate before it is measured.
- **Set by:** Design (inferences of the five-rule design, partly supported by the worth verifier on input models).
- **Where:** stage1c_design/lowworth_v2_design.txt:35, :186, :192, :197, :222; lowworth_v2_reviews.txt:71, 106-107; lowworth_v2_amendments.txt:42-43.
- **Switch:** the designed `integer_sources` (not in code) for the unknown source types; the chains themselves have no switch: they are inferences to be measured.
- **Alternatives:** None.
- **May change:** If the supportcase42 chain breaks at a link, the conversion's only clear size instance is lost and the rule is a pure change of the formulation; the source-type unknown decides how much the switch `integer_sources` matters.
- **Test at the end:** On the named instances: the reference runs before the first of the five rules [L0], then the engine runs after the implied-integer conversion [L2] and after the dominated column [L5].
- **State:** measurement only (inference).

## 8. Known defects and blind spots accepted for now

### AS-258 Three numerical defects of the base engine on models with extreme constants are open
- **Assumption:** Reviews with planted models found that the base engine gives wrong results on models with coefficient ratios of 1e9 inside a row or values of 1e15 and more (24 and 101 of 300,000 planted models). Three defects were named and not repaired: a merge of parallel rows of different scale next to a fixed term of 1e7 and more returns "infeasible" for a feasible model; integer row tightening rounds a side of 1.9e9 times a scale of 125 the wrong way; a doubleton substitution leaves a coefficient of about 1e-9 that the row rules take literally. The current sources hold guards that look aimed at the first two; that all three are repaired could not be confirmed, and no guard for the third was found. None was seen on the 240 or the 44 testbed and status instances.
- **Set by:** Code (found by reviewers, HISTORY Entries 22 and 23; listed as open in Entry 39).
- **Where:** HISTORY Entry 22 (lines 1342-1347), Entry 23 (lines 1422-1426), Entry 39 (lines 2189-2192); gpu_presolve/src/rules_parallel_rows.cpp:423-446; gpu_presolve/src/rules_tightening.cpp:36-38.
- **Switch:** no switch yet: needs one only if a size guard is built.
- **Alternatives:** Repair before any cluster run; one size rule in the state that leaves such rows alone.
- **May change:** A wrong status or a cut-off optimum on badly scaled models (constants of 1e9 and more).
- **Test at the end:** Not a quality pair: a campaign of planted models with extreme constants after the repair; a list of the 240 instances with such constants checked by hand; in every cluster arm, wrong statuses and infeasible postsolved solutions.
- **State:** in the engine.

### AS-259 Two oracle failures of 1.1e-9 are untriaged: dual fixing at an outward-moved bound
- **Assumption:** At the base commit the campaign with added singleton columns fails on two models (seed 32077278, profile huge_parallel; seed 5302642968, profile open_ends; a rate of about 1 in 67 million): the reduced optimum is 1.1e-9 below the true one and the oracle allows 1e-9. Dual fixing fixes a column at an explicit bound that bound propagation had moved outward by the error bound of an activity with a huge term (in the second model a continuous column with a lower bound of -5e21 and a term just under 1e20). Whether the oracle's tolerance or the outward rounding is at fault is not decided. The oracle also compares objectives relative to the value, so costs scaled by 2^30 give false alarms.
- **Set by:** Code (found by the groundwork implementer and its reviewer; HISTORY Entries 44 and 45: "an entry for the register").
- **Where:** HISTORY Entry 44 (lines 2419-2423) and Entry 45 ("Also in this interaction"); gpu_presolve/tests/oracle.hpp:59; gpu_presolve/src/implied.hpp:23-30, 163-181.
- **Switch:** no switch yet: needs one if the outward move is changed (AS-078); the campaign flag is `--singletons`.
- **Alternatives:** Widen the oracle's tolerance; make dual fixing fix at the bound before it was moved outward.
- **May change:** An objective value off by about 1e-9 relative on models with huge terms; far below cuOpt's tolerances.
- **Test at the end:** Not a quality test: triage of the two seeds, then the campaign.
- **State:** in the engine (base commit).

### AS-260 A column can be fixed at any finite bound, however large
- **Assumption:** Dual fixing fixes a column at its target bound whenever that bound is not an IEEE infinity; there is no limit on its size. The constant term that a column fixed at a huge value leaves in its rows swamps the sides, and each family guards itself against that separately (AS-076).
- **Set by:** Code (noted as a lesson in HISTORY Entry 18: "The engine allows fixing at bounds up to 1e20; that deserves one rule in the state rather than a guard in each family").
- **Where:** gpu_presolve/src/rules_dual_fixing.cpp:42-58; the guards are in rules_parallel_rows.cpp:317-320, rules_tightening.cpp:184-189, state_storage.cpp:150-198.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Refuse to fix at a bound above a limit (for example 1e8); treat such a bound as infinite.
- **May change:** Sides that lose precision next to a huge constant; a presolve that cannot move the fixed terms into the sides and so ends without a fixed point; the status "unbounded or infeasible" on a model with a huge finite bound.
- **Test at the end:** Not a quality pair first: the campaign with huge bounds and a structure check on the instances of the 240 with bounds above 1e8. A cluster pair only after a limit exists as a switch.
- **State:** in the engine.

### AS-261 The error bound of an activity is a first-order estimate, not a proven bound
- **Assumption:** Each stored row activity carries the error bound updates * 2^-106 * (largest term the sum ever held), and every guard compares against it. The review measured errors up to 38 times that formula on rows of same-sign terms, and the engine's own consistency check uses 9 * (n + 1)^3 times that amount because valid states exceed the estimate. The rule families sum a row from scratch before a test that the error could decide.
- **Set by:** Design (DESIGN.md section 2.4); left open by the state review (HISTORY Entries 14 and 15: "Open, it sits in the plan").
- **Where:** gpu_presolve/include/gprs/numerics.hpp:121-126; gpu_presolve/include/gprs/state.hpp:333-340; gpu_presolve/src/state_storage.cpp:682-696; HISTORY Entry 15 (lines 943-946).
- **Switch:** no switch yet: needs one; `options_t::tol.activity_error_share` sets when a row is summed again.
- **Alternatives:** The proven bound in the guards (sum of absolute terms); a fresh sum before every test; plain double sums with a relative tolerance.
- **May change:** Only models with huge terms or long rows of same-sign terms with many updates: a test can be decided by an activity whose true error is above the estimate.
- **Test at the end:** Not a quality test: a campaign with long rows of same-sign terms, many updates and huge terms.
- **State:** in the engine.

### AS-262 Doubleton equations are substituted without a numerical acceptance test or a fill limit
- **Assumption:** Every equation with two free columns is used to replace one column by the other when the scale is finite and not zero and the offset is finite. There is no limit on the size of the scale, on the number of entries of the substituted column, or on the growth of coefficients in the rows that receive the kept column.
- **Set by:** Code (left open by the review, HISTORY Entry 14: "aliases have no numeric acceptance test").
- **Where:** gpu_presolve/src/alias_forest.cpp:27-30, 39-64; gpu_presolve/src/rules_doubleton.cpp:110-156.
- **Switch:** no switch yet: needs one for a limit; `options_t::enabled[rule::doubleton]` turns the family.
- **Alternatives:** A limit on the absolute scale or on the coefficient ratio of the changed rows (the subtraction of parallel rows vetoes above 1e6); a limit on the entries of the substituted column.
- **May change:** Rows with coefficients of very different size; the postsolve multiplies a deviation of the kept column by the scale, so a postsolved point can violate bounds or rows of the original model.
- **Test at the end:** Standard cluster comparison: doubleton equations on against off; with a limit, a pair with a scale limit; the count of postsolved solutions that fail the feasibility check in each arm.
- **State:** in the engine.

### AS-263 The alias forest accepts an integer map without a drift guard
- **Assumption:** An integer column is replaced only by an integer column, when scale and offset are within 1e-9 of integers and the rounded map satisfies the equation in its own units; what the rounding does over the range of the kept column is not bounded. With a column of a huge range the removed equation can be violated by more than the tolerance after postsolve (the design's model: x - 1.0000000005 y = 0 with y integer up to 2,000,000). The repair, the shared integrality test (AS-093), is scheduled with the milestone of the singleton-column rules and the forcing column [M1], not before.
- **Set by:** Code (found by the design round of the five rules, HISTORY Entry 43; v2_amendments.txt:92-121).
- **Where:** gpu_presolve/src/alias_forest.cpp:38-64; stage1c_design/v2_amendments.txt:97-102, 120-121; lowworth_v2_design.txt:15, :224.
- **Switch:** no switch yet: needs one only if the guard is to be tested.
- **Alternatives:** Repair it now in the Stage 1 code.
- **May change:** A row violated by more than the tolerance at some reduced points on models with a coefficient that is integral only to 1e-9 and a large column range.
- **Test at the end:** Not a quality pair: a fixed oracle model and a wrong engine; on the cluster it would show only as an infeasible postsolved solution.
- **State:** in the engine; the repair is designed (milestone of the singleton-column rules and the forcing column [M1]).

### AS-264 Parallel rows are found through a hash on 20 mantissa bits, blind at rounding boundaries
- **Assumption:** Rows are grouped by a hash of their coefficients, normalised and rounded to 20 mantissa bits. Two rows that are parallel within 1e-9 but have a coefficient on a rounding boundary of the hash fall into different groups and are not compared. Of two parallel rows the one with the smaller index stays and takes the intersection of the sides.
- **Set by:** Design (the blind spot is stated in the rule's contract).
- **Where:** gpu_presolve/src/rules_parallel_rows.cpp:29-32, 42-71, 793-806; gpu_presolve/include/gprs/rules.hpp:267-269, 285-289.
- **Switch:** no switch yet: needs one.
- **Alternatives:** A second hash with shifted rounding; an exact canonical form.
- **May change:** A few parallel pairs stay in the model.
- **Test at the end:** Not a quality test: count the missed pairs on the 240 against an exhaustive search.
- **State:** in the engine.

### AS-265 A free singleton chain's growth factor does not follow a column through an alias
- **Assumption:** When a saved column of a free singleton record is later replaced by an alias, the growth factor is not carried on, so the bound of 1000 (AS-094) can be exceeded through that alias.
- **Set by:** Design (stated as a risk and confirmed by the validity review).
- **Where:** stage1c_design/v2_design.txt:269; v2_answers.txt:26; v2_reviews.txt:64.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Carry the factor through aliases.
- **May change:** Rare larger bound violations in postsolved solutions.
- **Test at the end:** Not a quality pair: the count of postsolved points outside the tolerance on the 240.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]).

### AS-266 The zero-cost singleton rule turns equations into ranged rows
- **Assumption:** After the zero-cost singleton rule an equation with a slack-like column is a ranged row, which the doubleton rule and the later aggregator cannot use as a pivot row. It is judged by the cheap-rules gate only, and is to be looked at again when sparsify and the aggregator exist.
- **Set by:** Design (stated as a risk, v2_design.txt:271).
- **Where:** stage1c_design/v2_design.txt:271.
- **Switch:** `options_t::sub_enabled[sub_rule::zero_cost_singleton]`.
- **Alternatives:** Apply the rule after the aggregator; keep equations.
- **May change:** Fewer substitutions in later stages; a different LP form (ranged rows instead of equations with slacks).
- **Test at the end:** Standard cluster comparison: `zero_cost_singleton` on against off, repeated when the aggregator stage exists.
- **State:** designed (milestone of the singleton-column rules and the forcing column [M1]).

### AS-267 No reduction looks at what it does to the integrality of the objective
- **Assumption:** cuOpt takes the objective as integral when every nonzero cost is an integer within 1e-9 and sits on an integer or implied-integer column, and it computes an objective step from the costs and the rows; branch and bound uses both to round its bound and to set its cutoff. Engine reductions move cost: a doubleton substitution moves it to the kept column times a ratio, a free singleton substitution spreads it over up to row length minus 1 columns, a merge of a mixed pair keeps the continuous column, the column scaling multiplies it. No design checks whether the objective of the reduced model is still integral.
- **Set by:** Design (an omission; not noted in DESIGN.md, HISTORY or the two rule designs).
- **Where:** cpp/src/mip_heuristics/problem/problem.cu:1282-1296 and :1335-1410; cpp/src/mip_heuristics/solver.cu:340-360; cpp/src/branch_and_bound/branch_and_bound.cpp:500-541; gpu_presolve/src/rules_doubleton.cpp:144-152; stage1c_design/v2_design.txt:37 and :83; lowworth_v2_design.txt:37-42.
- **Switch:** no switch yet: needs one (a guard per rule, or one option that keeps an integral objective). On cuOpt's side `mip_objective_step` (default 1) switches the use of the step.
- **Alternatives:** Refuse a substitution, merge or scaling that makes an integral objective non-integral; accept the loss; export what the engine knows about the step.
- **May change:** Whether cuOpt rounds node bounds and sets its cutoff a step below the incumbent: optimal count and final gap on models with an integral objective, with no trace in any size. It adds to AS-164, which is the same effect through the missing implied-integer flags.
- **Test at the end:** In every pair count per arm the instances with the log lines "Objective function is integral" and "Objective step size" (cpp/src/mip_heuristics/solver.cu:347 and :353); where the counts differ, read optimal count and final bound on those instances apart. With a guard built: standard cluster comparison, guard on against off.
- **State:** in the engine (doubleton substitution); designed for the free singleton rules [M1], the merge of parallel columns [M5] and the column scaling [L2].

### AS-268 After the singleton rules the parallel rows hash many rows again; that cost is unmeasured
- **Assumption:** Each round expands listed rows to their columns for the dual family, and after the singleton rules many rows change and are hashed again by the parallel rows. In HiGHS that stage takes 80 s on rd-rplusc-21; the engine was not measured on such states.
- **Set by:** Design (stated as a risk, v2_design.txt:268).
- **Where:** stage1c_design/v2_design.txt:268.
- **Switch:** no switch yet: needs one only if the cost proves high (a limit on hashing rows again); `options_t::budget[rule::parallel_rows]` bounds the work of a call.
- **Alternatives:** None stated.
- **May change:** Presolve time on a few instances.
- **Test at the end:** Not a quality test: timing and work per family at the milestones of the singleton-column rules and the forcing column [M1] and of dual bound strengthening and the single-lock alias [M4].
- **State:** designed.

### AS-269 Each column family keeps its own row list; a switched-off family still pays for it
- **Assumption:** The state pushes every row and column event onto the lists of the four new families whether or not a family is switched on. The design's rule was: engine time on the 240 within 3% of the time before, or the three column families share one row list. Measured on the node for the groundwork (minimum of three alternating passes, one process at a time, node load 12 to 40): +1.07% on the 240 (16.50 s to 16.68 s), +2.27% on the 31 and +1.91% on the 9 (dlr2 +5.1%, neos-4332810-sesia +3.2%); with the four families off +0.50%, +2.46% and +1.68%. By the criterion on the 240 each family keeps its own list, although the 31 and single large instances pay more. Two wrong engines (an event that does not feed the new lists) are caught by unit tests only until a rule reads the lists. The first rule milestone was started before the node had confirmed the groundwork.
- **Set by:** Taken as recommended (decided by the assistant "by the design's own criterion", HISTORY Entry 46; the criterion names the 240 only).
- **Where:** HISTORY Entry 46 (lines 2537-2543), Entry 44 (lines 2413-2417) and Entry 45; stage1c_design/v2_design.txt:107, :183, :233; v2_amendments.txt:78-79; gpu_presolve/include/gprs/state.hpp:159-243; gpu_presolve/analysis/stage1c_time.py.
- **Switch:** no switch yet: needs one (events masked by the enabled families, or the shared row list); `options_t::enabled` switches the four families but not their lists.
- **Alternatives:** One row list shared by the three column families (the design's fallback); events masked by the enabled families.
- **May change:** Presolve time by 1 to 2% in the sum and up to 5% on single large instances. The results are identical: status, sizes, hash, fixed-point flag, rounds and counts equal the engine before on the 240, the 31 and the 9. A missing event stays unseen until a rule reads the lists.
- **Test at the end:** Not a quality test: the same three alternating passes when the first rules read the lists, on the 240 and on the 9.
- **State:** in the engine (merged into feat/gpu-presolve as `6eed652cf` and `74350acb3`).

### AS-270 Implementation starts from designs with known open points
- **Assumption:** The nine-rule design (version 2) has sixteen open points, called minor, and the design of the five further rules (version 2) has fourteen, five of them serious (six from the validity review, eight from the worth, cost and state review). Each has a model and a fix and is binding for the milestone it belongs to; none is repaired in the design text itself. The verifiers did not read everything: the validity verifier of the five rules did not read the structure check, the oracle source, the plan or the result files, and the worth verifier's work counts are static counts on input models.
- **Set by:** Taken as recommended (accepted by the assistant on the word of two verifiers per design, HISTORY Entries 40 and 43).
- **Where:** HISTORY Entry 40 (lines 2200-2204) and Entry 43 (lines 2358-2361); stage1c_design/v2_amendments.txt:46-82 and 92-121; lowworth_v2_amendments.txt:10-23, 25-44; lowworth_v2_reviews.txt:7-40, 59-112.
- **Switch:** no switch of its own: each open point is binding for its milestone and has a fixed test or a wrong engine named with it (stage1c_design/v2_amendments.txt, lowworth_v2_amendments.txt).
- **Alternatives:** A third design version before any code.
- **May change:** If an amendment is missed in a milestone, the rule is built with the hole the review found: a conversion that cuts feasible points for coefficients below 0.1, an implied equation with a term above about 1e9 that loses an optimum (3332.33 becomes 4332.33 in the verifier's model), a wrong fixed-point flag for the weak test, a stalled dual column family on supportcase40.
- **Test at the end:** Not a quality test: each amendment has a fixed test or a wrong engine named with it, checked by the milestone's review.
- **State:** designed (binding per milestone).

### AS-271 The presolve can end without a fixed point, and the reason is not reported
- **Assumption:** The engine sets a flag when a second presolve of its result would change nothing; the reduced model is returned in either case. The flag is missing on 6 of the 240 (icir97_tension, comp21-2idx, neos-1122047, co-100, neos-5052403-cygnet, neos-2075418-temuka) and on 3 of the 40 (neos-5052403-cygnet, a2864-99blp, dlr2), the result does not say which limit caused it, and no gate asks for it. Even with the flag set, a second presolve can decide a test otherwise where the last bits of a stored activity decide it. For the column rules the flag is argued from the event sites by reading; four limits clear it on purpose (bucket limit, absorber limit, saved-row limit, growth refusal).
- **Set by:** Design (stated in the contract of `presolve()`); listed as open in HISTORY Entry 39 ("the reason of a missing fixed-point flag is not reported (nine instances)").
- **Where:** gpu_presolve/include/gprs/presolve.hpp:227-235, 305-325; gpu_presolve/src/presolve.cpp:437-438; gpu_presolve/results/bench240/v4pre/release_all/summary.json and gpu_presolve/results/stage1/host/v4 (field `fixed_point`); stage1c_design/v2_design.txt:127, :266; lowworth_v2_design.txt:93-99; lowworth_v2_amendments.txt:26-28.
- **Switch:** no switch yet: needs one for a report of the cause; the limits behind it are `options_t::max_rounds`, `max_cascade_passes`, the budgets and `parallel_bucket_limit`.
- **Alternatives:** Report the cause per instance; require the flag in a gate; run the presolve on its own result until it returns unchanged.
- **May change:** On these instances reductions are left that a second presolve would find; a missed event site loses reductions and makes the flag wrong.
- **Test at the end:** First a size run: presolve the reduced model again on the 240 and the 40 and compare the hashes. If every second presolve returns its input unchanged, the entry is closed. Any difference sends it to the standard cluster comparison, read on the instances that differ: one presolve against the presolve applied twice, or larger limits against the defaults.
- **State:** in the engine.

### AS-272 The limit on row dynamism is stated but not measured for the engine
- **Assumption:** The plan says the 99th percentile of the ratio of largest to smallest coefficient per row of our reduced model may be at most 10 times that of HiGHS's reduced model. The comparison script writes the reference's dynamism to its table, but nothing computes the engine's, so the limit is not checked. On rd-rplusc-21 the first engine had 35 times HiGHS's row dynamism.
- **Set by:** Design (DESIGN.md section 12: "Row dynamism is a constraint, not a goal"); the missing measurement is code.
- **Where:** DESIGN.md section 12 (line 766); gpu_presolve/analysis/stage1_compare.py:65-66, 119-135, 271-272; HISTORY Entry 17 (item 4).
- **Switch:** no switch yet: needs one (single rules have their own limits, for example `max_dynamism` = 1e6 in gpu_presolve/src/rules_parallel_rows.cpp:375).
- **Alternatives:** Write the row dynamism of the reduced model in the engine tool's report and gate on it; a tighter factor; an absolute limit.
- **May change:** The numerical quality of the model handed to cuOpt's LP and heuristics, more so with scaling off.
- **Test at the end:** Report the 99th percentile per instance for both sides on the 240 with the final engine, and the dynamism of both arms next to the quality figures.
- **State:** planned (not enforced in the scripts).

### AS-273 Instances with indicator constraints have no standalone baseline
- **Assumption:** supportcase21i, gfd-schedulen55f2d50m30k3i and elitserienhandball3i left the testbed lists because HiGHS cannot read an INDICATORS section. The standalone measurement covers 84 of the 85 structure features, all but "indicator", and no size or correctness measurement of the engine exists on models with indicator constraints. Whether any of the 240 has an INDICATORS section was not checked. cuOpt's indicator strengthening is a different matter: it acts on ordinary rows between binaries (AS-003). A search of the repository's C++ sources finds no reader of an INDICATORS section.
- **Set by:** Taken as recommended (forced by the baseline, HISTORY Entry 9, 2026-11-26; the owner was told and did not object).
- **Where:** DESIGN.md section 14 (lines 868-872, 925); gpu_presolve/testbed/testbed.json (lists `indicator` and `indicator_note`).
- **Switch:** no switch yet: needs one (the three instances are kept in the `indicator` list for tests inside cuOpt).
- **Alternatives:** Convert indicator constraints to big-M rows for HiGHS; read the model with cuOpt's reader; PaPILO inside cuOpt as the baseline on them.
- **May change:** Nothing in the engine. Models with indicator constraints are outside every standalone gate and check, and how cuOpt itself takes these three instances was not established.
- **Test at the end:** Not a quality pair: first establish whether cuOpt reads the three instances at all; if it does, add them to the cluster set of the main comparison and report them apart.
- **State:** measurement only.

### AS-274 The held-out set is no longer unseen: 23 of its 36 are among the 240
- **Assumption:** The plan keeps 36 held-out instances, drawn by the same rule with the main set's families excluded, that are not to be looked at before the completion stage. 23 of them are among the 240 that every decision now uses (for example nw04, rail02, neos-3656078-kumeu, supportcase10), so only 13 are still unseen. Nobody decided what replaces it.
- **Set by:** Design (DESIGN.md section 14); weakened when the owner asked for all decisions on the 240 (HISTORY Entry 36). The overlap was recomputed from the two instance lists.
- **Where:** DESIGN.md section 14 (lines 924-931) and section 13 (line 800); gpu_presolve/testbed/testbed.json (list `heldout`, 36 entries) against gpu_presolve/testbed/benchmark240.json; gpu_presolve/baseline/run_baseline.py:100-104.
- **Switch:** no switch of its own: the list `heldout` of gpu_presolve/testbed/testbed.json against gpu_presolve/testbed/benchmark240.json; a new held-out list has to be drawn.
- **Alternatives:** A new held-out set drawn outside the 240; the 13 unseen instances as the held-out set; accept the 240 as the tuning set and use another collection for the final check.
- **May change:** Nothing in the model; it removes the protection against tuning rules and tolerances to the instances they are judged on. The same holds for the end test: every threshold was chosen on the 240 it runs on.
- **Test at the end:** Another kind of test: a final comparison, presolve-only and on the cluster, on instances that were never used.
- **State:** planned.

### AS-275 The checks cannot see a restriction that cuts no tested point
- **Assumption:** A reduction that wrongly cuts off feasible points is seen only if it cuts a point that is tested or makes HiGHS prove a worse optimum. Known wrong engines that passed every check: an implied bound made 1e-4 too tight (uccase9, triptim1, kenana), propagation over a row with an infinite term (uccase9), and any restriction on the infeasible instances. A support test for each tightened bound, named as the remedy, is not built; pushed points are a partial remedy.
- **Set by:** Design (found by the attack with wrong engines, HISTORY Entry 20, "Still invisible"; accepted for now).
- **Where:** gpu_presolve/check/crush_check.hpp:51-60; gpu_presolve/tools/gprs_fuzz.cpp:26-33; HISTORY Entry 20 (lines 1212-1216).
- **Switch:** `--push-points` (at most 4 per point) and `--cost-variants` of gpu_presolve/check/run_check.py widen what is tested; no switch closes the gap: needs a support test per tightened bound.
- **Alternatives:** A support test per tightened bound; many more points per instance; exact solves.
- **May change:** A slightly too tight bound can cut the optimum on a real instance while every check passes.
- **Test at the end:** The end test is the strongest detector available: an arm that loses optimal solutions or ends with a worse primal gap on instances the other arm solves points to a restriction; look at instances that close the gap at an objective worse than the best known value.
- **State:** measurement only.

### AS-276 Instances without a usable point get no crush check
- **Assumption:** The crush check needs a feasible point that meets the original model within 1e-7. On the 240, 56 instances have no usable point in the newest check (the groundwork engine, tag s1c_m0: 184 instances with 2,593 points, recounted from the stored summary; with HiGHS points of 60 s alone it was 173 instances with 2,153 points, HISTORY Entry 27); on those the evidence is the presolve status and the structure check alone. On the 40, all 31 below 1e7 nonzeros have points since the published MIPLIB points were converted; of the nine large instances hawaiiv10-130 has none (its two published solutions violate rows by up to 5.3e-5 on the model HiGHS reads), so the check of the large set fails on that one item in every node chain.
- **Set by:** Design (accepted for now, HISTORY Entries 30 and 39, listed as open; `--require-points` makes the missing check a visible failure).
- **Where:** gpu_presolve/check/run_check.py:26-28, :67, 354-356; HISTORY Entry 27 (lines 1653-1655), Entry 30, Entry 39 (lines 2168-2179), Entry 46 (lines 2532-2536); gpu_presolve/results/bench240/s1c_m0/check_all/summary.json (field `crush.points_used`).
- **Switch:** `--require-points` of run_check.py; solution files in every `--solutions-dir`.
- **Alternatives:** Accept published points at MIPLIB's 1e-6 and judge the crushed and postsolved point against the violation it came with; polish the points; longer HiGHS runs; cuOpt's own incumbents as points.
- **May change:** Nothing in the engine; restrictions on about a quarter of the 240 are tested by no point.
- **Test at the end:** The end test covers these instances through cuOpt's own solutions, which can be fed back into the crush check; before it, points from earlier cuOpt runs could be converted.
- **State:** measurement only.

### AS-277 The structure check cannot prove the dual restrictions
- **Assumption:** The structure check verifies the stack, the types and the column map, but cannot prove that a dominance fixing had its absorber bound implied, that a dual bound saw every locking row, that a lock alias had its row released, that an implied equation had its lock single, or that a weak conversion or a fixing by stuffing, single equation or dominated column keeps an optimal point. Only the crush check on feasible points and the oracle test these.
- **Set by:** Design (v2_design.txt:153; lowworth_v2_design.txt:110-115).
- **Where:** stage1c_design/v2_design.txt:153; lowworth_v2_design.txt:110-115.
- **Switch:** no switch yet: needs one only if a dual certificate per record is built; the structure check has no option for it.
- **Alternatives:** A dual certificate per record.
- **May change:** A wrong dual reduction on a real instance is seen only if a known point is cut off or the reduced optimum is proved.
- **Test at the end:** Not testable by solution quality as a rule: the crush check on the 184 benchmark instances with points and the campaign. On the cluster a lost optimum shows as a worse best objective on an instance where the other arm finds the known optimum; compare per instance, not only means.
- **State:** designed.

### AS-278 The exact oracle decides only a small class of models and sees no tolerance
- **Assumption:** The oracle solves tiny random models by enumeration: at most 5 rows, 5 enumerated integer columns of width up to 3 and 2 other columns, integer coefficients up to 3 and powers of two, at most one non-enumerated column per row. It cannot generate rows with several continuous columns or sides above 1e10 (363 models of a 204-million campaign ended outside its class), cannot show a wrong tolerance (tolerance cases rest on a handful of fixed models), is exact for parallel columns only when both are continuous, and tests the singleton rules with a second continuous column in the pivot row, the dominated column with a compensator that has further rows and the single equation with absorbers that have further rows by sampling only. The 31 have free columns on 6 instances only and no finite bound of 1e8 or more, so the oracle is the only test for both.
- **Set by:** Design (the oracle's author; the limits were named by reviewers, HISTORY Entry 22, and by both rule designs). The oracle was built during the six hours of independent work, without a check-in.
- **Where:** gpu_presolve/tests/oracle.hpp:10-47, :59, 74-81, 241-244; gpu_presolve/tools/gprs_fuzz.cpp:459-467; DESIGN.md section 13 (lines 846-859); stage1c_design/v2_design.txt:160-161, :272; lowworth_v2_design.txt:119-132, :230.
- **Switch:** fields of `generator_options_t` (gpu_presolve/tests/oracle.hpp:76-130), set per profile in gpu_presolve/tools/gprs_fuzz.cpp:62-166, for example `p_singleton_col` (0 by default; the campaign flag `--singletons` raises it) and `p_parallel_row` (0.15). A knob `p_parallel_col` for parallel columns is designed only (v2_design.txt:159, :244) and is not in code.
- **Alternatives:** Planted models with a known solution for larger structures; a generator with non-integer data and an exact reference solver; a second exact solver for integer twins; a larger oracle class.
- **May change:** A wrong reduction that needs a structure outside the class, a tolerance error outside the fixed models, or a wrong merge of integer parallel columns (the case of AS-001) would pass the campaign and first show on real instances.
- **Test at the end:** Not a quality test by design; the end test covers real structures, which is why per-instance results of each arm should be looked at, not only means.
- **State:** measurement only.

### AS-279 The checks' proofs cover the rules that exist; a new rule must bring its part
- **Assumption:** The crush check's proof is an induction over the reductions with one sentence per dual rule: a dual rule that is added must bring its sentence and, where it moves a column that stays in the model, a record that the crush replays; the proof has to be reread for a rule that keeps no forward map (a bound on a row dual, a fixing at infinity). The structure check's proof is written for the rules that exist. Until its extension exists, a new rule is unchecked by these two checks; this happened with the row addition on brazil3 and piperout-d27.
- **Set by:** Design (stated in the check headers; the lesson of HISTORY Entry 25).
- **Where:** gpu_presolve/check/crush_check.hpp:21-49; gpu_presolve/check/structure_check.hpp:93-119.
- **Switch:** no switch of its own: each new rule brings its check extension; which checks run is set by the flags of gpu_presolve/check/run_check.py.
- **Alternatives:** None; each new rule brings its check extension and wrong engines.
- **May change:** A wrong reduction of a rule whose extension is missing passes the checks.
- **Test at the end:** Not a quality test: wrong engines for each new rule that the extended checks must catch.
- **State:** designed (per milestone).

### AS-280 Real instances are checked with default settings only
- **Assumption:** The check tool presolves each instance with default options, in default and in primal-only mode, plus cost variants. Settings that are held back or switched are never tested by the checks on real instances: a family or a column reduction off and `merge_integers` other than the default are drawn by the random campaign only, and exact parallel rows (`near_parallel_rows` false) are tested by unit tests only (gpu_presolve/tests/test_parallel_rows.cpp, test_e2e_presolve.cpp:410).
- **Set by:** Code.
- **Where:** gpu_presolve/check/gprs_check.cpp:463-467, 755-766; gpu_presolve/tools/gprs_fuzz.cpp:232-280.
- **Switch:** no switch yet: needs one (run_check.py has only `--primal-only`; run_engine.py has `--tool-arg`).
- **Alternatives:** Pass engine options through run_check.py.
- **May change:** A held-back setting can be wrong on real structures and reach a cluster arm unchecked.
- **Test at the end:** Before any end-test arm with a non-default setting: the structure check and the crush check on the 240 and the 40 with that setting.
- **State:** planned.

### AS-281 The engine inside cuOpt differs from the standalone engine on 2 of the 240, unexplained
- **Assumption:** The new presolve alone inside cuOpt (mode 2) leaves the same sizes as the standalone engine on 238 of the 240; momentum1 and var-smallemery-m6j6 differ by 7 and 2 nonzeros. The cause was not looked into. The log guesses that cuOpt hands over rows with sorted entries, which the standalone dump does not guarantee; the dump is documented as sorted inside each row, and the adapter sorts a row that is not sorted.
- **Set by:** Code (measured and left, HISTORY Entry 37: "Not looked into").
- **Where:** HISTORY Entry 37 (lines 2124-2126); gpu_presolve/baseline/README.md:124-143; cpp/src/presolve/gprs_presolve.cpp:91-103.
- **Switch:** no switch yet: needs one (a dump of the model the adapter builds, AS-179).
- **Alternatives:** Dump the model the adapter builds and run the standalone engine on that; make the engine's result independent of the entry order inside a row.
- **May change:** Standalone gates and checks are made on the dumps, not on the model cuOpt hands over; on these two instances the checked model is not the one cuOpt reduces.
- **Test at the end:** Not a quality test: equality of the input hash and of the reduced hash between adapter and standalone run on the 240; every arm logs the reduced sizes.
- **State:** open.

### AS-282 The adapter's conversions cost several times the engine; tracked, not optimised
- **Assumption:** Inside cuOpt the presolve call took 94 s on the 240 (128 s in an earlier run on a busier node) for work the standalone engine does in 18 s (the engine before the groundwork, one run); the adapter's conversions are a median of 650 ns per nonzero, and bohle, square47 and kasavu carry the most. The two sides were not timed in the same run, so the factor is rough. PaPILO's call pays conversions too (1,127 s in all on the 240).
- **Set by:** The owner (the rule on overheads of HISTORY Entry 33, AS-208); measured in Entries 27, 33 and 37 and listed as open in Entry 39.
- **Where:** HISTORY Entry 33 (lines 1994-1998) and Entry 37 (lines 2134-2135); cpp/src/presolve/gprs_presolve.cpp:53-126, 131-172; cpp/src/presolve/third_party_presolve.cpp:1033, 1059-1061.
- **Switch:** no switch yet: needs one only when the conversions are optimised (AS-208); the seconds are in the log line "GPRS presolve" of cpp/src/presolve/gprs_presolve.cpp.
- **Alternatives:** Build the engine's model directly from cuOpt's host arrays; hand the reduced model over without the intermediate copies.
- **May change:** Seconds before the solver starts (0.4 s per instance in the mean of the 240), so a small part of the primal integral; nothing in the reduced model.
- **Test at the end:** A time measurement, not a quality pair: adapter seconds and engine seconds from the "GPRS presolve" log line in every run; a large adapter cost would show as a worse primal integral on short instances.
- **State:** in the adapter.

### AS-283 No memory figure of the engine is measured or gated
- **Assumption:** The goal and every gate name size, time, determinism and correctness; none names memory. The engine tool records no peak memory (the HiGHS baseline tool records `peak_rss_mb` of its own process), although the state holds the matrix row-wise and column-wise with values and cross positions in both, and inside cuOpt the adapter keeps a second copy of the original model until the solve ends (AS-147).
- **Set by:** Design (by omission; no decision is recorded).
- **Where:** DESIGN.md section 1 (lines 237-238) and section 10 (line 732); gpu_presolve/include/gprs/state.hpp:62-69, 263-264; gpu_presolve/baseline/baseline_tool.cpp:596, :737; gpu_presolve/baseline/run_baseline.py:36, :283; HISTORY Entry 32 (lines 1949-1951, 1959-1962). A search of gpu_presolve/tools, src, include and analysis finds no peak-memory field.
- **Switch:** no switch yet: needs one (a peak-memory field in the engine tool and in the adapter's log line).
- **Alternatives:** Report peak memory next to seconds in every run; a limit in bytes per nonzero as a gate.
- **May change:** Nothing in the reduced model. On the largest instances the host's memory already decides the time of the state build (neos-4535459-waipa: about 7 GB of result arrays in the device state-build measurement on 15 instances), and inside cuOpt the presolve shares memory with the early heuristics, whose start test the plan names and the code lacks (AS-147).
- **Test at the end:** Not a quality test: peak memory of the engine on the 40 and the 240 next to HiGHS's `peak_rss_mb`, and in every cluster arm the count of runs that end on an out-of-memory error.
- **State:** measurement only (open).

### AS-284 Known cuOpt defects can hit both arms; one assertion failure is unexplained
- **Assumption:** Three assertions in cuOpt code fired in the pre-flight. Two were reproduced without the new presolve and are taken as cuOpt's own: an abort ("size mismatch") when the time limit ends during cuOpt's own presolve, where what a release build does is unknown, and a knapsack assertion ("not a cover") that a release build handles by adding no cut. The third (the published objective disagrees with the assignment, once on rmatr200-p5; 0 of 6 repeats with the new presolve, 0 of 8 without) is not explained and is taken as a timing event at the time limit. The fix of the first is in an open upstream pull request that a separate session reviewed; nothing was posted.
- **Set by:** The owner for the handling (HISTORY Entry 26: "if you see a critical bug in current cuopt, generate a PR on a separate session for merging to upstream."); the classification is the assistant's (Entry 23).
- **Where:** HISTORY Entry 23 (lines 1378-1388), Entry 26 (lines 1579-1582), Entry 29 (lines 1785-1787). The three cuOpt files were not opened for this register.
- **Switch:** no switch yet: needs one only in the sense of a merged fix: the defects are in cuOpt's base, the same in both arms, and the fix of the first is an upstream pull request (which one was not read for this register).
- **Alternatives:** Merge the upstream fix into the benchmark branch before the end test.
- **May change:** A run that ends its time limit inside cuOpt's presolve may behave wrongly in a release build, in both arms; if the third comes from the engine after all, a published objective can disagree with its assignment.
- **Test at the end:** Not a quality test: both arms are built from the same cuOpt base; repeated assertion-build runs of rmatr200-p5 with both presolves.
- **State:** in cuOpt (base).

### AS-285 A possible defect in cuOpt's probing cache is noted, not confirmed
- **Assumption:** The plan notes, to be confirmed, that cuOpt's probing cache sets an iteration limit of 50 on one bound-presolve object while the probes run on separate objects built afterwards. If true, cuOpt's probing in both arms runs with another limit than intended.
- **Set by:** Design (found while reading, DESIGN.md section 10, "To confirm"; "Not confirmed by a run", HISTORY Entry 4).
- **Where:** DESIGN.md section 10 (lines 733-734, which cite :919 against :934); in this tree the assignment is at cpp/src/presolve/probing_cache.cu:920 and the probe pool is built from :926 on.
- **Switch:** no switch yet: needs one only after the defect is confirmed (a fix in cuOpt, upstream); cuOpt's probing cache as a whole is switched by `mip_probing`.
- **Alternatives:** Confirm by reading or a unit test, then fix upstream in a separate change (the owner's rule of HISTORY Entry 26).
- **May change:** The depth of propagation in cuOpt's probing in every arm, which shifts the baseline of any probing comparison.
- **Test at the end:** None until confirmed; then cuOpt as is with and without the fix.
- **State:** not confirmed.

### AS-286 Early stored results carry no identity, and some step markers were written by hand
- **Assumption:** The stored results of the first engine versions (v1, v2s, v2p) carry no commit or binary identity, and an early skip rule reused results by file existence; later results carry source and binary checksums. In the full node chain of the merged engine (tag v4) the markers of the two solution steps were written by hand with the new identity instead of rerunning them, on the ground that solution files do not depend on the engine and take 3.5 hours to remake.
- **Set by:** Code (findings of the audit, HISTORY Entry 17, and of Entry 34; accepted by the assistant).
- **Where:** HISTORY Entry 16 (lines 1031-1036), Entry 17 (lines 1069-1073, 1086-1087), Entry 34 (lines 2021-2023); gpu_presolve/results/stage1/host; gpu_presolve/tools/run_info.py.
- **Switch:** no switch of its own: a defect of record keeping; later results carry source and binary checksums written by gpu_presolve/tools/run_info.py.
- **Alternatives:** Rerun the old versions with identity; rerun the solution steps for every tag.
- **May change:** Nothing in the model; old headline numbers cannot be tied to a commit, and the claim "every step passed" for tag v4 rests on solution files made for an earlier tag.
- **Test at the end:** Not a quality test.
- **State:** measurement only.

### AS-287 Statements about PaPILO were not read in its source
- **Assumption:** That PaPILO's singleton-column rule lacks the integrality test of the side, that it skips singleton columns among parallel columns and that it does not merge on pure binary models are statements of the first design or the task text; neither the designers nor the reviewers of the two rule designs read PaPILO. HiGHS's postsolve stack was not read either.
- **Set by:** Design (stated by both designs and their reviews).
- **Where:** stage1c_design/v2_design.txt:8, :35, :63, :273, :277; v2_reviews.txt:11, :79; lowworth_v2_design.txt:8, :235; lowworth_v2_answers.txt:74.
- **Switch:** no switch of its own: a gap of reading; the PaPILO methods cuOpt runs are listed in cpp/src/presolve/third_party_presolve.cpp:725-759 (AS-138).
- **Alternatives:** Read PaPILO's forms of the same rules (it is the presolve to be replaced in cuOpt).
- **May change:** Comparisons with PaPILO inside cuOpt may rest on a wrong picture of what PaPILO does; the designs may miss a form that PaPILO has and HiGHS has not.
- **Test at the end:** The main comparison (mode 0 against mode 3) would show a PaPILO-only form as a difference; it does not isolate it.
- **State:** measurement only.

## 9. Architecture and device

### AS-288 The same model and settings give the same reduced model on every machine
- **Assumption:** Engine thresholds, budgets, slot counts and every cap are constants or functions of the model and the settings; free memory, GPU type and core count are never read, and no result depends on a clock, an address or the order of a hash container. A machine cannot adapt its limits: a phase that does not fit its fixed budgets runs on the host or is skipped everywhere, and a failed allocation is a hard error. The evidence is equal hashes of the reduced model across repetitions, between the release and the assertion build, and between the Mac and the node; a second GPU type and other CPU types were not measured.
- **Set by:** The owner for determinism (HISTORY Entry 2, 2026-11-26: "Make sure the new presolve is deterministic. Use reductions instead of atomics if needed."); taken as recommended for one shipped set of thresholds (recommendation of plan version 3).
- **Where:** DESIGN.md section 0 (line 203), section 2.1 (lines 315-316), section 2.5 (lines 457-475); gpu_presolve/include/gprs/presolve.hpp:101-102; gpu_presolve/tools/run_engine.py:206-220; stage1b_design/final_design.txt:27, :70, :142.
- **Switch:** `options_t::phase_hashes` (default false) records a hash after every phase, and `--reps` of gpu_presolve/tools/run_engine.py fails on unequal hashes; determinism itself has no switch.
- **Alternatives:** Thresholds tuned per machine or read from free memory (results then differ between machines); budgets in seconds; a fallback to the host when memory is short.
- **May change:** A work budget that stops a rule on a large instance stops it on every machine, also on one that had time left. A run-to-run difference in the reduced model would make every one-switch comparison noisy.
- **Test at the end:** Not a quality test: log the hash of the reduced model in every cluster run and compare between repetitions, thread counts, a second GPU type and both memory paths.
- **State:** in the engine (host); the device part is planned.

### AS-289 Every tie goes to the smaller index; the reduced model depends on the input order
- **Assumption:** No hash container is used; buckets are sorted keys. In the Stage 1 engine a family serves its rows and columns in the order of their first event since it last took its list; the start step, the buckets of parallel rows and the switch to explicit bounds go in index order. The column rules are designed to take candidates in index order. Every tie goes to the smaller column index (the binary of simple probing and of the single-lock alias, the kept column of a merge, the absorber, the removed singleton column, the source of an implied equation, the best prices of a row). The reduced model therefore depends on the row and column order of the input: a permuted model can give another reduced model.
- **Set by:** Design (v2_design.txt:129; lowworth_v2_design.txt:101).
- **Where:** gpu_presolve/include/gprs/state.hpp:109-111; stage1c_design/v2_design.txt:129; lowworth_v2_design.txt:45, :59, :101.
- **Switch:** no switch yet: needs one (ties by a quality measure instead of the smaller index).
- **Alternatives:** Ties by a quality measure (largest pivot, most findings, cost, row length).
- **May change:** Which of several equally good columns is fixed, kept, made a source or used as a compensator.
- **Test at the end:** Not planned as a pair. A possible test: cuOpt on the 240 with permuted columns in both arms, to see the spread that ties cause; and a size run with permuted input.
- **State:** in the engine for Stage 1; designed for the column rules.

### AS-290 A row activity has one summation order, fixed by entry position
- **Assumption:** A row activity summed from scratch has one order in every engine: the live entries of the row in storage order are dealt to lanes (up to 64 entries at most two per lane, 256 lanes above) and the lanes are merged by a fixed tree, with no sort. The reason: summing in lane groups differs in bits from a single thread over the row on 9 of 32 matrices. Because the sum follows storage order, the same model with the entries of its rows stored in another order can reduce differently (AS-281).
- **Set by:** The owner (HISTORY Entry 13, 2026-11-26: "5. do as recommended").
- **Where:** DESIGN.md section 0 decision 5 (line 192), section 2.4 (lines 416-424), section 2.5 (lines 459-461); gpu_presolve/include/gprs/numerics.hpp:139-155.
- **Switch:** the constants `max_light_row` = 64 and `max_activity_lanes` = 256 (gpu_presolve/include/gprs/numerics.hpp:139-155); no option.
- **Alternatives:** A sequential sum on the host and another order on the device; sorted summation; two fixed orders; one thread per row up to 64 entries (1.45 times the cost of the lane-group kernel).
- **May change:** The last bits of activities, which decide borderline tests. The boundary 64 is also the boundary of the device's bins, so changing it changes the bits.
- **Test at the end:** Not a quality pair: a size run on the 240 with the entries of every row permuted, counting how many reduced models differ; bit equality of the host sum and the device sum.
- **State:** in the engine (host); device kernels planned.

### AS-291 Activities are compensated sums, updated incrementally, kept without a measured benefit
- **Assumption:** Minimum and maximum activity of a row are kept as compensated sums (a high and a low part); a bound change removes the old term and adds the new one as error-free products instead of recomputing. Stage 0 measured the price (39% of the activity kernel, 14% of a sweep) and no benefit: on one fresh sweep the derived bounds differ from plain sums on 5 of 32 matrices, by at most 2.2e-9.
- **Set by:** The owner (HISTORY Entry 2: "I would consider using compensated or Kahan summation for the activities instead of recomputing them from scratch."); "kept without a measured benefit" is the Stage 0 report (change 7), which entered the plan with the block of thirteen changes.
- **Where:** DESIGN.md section 2.4 (lines 398-411); gpu_presolve/include/gprs/numerics.hpp:58-103.
- **Switch:** no switch yet: needs one (the arithmetic is built in).
- **Alternatives:** Plain double sums with periodic recomputation; recomputation from scratch; HiGHS's scheme.
- **May change:** Presolve time, and decisions at the tolerance boundary after many updates.
- **Test at the end:** Not a quality test: a build with plain sums, first the campaign with huge values, then reduced sizes and seconds on the 240; a cluster pair only if reduced models differ.
- **State:** in the engine.

### AS-292 Counts of infinite terms are exact, and one infinite term still gives one bound candidate
- **Assumption:** The number of infinite terms of each row activity is kept exactly, because a capped count cannot be decremented when a bound becomes finite; this differs from the owner's own review comment on cuOpt pull request 2372 (cap at 2). With no infinite term every column of the row may get a bound candidate; with exactly one, only the column that holds it; with two or more, none. On the device the counts are to be 16-bit integers in arrays apart from the sums, and a count that saturates marks the row as skipped until it is recounted (split arrays kept without a deciding measurement).
- **Set by:** Design (HISTORY Entry 5: "One difference from the review, forced by incremental maintenance", after the owner's "Consider the review comments there too."). The split arrays for the counts entered the plan with the block of thirteen changes of the Stage 0 report.
- **Where:** DESIGN.md section 2.4 (lines 435-455); gpu_presolve/include/gprs/state.hpp:86-90; HISTORY Entry 5.
- **Switch:** no switch yet: needs one for bound candidates from rows with one infinite term; the count itself is always exact.
- **Alternatives:** Counts capped at 2 with a recount; 32-bit counts; paired arrays.
- **May change:** Bound candidates from rows with a single infinite term, which cuOpt's built propagation cannot derive. A row whose 16-bit count saturates is skipped by device propagation, which the host would not do: a possible difference between the engines on rows longer than 65,535 entries.
- **Test at the end:** Not a quality test: equality of host and device hashes on models with such rows. A quality pair would need a switch for candidates from rows with one infinite term.
- **State:** in the engine (host); the device layout is planned.

### AS-293 No floating-point atomics and no contracted floating-point operations
- **Assumption:** A floating-point sum always has one fixed order; library reductions are used for integers and for minimum and maximum only; integer atomics are allowed. Contracted floating-point operations are off: the engine and the adapter are compiled with `-ffp-contract=off`, and the only fused multiply-add is the explicit one in the error-free product. The adapter therefore must not instantiate floating-point templates shared with the rest of the library; the compiler flags for ARM are left open.
- **Set by:** The owner (HISTORY Entry 2: "Use reductions instead of atomics if needed."); design for the compile flag.
- **Where:** DESIGN.md section 2.5 (lines 459-469); gpu_presolve/include/gprs/numerics.hpp:14-15; gpu_presolve/CMakeLists.txt:37-43; cpp/CMakeLists.txt:635-647; cpp/src/presolve/gprs_presolve.cpp:8-9.
- **Switch:** the compile option `-ffp-contract=off` on the engine sources (gpu_presolve/CMakeLists.txt:37-38) and on the engine and adapter sources inside cuOpt (cpp/CMakeLists.txt:635-641).
- **Alternatives:** Floating-point atomics (faster, order-dependent).
- **May change:** If a build drops the flag, activities and with them reductions can differ in the last bits between compilers and machines.
- **Test at the end:** Not a quality test: the hash of the reduced model on the dev node and in the cluster build for the 240.
- **State:** in the engine and the build.

### AS-294 The engine is compiled with other flags inside cuOpt than standalone; equal results are assumed
- **Assumption:** The standalone engine, on which every size gate, check and campaign runs, is compiled with `-O2`, `-ffp-contract=off` and, on x86-64, `-mfma`. Inside cuOpt the same sources are compiled with cuOpt's own flags, which on x86-64 add `-march=x86-64-v3 -fno-math-errno` (and `-fno-trapping-math` in a build without assertions), plus `-ffp-contract=off`. It is taken for granted that both builds, and every compiler, give the same reduced model bit for bit.
- **Set by:** Code (the build files; no decision is recorded).
- **Where:** gpu_presolve/CMakeLists.txt:17-20, 37-43 (the comment on `-mfma`: "The result is the same bits either way"); gpu_presolve/build_host_local.sh:21-24; cpp/CMakeLists.txt:610-618, 635-641; cpp/src/presolve/gprs_presolve.cpp:264-276 (the log line has sizes and no hash); HISTORY Entry 14 (line 893: Apple clang 17 on the Mac, gcc 14 on the node), Entry 45 (lines 2500-2504), Entry 37 (lines 2124-2126). The optimisation level of the cuOpt build was not read.
- **Switch:** no switch yet: needs one (the flags are lines of the two build files).
- **Alternatives:** Compile the engine sources inside cuOpt with exactly the standalone flags; build the standalone engine with cuOpt's flags for the gates and checks.
- **May change:** If a flag changes one rounding, an activity and with it a reduction differs, and the engine that the checks and the campaign cleared is not the engine a cluster arm runs. Evidence for equality: the Mac build gives the hashes of the node build on local dumps of all 240. Evidence missing: inside cuOpt only sizes were compared (equal on 238 of the 240, two unexplained, AS-281), and the adapter's log line carries no hash.
- **Test at the end:** Not a quality test: log the hash of the reduced model in the cuOpt build and compare it with the standalone hash on the 240 on the same input model (AS-179), for the exact build of every cluster arm.
- **State:** in the build.

### AS-295 Fixed columns stay in their rows as constants until a rebuild
- **Assumption:** A fixed column is not removed from its rows at once: its term stays as a constant and every row test subtracts it from the side. The state is rebuilt between phases when more than 25% of the slots are dead or more than 25% of the live entries belong to fixed columns, and only if every row can take its constants into its sides; before the last rounds the constants are moved into the sides and the rounds run again.
- **Set by:** Design (plan versions 3 and 4; the repairs of HISTORY Entries 18 to 25).
- **Where:** DESIGN.md section 2.3 (lines 364-366); gpu_presolve/include/gprs/presolve.hpp:151-154, 283-294; gpu_presolve/src/presolve.cpp:314-329, 365-384, 425-435; gpu_presolve/src/state_storage.cpp:80-119, 150-217.
- **Switch:** `options_t::rebuild_dead_fraction` and `options_t::rebuild_fixed_fraction` (both default 0.25); `plain_size` = 1e5 is a named constant.
- **Alternatives:** Remove a fixed column from its rows at once and shift the sides; rebuild after every phase, or never before the end.
- **May change:** The side of the free part is a difference of two doubles, a term above 1e20 makes an activity infinite, and size limits on a side see the fixed terms, so the moment of a rebuild can change borderline tests; when a row cannot take its constants the presolve ends without a fixed point.
- **Test at the end:** Not a quality test first: a size run with the fractions at 0, 0.25 and 1, and each reduced model presolved a second time; count the instances whose result differs.
- **State:** in the engine.

### AS-296 The state cannot add a row or a column, and a row cannot grow
- **Assumption:** Rows and columns keep the indices of the input for the whole presolve, a row owns the slots it had at the start, and an entry can only be changed, deleted or moved to another column in the same slot. No reduction of the engine adds an entry, and the termination argument of the row subtraction rests on that.
- **Set by:** Design (Stage 1 code structure).
- **Where:** gpu_presolve/include/gprs/state.hpp:52-61, 245-250, 464-473; gpu_presolve/src/state_build.cpp:264-272; gpu_presolve/include/gprs/rules.hpp:305-307.
- **Switch:** no switch yet: needs one, and new state operations first (append a row or a column, let a row grow).
- **Alternatives:** State operations that append rows and columns and let a row grow (the plan's row overlay).
- **May change:** Reductions that strengthen by adding (indicator strengthening that appends rows, clique rows, substitutions with fill-in) cannot be built in this engine as it is; today they can only run after it, in cuOpt's own reductions.
- **Test at the end:** Not testable by a switch. The end test of the added-constraint reductions is at the adapter: mode 2 against mode 3, and the pair of AS-003.
- **State:** in the engine.

### AS-297 The engine's model has continuous and integer columns only and keeps the input order
- **Assumption:** A model is a minimisation with rows between two sides and columns of type continuous or integer; a binary is an integer with bounds 0 and 1. There is no implied-integer type, no semi-continuous column and no indicator constraint. The reduced model keeps the remaining rows and columns in the order of the input.
- **Set by:** Design (Stage 1 code structure).
- **Where:** gpu_presolve/include/gprs/state.hpp:18-22, :43; gpu_presolve/src/state_build.cpp:15-45; gpu_presolve/src/state_storage.cpp:403-515.
- **Switch:** no switch yet: needs one for a third type.
- **Alternatives:** A type for implied integers (AS-165); indicator rows in the model.
- **May change:** A problem with something the model cannot carry is not presolved by the engine at all (AS-156); column and row order can decide ties in the solver.
- **Test at the end:** Not a quality test: count the instances of the 240 that the adapter refuses.
- **State:** in the engine.

### AS-298 Postsolve keeps few kinds of records and rounds integer columns before the undo
- **Assumption:** The engine writes a postsolve record when a column is fixed or aliased, and for a merged parallel row and a row subtraction (the last two are ignored by the primal undo); other row edits leave none. The plan's further kinds (merge, singleton column with its row, stuffing, shift and scale) come with the column rules. Undo runs in reverse order with one tolerance set. Before the records are undone, each integer column of the reduced point is replaced by its nearest integer, whatever its distance from it; substituted columns are then computed from the rounded values in plain double arithmetic, without moving them into their bounds.
- **Set by:** Design (DESIGN.md section 8: "Integers are snapped before the undo").
- **Where:** DESIGN.md section 8 (lines 691-697); gpu_presolve/src/postsolve.cpp:144-165; gpu_presolve/include/gprs/postsolve.hpp:104-107.
- **Switch:** no switch yet: needs one for the rounding.
- **Alternatives:** No rounding; rounding only within the integrality tolerance; full primal and dual postsolve as HiGHS and PaPILO keep.
- **May change:** A solver solution whose integer values are off by up to the solver's integrality tolerance is moved by that much, which chains of substitutions can multiply; the postsolved point can violate a row that the unrounded point satisfied, or satisfy one it violated.
- **Test at the end:** Needs the switch. Standard cluster comparison: rounding on against off, with the number of incumbents that fail the feasibility check on the original model.
- **State:** in the engine.

### AS-299 A point is crushed by dropping removed columns; for the column rules by forward replay
- **Assumption:** Today crush copies the values of the columns that are left; in primal-only mode a feasible point gives a feasible point with the same objective value. For the nine column rules the design makes the crush a forward replay (moving weight at a dominance fixing, clipping at a dual bound, adding at a merge) that gives a feasible reduced point with an objective not above; the same promise holds for the implied equation, stuffing, the single equation, the dominated column and the weak conversion. Two of the five further rules have no forward map as HiGHS states them. A false alarm of the crush check on a real instance "is still possible where terms cancel".
- **Set by:** Design (DESIGN.md section 8; v2_design.txt:113; lowworth_v2_design.txt:26-61, :229; a binding amendment for the weak crush, lowworth_v2_amendments.txt:15-16).
- **Where:** gpu_presolve/include/gprs/postsolve.hpp:109-114; gpu_presolve/src/postsolve.cpp:167-175; gpu_presolve/include/gprs/presolve.hpp:214-219; stage1c_design/v2_design.txt:113, :136, :265; lowworth_v2_design.txt:26, :33, :47, :54, :61, :106, :229.
- **Switch:** `options_t::primal_only` decides which promise holds; the forward replay itself has no switch.
- **Alternatives:** Dual rules tested by the oracle only; an equal-objective crush for every rule.
- **May change:** Whether start solutions and incumbents that cuOpt hands to the reduced problem stay feasible and keep their objective value (AS-169); a crushed point differs from the original point in the moved columns.
- **Test at the end:** Not a quality test: the crush check on published MIPLIB solutions and HiGHS points (184 of the 240 in the newest check and 39 testbed instances) in both modes, after each milestone, and the wrong engines of the replay.
- **State:** in the engine (dropping); the replay is designed (milestone of dominance between parallel columns [M3]).

### AS-300 No column changes type until the implied-integer conversion, which is the first type change
- **Assumption:** The nine rules change no column type and widen no bound of a column that keeps its meaning; only the kept column of a merge gets wider bounds. All existing families were written under "no type change". The conversion is the first change of a column type after construction; the design sets the rule that no family keeps a fact that depends on a column type across calls unless it follows the events of the type change, and that postsolve reads "integer at the position of the record" instead of the input type.
- **Set by:** Design (v2_design.txt:109; lowworth_v2_design.txt:153, :157, :226; taken into v2_amendments.txt:102-104, 115-117).
- **Where:** stage1c_design/v2_design.txt:109; v2_amendments.txt:14-18, 94-117; lowworth_v2_design.txt:153, :157, :226; lowworth_v2_answers.txt:18.
- **Switch:** the three designed conversion switches `implied_integer_equation`, `implied_integer_weak` and `implied_integer_scaling` (not in code): all three off avoids any type change.
- **Alternatives:** A separate implied-integer type that the old families treat as continuous.
- **May change:** A family that caches a type without an event would act on a stale one; the undo of a merge after a conversion gives another value with the type at the record than with the input type (the critic's chain: 2 instead of 3/2).
- **Test at the end:** Not a quality test: the milestone of the implied-integer conversion [L2] reruns the tests and campaigns of the nine rules with conversions happening, plus two wrong engines.
- **State:** designed (milestone of the implied-integer conversion [L2]; parts are due with the singleton-column rules [M1] and with the merge of parallel columns [M5]).

### AS-301 No bounds on row duals are kept as state
- **Assumption:** The engine stores no bounds on row duals and no bounds on reduced costs. Every premise of the nine rules is local to one row or column; the implied equation reads the one price it needs at the moment of application, and the dominated-column rule recomputes prices of rows per pass. The plan still says the five rules "need bounds on the row duals as new state" (DESIGN.md:130-131); the design that followed found that every such bound in HiGHS is cost over coefficient of one column with at most one lock in a direction. The plan text was not updated.
- **Set by:** Design (v2_design.txt:99; decision 1 of lowworth_v2_design.txt:238, put to the owner; no answer in HISTORY up to Entry 46).
- **Where:** DESIGN.md amendment A8 (lines 130-131) against HISTORY Entry 41 (lines 2279-2283) and Entry 43; stage1c_design/v2_design.txt:99; v2_amendments.txt:14-18; lowworth_v2_design.txt:68, :238.
- **Switch:** no switch yet: needs one only if stored bounds are built; on the HiGHS side `implied_row_dual_bounds` freezes its bounds.
- **Alternatives:** HiGHS's kept implied bounds: about 24 bytes per row and 44 per column, a reset in about a dozen state operations, the same on the device engine, and stale bounds as a new kind of error.
- **May change:** A bound on what the choice can lose: with HiGHS's implied bounds on the row duals frozen, all cheap rules lose 0.9990 / 0.9984 / 0.9988 (nonzeros / rows / columns) on the 240 and 0.9961 / 0.9960 / 0.9966 on the 40. Later dual rules (pairwise dominated columns, a stronger free-column substitution) would also read such bounds.
- **Test at the end:** First a size run: the engine's dual rules against HiGHS with all cheap rules per instance on the 240, once the five rules exist. Only if a gap traces to dual bounds: stored bounds behind a switch and a standard cluster comparison.
- **State:** designed.

### AS-302 Engine and cheap rules run on one host thread; inside cuOpt nothing uses the GPU
- **Assumption:** The Stage 1 engine uses the standard library only, no CUDA and no threads; the nine column rules and the five further rules are designed as sequential rules of the host engine in index order, with no GPU batch form. cuOpt compiles the host engine and the adapter; the device code of the project is not part of the build, so the GPU does no presolve work in any arm, at any size.
- **Set by:** Design (host engine first, DESIGN.md amendment A8 line 101 and section 10); the owner for the timing of the GPU work (HISTORY Entry 33: "Let's do it after the cheap rules.").
- **Where:** gpu_presolve/CMakeLists.txt:6; gpu_presolve/include/gprs/presolve.hpp:10; cpp/src/presolve/CMakeLists.txt:6-12; stage1c_design/v2_design.txt:99; lowworth_v2_design.txt:66.
- **Switch:** no switch yet: needs one (the plan foresees a size switch between the engines that must not change a result).
- **Alternatives:** Device batches with the plan's reservation table; several host threads; device state build and device phases behind a size threshold.
- **May change:** Presolve time only, and through it the primal integral, as long as a later device form gives the bits of the host.
- **Test at the end:** Not a quality pair: a timing test at the end (host against device), with equal hashes.
- **State:** in the engine and the adapter.

### AS-303 The prices of the dominated-column rule are cached for one pass
- **Assumption:** The best compensators and their prices of a scanned row are cached for one pass of the family and cleared when a fixing touches a cached compensator; the assertion build compares every used price with a fresh scan.
- **Set by:** Design (lowworth_v2_design.txt:59, :89; refined by an amendment, lowworth_v2_amendments.txt:38-39).
- **Where:** stage1c_design/lowworth_v2_design.txt:59, :62, :89; lowworth_v2_amendments.txt:38-39; lowworth_v2_answers.txt:78, :146.
- **Switch:** no switch yet: needs one only to test the cache (prices recomputed per candidate); the assertion build compares every used price with a fresh scan.
- **Alternatives:** Recompute per candidate; stored dual bounds as in HiGHS.
- **May change:** Nothing in the result if the invalidation is exact; a stale price would give a wrong or a missed fixing.
- **Test at the end:** Not a quality test: the assertion build and the second-presolve tests.
- **State:** designed (milestone of the dominated column [L5]).

### AS-304 The device builds the state with the host's bits; integration comes after the cheap rules
- **Assumption:** The GPU part of Stage 1 is the state build (transpose, cross positions, locks, then the first activity sums), each array bit-equal to the host's, so that the choice of engine cannot change a result and needs only a size switch. It exists as a separate bench project and is not integrated. Measured as a presolve call gets it (first build of a fresh process, 15 instances from 7e4 to 2.6e8 nonzeros, one H200): the device loses up to 850,000 nonzeros and wins from 1.3 million (1.9 to 2.0 times). Three quarters of the device path's time is the host creating the result arrays; huge pages alone take the host constructor from 34 s to 24 s on the 15 instances. The default transpose is a stable radix sort; cuSPARSE's order inside a column was only observed. No threshold is fixed in code.
- **Set by:** The owner for cuSPARSE and the timing (HISTORY Entry 26: "you can try cusparse for transpose and spmv or other sparse work"; Entry 33: "Let's do it after the cheap rules."); design for the bit equality (DESIGN.md amendment A4).
- **Where:** DESIGN.md amendment A4 (lines 48-60) and amendment A9 (lines 154-157); HISTORY Entry 32 (lines 1908-1955); gpu_presolve/device/include/gprs/device/state_build.hpp:23-77; gpu_presolve/device/bench/.
- **Switch:** no switch yet: needs one (a size switch); the bench tool has `build_options_t::transpose` (`cusparse`, `radix_sort`, `scatter_sort`) and `::transfer` (`pageable`, `pinned`, `pinned_chunked`).
- **Alternatives:** Host only; device from another size; huge pages on the host without a GPU; device propagation after the first host cascade, as in plan version 4.
- **May change:** Nothing in the reduced model while bit equality holds (the device arrays had the host's bits in all 3,734 compared builds); presolve seconds (the state build is 43 to 46% of engine time). If a library version changed the order inside a column, results could differ between host and device. DESIGN.md amendment A4 supersedes the rule that the device copy is created after the first host cascade, which had its own reason (the Stage 1 rules remove at least half of the rows on 9 of the 31).
- **Test at the end:** Not a quality test: hash identity with the device path forced on the 240 and the 40, the campaign with the device path forced, then three alternating timing runs above the switch size.
- **State:** designed; built as a bench project, integration planned after the cheap column rules.

### AS-305 Device bound propagation is postponed, ships off, and need not agree with the host
- **Assumption:** Of the design of GPU bound propagation only the parts that the later probing stage reuses are built now (memory resource, device image, bit-equal row sums, hand-over); frontier kernels and the threshold measurements wait. The design ships with the device off (`engine = host`) and provisional values for the settings that can change a result: `min_nnz` = 500000, `frontier_work` = 20000, `handback_work` = 20000, `handback_rounds` = 1, `fixed_cost_per_nnz` = 5, `byte_budget` = 0. In the design the device applies a round from one snapshot of the bounds and the host goes row by row, so the two are each checked for validity and are required to reach the same bounds only on integer models with exact data run to an empty frontier.
- **Set by:** Design (the Stage 1b design round; narrowed by DESIGN.md amendment A4 after the owner's cuSPARSE message).
- **Where:** stage1b_design/final_design.txt:45, :142, :171, :185, :224, :246; DESIGN.md amendment A4 (lines 56-60) and section 13 (lines 853-855).
- **Switch:** `options_t::device` (designed `device_options_t` with `engine` = `host`, `automatic`, `device`); not in the engine headers.
- **Alternatives:** Ship device propagation off for good; other thresholds after measurement; a device engine that reproduces the host's order (slower).
- **May change:** With the device on, bounds are tightened in rounds from one snapshot, which can use the work budget differently and leave other bounds than the host rule, so the reduced model can differ between a run that used the device and one that did not.
- **Test at the end:** If built: `engine` = `host` against `device` forced, hashes and reductions first, then a standard cluster comparison.
- **State:** designed (Stage 1b); waits for the probing and clique stage.

### AS-306 The engine of a phase is chosen by a work estimate against provisional thresholds
- **Assumption:** Each phase is given to the host or the device by a deterministic work estimate against thresholds, which includes a constant per kernel launch and per read-back; each presolve call has a device fixed-cost budget (proposed: 5 work units per nonzero, "a policy choice, not derived"), and a phase whose fixed cost does not fit runs on the host or is skipped. Provisional Stage 0 values (the 31, one H200): work unit 10 ns; full sweep on the GPU from 32,000 nonzeros (nothing below 71,477 nonzeros was measured); GPU sweep of changed rows from 20,000 work units; rebuild on the GPU from 100,000 nonzeros; pinned host mirror below 64 MiB (a placeholder). Not set at all: the hand-back threshold, the byte budget, the probing budget with its caps, the hard cap of pairwise dominated columns, the budgets of every phase not built.
- **Set by:** The owner accepted them as provisional (decision 6, HISTORY Entry 13); the values are the Stage 0 report's. They entered the plan with the block of thirteen changes, not one by one.
- **Where:** DESIGN.md section 2.1 (lines 302-338) and section 13 (lines 803-819, 833-844); STAGE0_REPORT.md lines 336-344; gpu_presolve/analysis/stage0_engine.py.
- **Switch:** no switch yet: needs one per threshold (only the row weight `line_visit_work` exists in code).
- **Alternatives:** The ranges the plan itself gives (for example 9,514 to 53,817 work units for the changed-rows threshold); always the host when the device does not fit; no fixed-cost budget.
- **May change:** The engine thresholds change time only while host and device give the same bits. "Or is skipped" means a reduction can be left undone because of a budget, and the probing and pairwise caps decide how far reductions go, so those change the reduced model (the same on every machine).
- **Test at the end:** For the thresholds: timing with each forced low and high, and hash equality. For each cap that can cut a reduction short: standard cluster comparison, the cap at its default against far above it, with the count of skipped phases per instance.
- **State:** planned (device engine and wide phases).

### AS-307 A device threshold is fixed once, at the smallest summed time over the testbed
- **Assumption:** A threshold that can change a result is to be fixed once, at the value with the smallest summed time of the logged propagation calls over the testbed, with ties going to the host and under the condition that with the defaults no instance is slower than host only by more than the larger of 5% and 2 ms on the minimum of three alternating runs.
- **Set by:** Design (the Stage 1b design round).
- **Where:** stage1b_design/final_design.txt:222.
- **Switch:** no switch of its own: a rule for setting the defaults of the designed `device_options_t` (AS-305).
- **Alternatives:** Fix on the 240; allow a small loss on single instances for a gain in the sum.
- **May change:** The shipped thresholds and so, on every machine, which engine works on which call.
- **Test at the end:** Not a quality test: a timing test.
- **State:** designed (milestone 7 of Stage 1b).

### AS-308 Memory: pinned mirror on x86, managed memory on Grace Hopper, unmeasured until the end
- **Assumption:** On x86 with a discrete GPU the matrix lives in device memory with a pinned host mirror below a size threshold, and mutable state crosses as lists of changed entries in a fixed record order (the cheapest hand-back measured, 46 microseconds for an unordered list, is not used; the ordered list costs up to about 141 microseconds). On Grace Hopper the arrays are to be managed memory read in place. A Grace Hopper node exists, but benchmarking and performance work there wait until the end; that path is functional only, is tested by forcing it on x86, and is outside every gate. Every GPU number so far comes from one H200 with an x86 host.
- **Set by:** The owner (HISTORY Entry 2: "consider Grace Hopper low-latency unified memory workloads"; Entry 45, 2026-11-27: "GH is available but let's leave the benchmarking and perf optimization there to the end.").
- **Where:** DESIGN.md amendment A10 (lines 177-178), section 2.2 (lines 340-360), section 13 (line 800); stage1b_design/final_design.txt:45, :99, :199, :254, :257.
- **Switch:** no switch yet: needs one (designed `memory_path` = `automatic`, `pinned`, `managed`).
- **Alternatives:** Measure on Grace Hopper and a second GPU type during development; managed memory everywhere (on x86, 64 MiB touched by the host after a kernel cost 14.4 ms as managed memory against 4.0 ms as a pinned copy); drop the unified-memory path.
- **May change:** Time and memory only, while both memory paths give the same hashes; thresholds and crossovers tuned on x86 with a discrete GPU may be wrong there.
- **Test at the end:** Not a quality test: determinism hashes equal on both memory paths and on Grace Hopper against x86 on the 240 and the 40, then the time gates repeated on a GH200 node; the end test can then be repeated there with the same arms.
- **State:** planned.

### AS-309 The device matrix is binned by row and column length, partly without a measurement
- **Assumption:** On the device the matrix is to be stored binned by length: up to 64 entries, up to 8,192 (one block), chunks above. Rows up to 64 entries are folded by lane groups between 5e4 and 1e7 nonzeros; above, the scheduling is open. Storage in binned order is kept without a measurement, and the lane-group rule is decided in the geometric mean: in the size class from 3e6 to 1e7 nonzeros lane groups are fastest on 1 of 5 matrices.
- **Set by:** The owner accepted it with decision 6 (changes 7 and 13 of the Stage 0 report); taken from cuOpt's load-balanced propagation, which the owner suggested using (HISTORY Entry 2). Both changes entered the plan with the block of thirteen changes, not one by one.
- **Where:** DESIGN.md section 2.3 (lines 379-394).
- **Switch:** no switch yet: needs one.
- **Alternatives:** Other bin boundaries; one thread per row; no binning.
- **May change:** Time only; the boundary 64 is shared with the summation order (AS-290).
- **Test at the end:** Not a quality test: kernel timing.
- **State:** planned (device).

### AS-310 GPU batches are made independent by a reservation table; a losing candidate is dropped
- **Assumption:** Candidates of a GPU batch claim cells (row, column entries, column bounds, cost, type) with keys of hash and candidate id; the selection is iterated to a maximal independent set, and a candidate that loses leaves no trace and has to be found again later. The new build order does not say where the reservation table goes.
- **Set by:** Design (plan versions 2 to 4).
- **Where:** DESIGN.md section 4 (lines 508-512) and section 13 (line 799).
- **Switch:** no switch yet: needs one.
- **Alternatives:** Sequential application on the host with live counts.
- **May change:** Which of two conflicting reductions is applied is decided by a hash order instead of by rule order or score, so the reduced model can differ from a sequential application; deterministic in both cases.
- **Test at the end:** When built: standard cluster comparison, batches through the reservation table against sequential host application of the same candidates.
- **State:** planned.

### AS-311 The planned probing launches many sources at once that do not see each other's results
- **Assumption:** The new probing launches many probe sources at once on a shared base state with a sparse overlay per slot; sources of one launch do not see each other's results, candidates are ordered by HiGHS's implication score, and re-probing covers only sources whose neighbourhood changed. A review noted that this does not scale past about 1e6 nonzeros with dense slots and that one re-probe pass does not recover what sequential probing finds in a chain. The kernels are to come from cuOpt's load-balanced propagation, which no longer compiles.
- **Set by:** Design (plan versions 2 to 4), after the owner's "There is probing already implemented in cuOpt, and there is also load-balanced bound propagation. You may use that" (HISTORY Entry 2).
- **Where:** DESIGN.md section 7 (lines 604-648); HISTORY Entry 4.
- **Switch:** no switch yet: needs one.
- **Alternatives:** Sequential probing that uses each result at once (HiGHS, cuOpt today); cuOpt's probing cache as it is.
- **May change:** A batch finds less per probe than sequential probing; the fixings, implications and cliques handed to the solver strengthen without a size effect.
- **Test at the end:** Standard cluster comparison when built: cuOpt as is against cuOpt with the new probing in place of the probing cache, PaPILO unchanged; and the export of implications off against on.
- **State:** planned (step 4 of the build order).

### AS-312 The clique pipeline replaces the paper's random choices by fixed rules
- **Assumption:** The clique pipeline follows Dai and Chen (2025), adapted to GPU kernels (the paper is a CPU-thread method): cliques from knapsack rows are stored implicitly, explicit edges are kept only for small cliques, and a clique is extended greedily in literal order from the neighbours of its minimum-degree member. The paper's random shuffles and random truncation are replaced by fixed rules, and caps are functions of the model.
- **Set by:** Design, after the owner's pointer to the paper (HISTORY Entry 2: "If you think it's valuable, read that paper and implement that clique extraction from that paper."; Entry 3).
- **Where:** DESIGN.md section 7 (lines 650-668, 680-681); HISTORY Entry 3.
- **Switch:** no switch yet: needs one.
- **Alternatives:** The paper's randomised version (not deterministic); HiGHS's explicit clique table; cuOpt's existing table.
- **May change:** Greedy extension in literal order depends on the numbering of the columns: the same model with columns renumbered can get other cliques.
- **Test at the end:** Standard cluster comparison when built: cuOpt's clique table against the new pipeline; plus a test with columns renumbered.
- **State:** planned.

### AS-313 The presolve is a standalone library with no cuOpt code and no PaPILO inside
- **Assumption:** The presolve is a static library with its own CMake project that uses no cuOpt target or header and no PaPILO code, also not as a tail for rules it lacks. Kernels and helpers from cuOpt are lifted once as copies, which do not follow later fixes in cuOpt, and nothing is taken from cuOpt's probing cache. The adapter converts between cuOpt's problem and the engine's model on every call.
- **Set by:** The owner (HISTORY Entry 2: "create a separate project not including cuOpt to speed up compilation and comparison against other presolves"; "Do not use PaPILO in the CPU part. You can use it for comparing against, but do not use it within the new presolve.").
- **Where:** DESIGN.md section 0 (line 206), section 1 (line 237), section 11 (lines 738-743).
- **Switch:** `presolve_gprs` inside cuOpt: mode 1 runs PaPILO after the engine and mode 2 the engine alone; the library itself has no PaPILO to switch.
- **Alternatives:** Develop inside cuOpt's targets; PaPILO as the tail (plan version 1).
- **May change:** Until the engine has all rules, the mode without PaPILO has fewer reductions than cuOpt today.
- **Test at the end:** Not a quality test of its own; `presolve_gprs` 1 against 2 shows what PaPILO still adds.
- **State:** in the repository.

### AS-314 The standalone code follows the semantics of three open cuOpt pull requests
- **Assumption:** Propagation, the bound-change stack and objective propagation follow the semantics of cuOpt pull requests 2372 (single-infinity tightening), 2127 (CPU domain propagation) and 2129 (objective propagation), read on 2026-11-26, so that a back-port matches.
- **Set by:** Design (the owner pointed to pull request 2372, HISTORY Entry 5).
- **Where:** DESIGN.md section 7 (lines 615-617) and section 2.1 (lines 304-306).
- **Switch:** no switch yet: needs one only if the engine is to follow cuOpt's merged code instead; the semantics are built into the engine's propagation.
- **Alternatives:** Follow cuOpt's merged code only.
- **May change:** If the pull requests change before they merge, the engine and cuOpt's propagation disagree on borderline bounds.
- **Test at the end:** Not a quality test; the numeric regression cases of pull request 2372 are in the oracle.
- **State:** in the engine.

### AS-315 Postsolve and crush of the engine are called from several threads at once
- **Assumption:** The early CPU heuristic and the solver's publication path call the adapter's postsolve and crush concurrently; it is assumed that these engine functions only read shared state. No record of a thread-sanitizer run was found.
- **Set by:** Design (DESIGN.md section 9: "postsolve object with a host entry callable from several threads").
- **Where:** cpp/src/presolve/gprs_presolve.hpp:55-62; cpp/src/mip_heuristics/solve.cu:286-381.
- **Switch:** no switch yet: needs one only if a lock is put around the calls; both functions are documented as not changing the object (gpu_presolve/include/gprs/postsolve.hpp:104-114).
- **Alternatives:** A lock around the calls.
- **May change:** A data race would corrupt published solutions without any message.
- **Test at the end:** Not a quality test: a thread-sanitizer run of a short solve in mode 3.
- **State:** in the adapter.

### AS-316 Storage housekeeping follows fixed thresholds
- **Assumption:** A row or column is compacted when it has more deleted slots than live entries; a column that receives an entry without a free slot moves to the end of the store with twice its capacity, at least 4 slots; the parallel-row search sorts all keys again when the new keys reach one in 32 rows. Compaction sums the activities of the row again.
- **Set by:** Code (performance choices of the implementation).
- **Where:** gpu_presolve/src/state_ops.cpp:101-113, 167-197; gpu_presolve/src/state_storage.cpp:64-78; gpu_presolve/src/rules_parallel_rows.cpp:132-138.
- **Switch:** the named constants `min_moved_capacity` = 4, `sort_share` = 32 and `comparison_sort_size` = 64; no option.
- **Alternatives:** Other thresholds.
- **May change:** Presolve time; through the fresh sums after a compaction, the last bits of an activity.
- **Test at the end:** Not a quality test: timing and identity of the reduced models on the 240.
- **State:** in the engine.
