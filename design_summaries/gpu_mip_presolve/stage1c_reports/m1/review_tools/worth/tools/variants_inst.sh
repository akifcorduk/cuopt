#!/usr/bin/env bash
# variants_inst.sh: the instrumented tool on the 240 local dumps with the measurement overrides
# of diag.hpp (scratch only; none of them is in the engine).
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_worth
T=$S/bin/gprs_presolve_inst
B=$S/tools/bench240.sh
bash "$B" "$T" inst_default
GPRS_X_GROWTH_MODE=1 bash "$B" "$T" inst_growth_cont
GPRS_X_GROWTH_MODE=2 bash "$B" "$T" inst_growth_intpivot
GPRS_X_GROWTH_MODE=3 bash "$B" "$T" inst_growth_both
GPRS_X_IMPLIED_TOL=2 bash "$B" "$T" inst_implied_2
GPRS_X_IMPLIED_TOL=1000 bash "$B" "$T" inst_implied_1000
GPRS_X_SIDE_SCALE=2 bash "$B" "$T" inst_side_2
GPRS_X_SIDE_SCALE=1e300 bash "$B" "$T" inst_side_off
bash "$B" "$T" inst_growth_inf --pivot-growth-limit inf
bash "$B" "$T" inst_primal --primal-only
echo all done
