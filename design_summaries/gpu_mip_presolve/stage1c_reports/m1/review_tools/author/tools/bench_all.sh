#!/usr/bin/env bash
# bench_all.sh: the final binary on the 240 local dumps with its defaults and with each
# assumption of the singleton-column rules at another value, one run each.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1
T=$S/bin/gprs_presolve_final
B=$S/tools/bench240.sh
bash "$B" "$T" final
bash "$B" "$T" family_off --off singleton_cols
bash "$B" "$T" growth_1e9 --pivot-growth-limit 1e9
bash "$B" "$T" growth_1e6 --pivot-growth-limit 1e6
bash "$B" "$T" growth_1e4 --pivot-growth-limit 1e4
bash "$B" "$T" growth_100 --pivot-growth-limit 100
bash "$B" "$T" growth_1 --pivot-growth-limit 1
bash "$B" "$T" saved_0 --saved-row-factor 0
bash "$B" "$T" saved_1 --saved-row-factor 1
bash "$B" "$T" drift_0 --integral-drift-share 0
bash "$B" "$T" drift_1 --integral-drift-share 1
bash "$B" "$T" only_r5 --off free_singleton --off zero_cost_singleton --off forcing_column
bash "$B" "$T" only_r2 --off free_singleton_dual --off zero_cost_singleton --off forcing_column
bash "$B" "$T" only_r3 --off free_singleton --off free_singleton_dual --off forcing_column
bash "$B" "$T" only_r4 --off free_singleton --off free_singleton_dual --off zero_cost_singleton
bash "$B" "$T" no_r5 --off free_singleton_dual
bash "$B" "$T" no_r2 --off free_singleton
bash "$B" "$T" no_r3 --off zero_cost_singleton
bash "$B" "$T" no_r4 --off forcing_column
bash "$B" "$T" primal --primal-only
