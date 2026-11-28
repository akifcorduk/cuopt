#!/usr/bin/env bash
# variants.sh: their release binary on the 240 local dumps with one rule on alone, one rule off,
# and each limit that has a switch doubled and off.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_worth
T=${TOOL:-$S/bin/gprs_presolve_m1}
B=$S/tools/bench240.sh
bash "$B" "$T" family_off --off singleton_cols
bash "$B" "$T" only_r2 --off free_singleton_dual --off zero_cost_singleton --off forcing_column
bash "$B" "$T" only_r3 --off free_singleton --off free_singleton_dual --off forcing_column
bash "$B" "$T" only_r4 --off free_singleton --off free_singleton_dual --off zero_cost_singleton
bash "$B" "$T" only_r5 --off free_singleton --off zero_cost_singleton --off forcing_column
bash "$B" "$T" no_r2 --off free_singleton
bash "$B" "$T" no_r3 --off zero_cost_singleton
bash "$B" "$T" no_r4 --off forcing_column
bash "$B" "$T" no_r5 --off free_singleton_dual
bash "$B" "$T" growth_2000 --pivot-growth-limit 2000
bash "$B" "$T" growth_inf --pivot-growth-limit inf
bash "$B" "$T" growth_1e6 --pivot-growth-limit 1e6
bash "$B" "$T" growth_500 --pivot-growth-limit 500
bash "$B" "$T" saved_8 --saved-row-factor 8
bash "$B" "$T" saved_inf --saved-row-factor inf
bash "$B" "$T" saved_0 --saved-row-factor 0
bash "$B" "$T" drift_02 --integral-drift-share 0.2
bash "$B" "$T" drift_inf --integral-drift-share inf
bash "$B" "$T" drift_0 --integral-drift-share 0
echo all variants done
