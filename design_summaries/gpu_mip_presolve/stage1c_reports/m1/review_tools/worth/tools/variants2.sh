#!/usr/bin/env bash
# variants2.sh: further measurements on the 240 local dumps: the family as the first phase after
# the cascade (scratch switch), two more values of the growth limit, and the alternatives of the
# growth ratio together with a doubled limit.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_worth
T=$S/bin/gprs_presolve_inst
B=$S/tools/bench240.sh
bash "$B" "$T" inst_default
GPRS_X_ORDER=1 bash "$B" "$T" inst_order_early
GPRS_X_ORDER=1 bash "$B" "$T" inst_order_early_growth_inf --pivot-growth-limit inf
bash "$B" "$S/bin/gprs_presolve_m1" growth_100 --pivot-growth-limit 100
bash "$B" "$S/bin/gprs_presolve_m1" growth_1 --pivot-growth-limit 1
bash "$B" "$S/bin/gprs_presolve_m1" growth_1e4 --pivot-growth-limit 1e4
bash "$B" "$S/bin/gprs_presolve_m1" primal_growth_inf --primal-only --pivot-growth-limit inf
echo all done
