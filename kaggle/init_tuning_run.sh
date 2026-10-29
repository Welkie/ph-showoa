#!/usr/bin/env bash
# Round 1 of the initialization tuning (kaggle/init_tuning.py), 4 variants at a time on the CPU.
cd "$(dirname "$0")/.."
OUT=kaggle/init_tuning_results
run() { python -u kaggle/init_tuning.py "$1" "$2" "$OUT/$1.csv" > "$OUT/$1.log" 2>&1; echo "done $1"; }
run sa '{"init": "sa"}' &
run grasp '{}' &
run mixed '{"init": "mixed"}' &
run rc0 '{"rc_weight": 0.0}' &
wait
run seed_due '{"seed_rule": 1}' &
run seed_far '{"seed_rule": 2}' &
run rs0 '{"rs_weight": 0.0}' &
run re20 '{"re_passes": 20}' &
wait
run alpha_low '{"alpha_lo": 0.0, "alpha_hi": 0.2}' &
wait
echo ALL DONE
