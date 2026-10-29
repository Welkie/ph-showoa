"""Summarise kaggle/init_tuning_results/*.csv: per variant and instance group, relative to Algorithm 2 (sa)."""
import csv
import glob
import os
import statistics as st

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_tuning_results")
rows = [r for f in glob.glob(os.path.join(RES, "*.csv")) for r in csv.DictReader(open(f))]
by = {}
for r in rows:
    by.setdefault((r["variant"], r["instance"]), []).append(r)
variants = sorted({r["variant"] for r in rows})
insts = ["cdp101", "cdp103", "cdp203", "cdp207", "rdp103", "rdp107", "rdp201", "rdp210",
         "rcdp101", "rcdp104", "rcdp202", "rcdp207"]
m = lambda v, i, k: st.mean(float(x[k]) for x in by[(v, i)]) if (v, i) in by else None

print("Best cost in the population after generation 1 (mean of 3 seeds); NV min; diversity")
print(f"{'instance':9s}" + "".join(f"{v:>16s}" for v in variants))
for i in insts:
    print(f"{i:9s}" + "".join(f"{m(v, i, 'cost_best'):10.1f} ({m(v, i, 'nv_min'):4.1f})" if m(v, i, 'cost_best') else f"{'-':>16s}" for v in variants))
print()
print(f"{'summary':9s}" + "".join(f"{v:>16s}" for v in variants))
for label, key in (("best vs sa %", "cost_best"), ("mean vs sa %", "cost_mean")):
    line = f"{label:13s}"
    for v in variants:
        d = [100 * (m(v, i, key) - m("sa", i, key)) / m("sa", i, key) for i in insts if m(v, i, key) and m("sa", i, key)]
        line += f"{st.mean(d):+12.2f}%" if d else f"{'-':>13s}"
    print(line)
for label, key in (("NV min", "nv_min"), ("NV mean", "nv_mean"), ("diversity", "diversity")):
    line = f"{label:13s}"
    for v in variants:
        d = [m(v, i, key) for i in insts if m(v, i, key) is not None]
        line += f"{st.mean(d):13.3f}" if d else f"{'-':>13s}"
    print(line)
line = f"{'beats sa':13s}"
for v in variants:
    w = sum(1 for i in insts if m(v, i, 'cost_best') and m('sa', i, 'cost_best') and m(v, i, 'cost_best') < m('sa', i, 'cost_best') - 1e-6)
    l = sum(1 for i in insts if m(v, i, 'cost_best') and m('sa', i, 'cost_best') and m(v, i, 'cost_best') > m('sa', i, 'cost_best') + 1e-6)
    line += f"{f'{w}W/{l}L':>13s}"
print(line)
