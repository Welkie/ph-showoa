"""Other angles on sections 2 and 3: time, anytime convergence, best/worst run, spread, NV.

usage: python kaggle/analyze_extra.py
For each pair (reference, variant) and each instance: mean, best, worst, sd of the final cost, median s/run,
and mean cost of the population best at several generations. Summary rows count the instances where the
variant is better (lower) and give the mean relative difference in %.
"""
import csv
import os
import statistics as st
import sys

RES = os.environ.get("ABLATION_RESULTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_ablation", "results")
ALL12 = ["cdp101", "cdp103", "cdp203", "cdp207", "rdp103", "rdp107", "rdp201", "rdp210",
         "rcdp101", "rcdp104", "rcdp202", "rcdp207"]
GENS = (1, 10, 25, 50, 100, 200, 500, 1000)


def load(name, inst):
    p = os.path.join(RES, f"{name}__{inst}.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def conv(name, inst):
    p = os.path.join(RES, f"{name}__{inst}__conv.csv")
    if not os.path.exists(p):
        return None
    out = {}
    for r in csv.DictReader(open(p)):
        g = int(r["iteration"])
        if g in GENS:
            out.setdefault(g, []).append(float(r["cost"]))
    return out


def first_hit(name, inst, target):
    """Mean first generation at which the run's best reaches cost <= target (1000 if never)."""
    p = os.path.join(RES, f"{name}__{inst}__conv.csv")
    hit = {}
    for r in csv.DictReader(open(p)):
        run = int(r["run"])
        if run not in hit and float(r["cost"]) <= target + 1e-6:
            hit[run] = int(r["iteration"])
    return st.mean(hit.get(run, 1000) for run in range(1, 31))


def compare(ref, var, label):
    print(f"\n=== {label}: {var} vs {ref} (negative % = {var} better) ===")
    agg = {k: [] for k in ("mean", "best", "worst", "sd", "time")}
    gen_agg = {g: [] for g in GENS}
    hit_agg = []
    for inst in ALL12:
        a, b = load(ref, inst), load(var, inst)
        if not a or not b:
            continue
        ca, cb = [float(x["cost"]) for x in a], [float(x["cost"]) for x in b]
        ta = st.median(float(x["elapsed_seconds"]) for x in a[1:])
        tb = st.median(float(x["elapsed_seconds"]) for x in b[1:])
        row = {"mean": (st.mean(ca), st.mean(cb)), "best": (min(ca), min(cb)), "worst": (max(ca), max(cb)),
               "sd": (st.stdev(ca), st.stdev(cb)), "time": (ta, tb)}
        for k, (x, y) in row.items():
            if k == "sd":
                agg[k].append(y - x)
            else:
                agg[k].append(100 * (y - x) / x)
        ga, gb = conv(ref, inst), conv(var, inst)
        if ga and gb:
            for g in GENS:
                if g in ga and g in gb:
                    gen_agg[g].append(100 * (st.mean(gb[g]) - st.mean(ga[g])) / st.mean(ga[g]))
            target = max(st.mean(ca), st.mean(cb))         # generations to reach the worse of the two final means
            hit_agg.append((first_hit(ref, inst, target), first_hit(var, inst, target)))
        print(f"{inst:8s} mean {row['mean'][0]:9.1f} -> {row['mean'][1]:9.1f} | best {row['best'][0]:9.1f} -> "
              f"{row['best'][1]:9.1f} | worst {row['worst'][0]:9.1f} -> {row['worst'][1]:9.1f} | sd "
              f"{row['sd'][0]:6.1f} -> {row['sd'][1]:6.1f} | s/run {ta:5.0f} -> {tb:5.0f}")
    for k, v in agg.items():
        if v:
            better = sum(1 for x in v if x < -1e-9)
            worse = sum(1 for x in v if x > 1e-9)
            unit = "" if k == "sd" else "%"
            print(f"  {k:6s}: mean diff {st.mean(v):+.2f}{unit}, {var} better {better}, worse {worse}, of {len(v)}")
    for g in GENS:
        v = gen_agg[g]
        if v:
            print(f"  gen {g:4d}: mean diff {st.mean(v):+.2f}%, {var} better {sum(1 for x in v if x < -1e-9)}/"
                  f"{len(v)}, worse {sum(1 for x in v if x > 1e-9)}")
    if hit_agg:
        print(f"  generations to reach the worse final mean: {ref} {st.mean(h[0] for h in hit_agg):.0f}, "
              f"{var} {st.mean(h[1] for h in hit_agg):.0f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:                       # python analyze_extra.py <reference> <variant> [<variant> ...]
        for v in sys.argv[2:]:
            compare(sys.argv[1], v, "custom")
        sys.exit()
    compare("sa_1000", "B", "Section 2 (1000 iterations)")
    compare("sa_100", "grasp_100", "Section 2 (100 iterations)")
    compare("sa_200", "grasp_200", "Section 2 (200 iterations)")
    compare("isl1", "B", "Section 3a (P=30, 6 vs 1 island)")
    compare("isl1", "P30_isl3", "Section 3b (P=30, 3 vs 1 island)")
    compare("isl1", "P30_isl10", "Section 3b (P=30, 10 vs 1 island)")
    compare("P60_isl1", "P60_isl12", "Section 3b (P=60, 12 vs 1 island)")
