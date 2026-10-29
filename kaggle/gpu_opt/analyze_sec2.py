"""Section 2 ablation (docs/improved.md): every initialization variant against --init sa (Algorithm 2).

usage: python kaggle/gpu_opt/analyze_sec2.py <results dir> <reference> <variant> [<variant> ...]
Per instance: cost mean +- sd, NV distribution, d = (mean ref - mean var) / pooled sd (d > 0: variant better) and
the two-sided Mann-Whitney p. Summary: mean relative difference, better / worse counts (p < 0.05), mean d with a
one-sample t-test and Wilcoxon over the instances, median seconds per run. Environment RUNS: use only the first RUNS
runs of every file (the same seeds 42.. for a shorter trial).
"""
import collections
import csv
import os
import statistics as st
import sys

from scipy.stats import mannwhitneyu, ttest_1samp, wilcoxon

ALL12 = ["cdp101", "cdp103", "cdp203", "cdp207", "rdp103", "rdp107", "rdp201", "rdp210",
         "rcdp101", "rcdp104", "rcdp202", "rcdp207"]
res, ref, variants = sys.argv[1], sys.argv[2], sys.argv[3:]


def load(name, inst):
    p = os.path.join(res, f"{name}__{inst}.csv")
    return list(csv.DictReader(open(p)))[:int(os.environ.get("RUNS", "30"))] if os.path.exists(p) else None


def nvs(rows):
    c = collections.Counter(int(r["NV"]) for r in rows)
    return ", ".join(f"{k}: {v}" for k, v in sorted(c.items()))


for var in variants:
    print(f"\n=== {var} vs {ref} (d > 0: {var} better) ===")
    ds, rel, better, worse, times = [], [], 0, 0, []
    for inst in ALL12:
        a, b = load(ref, inst), load(var, inst)
        if not a or not b:
            continue
        ca, cb = [float(r["cost"]) for r in a], [float(r["cost"]) for r in b]
        sp = ((st.variance(ca) + st.variance(cb)) / 2) ** 0.5
        d = (st.mean(ca) - st.mean(cb)) / sp if sp > 0 else 0.0
        p = mannwhitneyu(ca, cb).pvalue if ca != cb else 1.0
        ds.append(d)
        rel.append(100 * (st.mean(cb) - st.mean(ca)) / st.mean(ca))
        if p < 0.05:
            better += d > 0
            worse += d < 0
        ta = st.median(float(r["elapsed_seconds"]) for r in a)
        tb = st.median(float(r["elapsed_seconds"]) for r in b)
        times.append((ta, tb))
        print(f"{inst:8s} {st.mean(ca):9.1f} +- {st.stdev(ca):6.1f} ({nvs(a)}) | {st.mean(cb):9.1f} +- "
              f"{st.stdev(cb):6.1f} ({nvs(b)}) | d {d:+.2f} p {p:.3f} | s/run {ta:.1f} -> {tb:.1f}")
    if ds:
        tp = ttest_1samp(ds, 0).pvalue
        wp = wilcoxon(ds).pvalue if any(abs(x) > 1e-12 for x in ds) else 1.0
        print(f"  mean diff {st.mean(rel):+.2f}% | significant better {better}, worse {worse} of {len(ds)} | "
              f"mean d {st.mean(ds):+.2f} (t p={tp:.3f}, Wilcoxon p={wp:.3f}) | "
              f"time {sum(t[0] for t in times):.0f}s -> {sum(t[1] for t in times):.0f}s")
