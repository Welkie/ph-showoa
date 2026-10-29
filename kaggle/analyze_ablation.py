"""Summarise the ablation results (docs/improved.md sections 2 and 3)."""
import csv, os, statistics as st
import numpy as np
from scipy import stats

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_ablation", "results")
ALL12 = ["cdp101", "cdp103", "cdp203", "cdp207", "rdp103", "rdp107", "rdp201", "rdp210",
         "rcdp101", "rcdp104", "rcdp202", "rcdp207"]
GROUP = {"cdp1": "Cdp1", "cdp2": "Cdp2", "rdp1": "Rdp1", "rdp2": "Rdp2", "rcdp1": "RCdp1", "rcdp2": "RCdp2"}
VARIANTS = ["sa", "A123"]      # −mục 2, −mục 3 (A1, A2, A3 of the earlier runs are not reported)

def load(name, inst):
    p = os.path.join(RES, f"{name}__{inst}.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    assert all(r["feasible"] == "True" for r in rows), (name, inst)
    return rows

def costs(name, inst):
    rows = load(name, inst)
    return None if rows is None else np.array([float(r["cost"]) for r in rows])

def conv(name, inst, gen):
    p = os.path.join(RES, f"{name}__{inst}__conv.csv")
    if not os.path.exists(p):
        return None
    return np.array([float(r["cost"]) for r in csv.DictReader(open(p)) if int(r["iteration"]) == gen])

def nv_hist(name, inst):
    rows = load(name, inst)
    c = {}
    for r in rows:
        c[int(r["NV"])] = c.get(int(r["NV"]), 0) + 1
    return " ".join(f"{k}:{v}" for k, v in sorted(c.items()))

def chunk_starts(name, inst):
    """First run of every chunk of (name, inst) but run 1: its time includes a second JIT compile."""
    import orchestrate_ablation as o
    return {j["run_lo"] for b in o.build_jobs() for j in b["jobs"] if j["name"] == name and j["inst"] == inst} - {1}

def runtime(name, inst):
    """Mean seconds per run as for the baseline (30 runs in one process, the JIT compile in run 1): the first run
    of every other chunk is left out, as its JIT compile would not happen in a single process."""
    skip = chunk_starts(name, inst)
    return st.mean(float(r["elapsed_seconds"]) for r in load(name, inst) if int(r["run"]) not in skip)

def d_eff(var, ref):
    """(mean variant - mean reference) / pooled sd; > 0 means the reference (current code) is better."""
    sp = np.sqrt((var.var(ddof=1) + ref.var(ddof=1)) / 2)
    return 0.0 if sp == 0 else (var.mean() - ref.mean()) / sp

def mw_less(ref, var):
    if np.all(ref == ref[0]) and np.all(var == ref[0]):
        return float("nan")
    return stats.mannwhitneyu(ref, var, alternative="less").pvalue

def fmt(x):
    return f"{x.mean():.1f} ± {x.std(ddof=1):.1f}"

def pval(p):
    return "—" if p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")

def summary(ds, label):
    ds = [d for d in ds if d is not None]
    if len(ds) < 2:
        return
    t = stats.ttest_1samp(ds, 0, alternative="greater").pvalue
    try:
        w = stats.wilcoxon(ds, alternative="greater").pvalue
    except ValueError:
        w = float("nan")
    print(f"| {label} | {np.mean(ds):+.2f} | {sum(d > 0 for d in ds)}/{len(ds)} | {pval(t)} | {pval(w)} |")

def compare(ref_name, var_names, insts, title):
    print(f"\n### {title}\n")
    hdr = "| Instance | " + ref_name + " | " + " | ".join(f"{v} | d | p" for v in var_names) + " |"
    print(hdr); print("|" + "---|" * (1 + 3 * len(var_names) + 1))
    all_d = {v: [] for v in var_names}
    for i in insts:
        ref = costs(ref_name, i)
        if ref is None:
            continue
        cells = [i, fmt(ref)]
        for v in var_names:
            x = costs(v, i)
            if x is None:
                cells += ["(chưa có)", "", ""]; all_d[v].append(None); continue
            no_var = np.all(ref == ref[0]) and np.all(x == ref[0])
            d = None if no_var else d_eff(x, ref)
            all_d[v].append(d)
            cells += [fmt(x), "—" if d is None else f"{d:+.2f}", pval(mw_less(ref, x))]
        print("| " + " | ".join(cells) + " |")
    print("\n| So sánh | d TB | hiện tại tốt hơn | p (t một mẫu) | p (Wilcoxon) |\n|---|---|---|---|---|")
    for v in var_names:
        summary(all_d[v], f"{ref_name} vs {v}")
    return all_d

def holm(ps):
    """Holm-adjusted p-values."""
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj

def main_table():
    """The ablation table of docs/improved.md: cost mean (NV mean) per instance, + / - when fullcaitienGPU (B) is
    significantly better / worse (two-sided Mann-Whitney, Holm over the 12 instances), then the summary rows
    (Wilcoxon on the 12 pairs of means; Levene (median) with Holm for the standard deviation)."""
    names = {"sa": "−mục 2", "A123": "−mục 3"}
    print("| Nhóm | Instance | fullcaitienGPU | " + " | ".join(names[v] for v in VARIANTS) + " |")
    print("|---|---|---|" + "---|" * len(VARIANTS))
    mark, smark, rel, pairs = {}, {}, {}, {}
    for v in VARIANTS:
        mw, lev, d, dsd = [], [], [], []
        for i in ALL12:
            b, x = costs("B", i), costs(v, i)
            same = np.all(b == b[0]) and np.all(x == b[0])
            mw.append(1.0 if same else stats.mannwhitneyu(b, x, alternative="two-sided").pvalue)
            lev.append(1.0 if same else stats.levene(b, x, center="median").pvalue)
            d.append(np.sign(x.mean() - b.mean()))
            dsd.append(np.sign(x.std(ddof=1) - b.std(ddof=1)))
        mark[v] = ["+" if q < 0.05 and s > 0 else "−" if q < 0.05 and s < 0 else "" for q, s in zip(holm(mw), d)]
        smark[v] = ["+" if q < 0.05 and s > 0 else "−" if q < 0.05 and s < 0 else "" for q, s in zip(holm(lev), dsd)]
        rel[v] = [(costs(v, i).mean() - costs("B", i).mean()) / costs("B", i).mean() * 100 for i in ALL12]
        pairs[v] = np.array([(costs("B", i).mean(), costs(v, i).mean()) for i in ALL12]).T
    nv = lambda n, i: np.mean([int(r["NV"]) for r in load(n, i)])
    for k, i in enumerate(ALL12):
        cells = [GROUP[i[:-2]], i, f"{costs('B', i).mean():.1f} ({nv('B', i):.3g})"]
        cells += [f"{costs(v, i).mean():.1f} ({nv(v, i):.3g})" + (f" {mark[v][k]}" if mark[v][k] else "") for v in VARIANTS]
        print("| " + " | ".join(cells) + " |")
    count = lambda m: f"{m.count('+')} / {len(m) - m.count('+') - m.count('−')} / {m.count('−')}"
    row = lambda label, b, f: print(f"| | **{label}** | {b} | " + " | ".join(f(v) for v in VARIANTS) + " |")
    row("ΔCost TB so với fullcaitienGPU", "0", lambda v: f"{np.mean(rel[v]):+.2f}%")
    row("fullcaitienGPU tốt hơn / ngang / kém hơn", "—", lambda v: count(mark[v]))
    row("$p$ (Wilcoxon, 12 instance)", "—", lambda v: pval(stats.wilcoxon(*pairs[v]).pvalue))
    row("sd nhỏ hơn / ngang / lớn hơn", "—", lambda v: count(smark[v]))
    tmean = lambda n: f"{np.mean([runtime(n, i) for i in ALL12]):.0f}"
    row("Thời gian TB (s/run)", tmean("B"), tmean)

if __name__ == "__main__":
    main_table()
    print("\n## NV distribution")
    for i in ALL12:
        print(i, " | ".join(f"{n}: {nv_hist(n, i)}" for n in ["B"] + VARIANTS if load(n, i)))
    print("\n## Mean time per run (s), JIT compile once (run 1)")
    for n in ["B"] + VARIANTS:
        print(n, " ".join(f"{i}:{runtime(n, i):.0f}" for i in ALL12 if load(n, i)))
