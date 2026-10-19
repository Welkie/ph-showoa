import os, math, statistics as st
from scipy import stats
from paired import load, c
ALL12=["cdp101","cdp103","cdp201","rcdp101","rcdp104","rcdp202","rcdp205","rcdp207","rdp103","rdp201","rdp203","rdp210"]
def f_global(inst,tag):
    if tag=="n6":
        return f"isl_{inst}_6.txt" if inst in ("cdp101","rcdp104","rdp210") else f"i2_{inst}_n6.txt"
    if tag=="k10": return f"i2_{inst}_k10.txt"
    if tag=="p60": return f"i2_{inst}_p60i12.txt"
    if tag=="p120": return f"st_G_{inst}_p120.txt"
def f_island(inst,tag):
    if tag=="n6":
        if inst in ("cdp101","rcdp104","rdp210"): return f"obj_{inst}_1000.txt"
        if inst in ("cdp103","rcdp101","rdp103","rcdp205"): return f"x_{inst}_1000_A_s42.txt"
    return f"st_I_{inst}_{tag}.txt"
def final(f): 
    R=load(f); return [r[1000] for r in R]
def block(tag,insts):
    print(f"== {tag}  (A = global stagnation [current code], B = per-island stagnation)")
    ds=[]
    for i in insts:
        fa,fb=f_global(i,tag),f_island(i,tag)
        if not(os.path.exists(fa) and os.path.exists(fb)): print("  missing",i,fa,fb); continue
        A=final(fa);B=final(fb)
        if len(A)<30 or len(B)<30: print("  incomplete",i,len(A),len(B)); continue
        ca=[c(x) for x in A];cb=[c(x) for x in B]
        mn=min(x[0] for x in A+B)
        sp=math.sqrt((st.variance(ca)+st.variance(cb))/2)
        d=(st.mean(ca)-st.mean(cb))/sp if sp>0 else 0.0
        p=stats.mannwhitneyu(cb,ca,alternative='less').pvalue if sp>0 else 1.0
        print(f"  {i:8s} A {st.mean(ca):9.1f}±{st.stdev(ca):6.1f} hit={sum(x[0]==mn for x in A):2d} | B {st.mean(cb):9.1f}±{st.stdev(cb):6.1f} hit={sum(x[0]==mn for x in B):2d} | D(B-A)={st.mean(cb)-st.mean(ca):7.1f} d={d:+.2f} p(B<A)={p:.3f}")
        if sp>0: ds.append(d)
    if len(ds)>1:
        print(f"  -> mean d={st.mean(ds):+.3f} wins(B)={sum(x>0 for x in ds)}/{len(ds)} t p(d>0)={stats.ttest_1samp(ds,0,alternative='greater').pvalue:.3f} two-sided p={stats.ttest_1samp(ds,0).pvalue:.3f}")
    return ds
if __name__=="__main__":
    block("n6",ALL12)
    block("k10",["cdp103","rcdp205","rdp203","rcdp104","rcdp207"])
    block("p60",["cdp103","rcdp205","rdp203","rcdp104","rcdp207"])
    block("p120",["rcdp104","rcdp207","cdp103"])
