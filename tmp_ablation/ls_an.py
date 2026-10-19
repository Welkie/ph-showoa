import os, math, statistics as st, re
from scipy import stats
from paired import load, c
ALL12=["cdp101","cdp103","cdp201","rcdp101","rcdp104","rcdp202","rcdp205","rcdp207","rdp103","rdp201","rdp203","rdp210"]
def cur(i): return f"isl_{i}_6.txt" if i in ("cdp101","rcdp104","rdp210") else f"i2_{i}_n6.txt"
def secs(f):
    m=re.findall(r"consumed (\d+) sec",open(f,encoding='utf-8',errors='ignore').read()); return int(m[-1]) if m else None
def compare(var,label,gens=(1000,)):
    print(f"== {label}: A = variant ({var}), B = current code (LS all P, 5 operators); d>0 => current better")
    for g in gens:
        ds=[]
        for i in ALL12:
            fa=f"st_{var}_{i}_n6.txt"; fb=cur(i)
            if not os.path.exists(fa): print("  missing",i); continue
            A=[r[g] for r in load(fa)]; B=[r[g] for r in load(fb)]
            if len(A)<30: print("  incomplete",i,len(A)); continue
            ca=[c(x) for x in A]; cb=[c(x) for x in B]
            mn=min(x[0] for x in A+B)
            sp=math.sqrt((st.variance(ca)+st.variance(cb))/2)
            d=(st.mean(ca)-st.mean(cb))/sp if sp>0 else 0.0
            p=stats.mannwhitneyu(cb,ca,alternative='less').pvalue if sp>0 else 1.0
            if g==1000:
                print(f"  {i:8s} A {st.mean(ca):9.1f}±{st.stdev(ca):6.1f} NV={st.mean(x[0] for x in A):5.2f} hit={sum(x[0]==mn for x in A):2d} | B {st.mean(cb):9.1f}±{st.stdev(cb):6.1f} NV={st.mean(x[0] for x in B):5.2f} hit={sum(x[0]==mn for x in B):2d} | D(B-A)={st.mean(cb)-st.mean(ca):8.1f} d={d:+.2f} p(B<A)={p:.3f} | time A={secs(fa)}s B={secs(fb)}s")
            if sp>0: ds.append(d)
        if len(ds)>1:
            print(f"  -> gen {g}: mean d={st.mean(ds):+.3f} current better in {sum(x>0 for x in ds)}/{len(ds)}  t p(d>0)={stats.ttest_1samp(ds,0,alternative='greater').pvalue:.4f}  wilcoxon p={stats.wilcoxon(ds,alternative='greater').pvalue:.4f}")
if __name__=="__main__":
    compare("LSB","(1) LS only on best individual",gens=(1000,100,250,500))
    compare("NO2","(2) LS on all P without Inter 2-opt*",gens=(1000,100,250,500))
