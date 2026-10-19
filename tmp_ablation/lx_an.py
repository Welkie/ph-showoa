import os, re, math, statistics as st
from scipy import stats
from paired import load, c
ALL12=["cdp101","cdp103","cdp201","rcdp101","rcdp104","rcdp202","rcdp205","rcdp207","rdp103","rdp201","rdp203","rdp210"]
def cur_scalar(i): return f"isl_{i}_6.txt" if i in ("cdp101","rcdp104","rdp210") else f"i2_{i}_n6.txt"
def noswap_lex(i):
    if i in ("cdp101","rcdp104","rdp210"): return f"head_{i}_6.txt"
    if i=="cdp103": return "var_commit.log"
    return f"lx_noswap_lex_{i}.log"
def noswap_scalar(i): return "var_obj.log" if i=="cdp103" else f"lx_noswap_scalar_{i}.log"
def block(title, insts, f_lex, f_sca):
    print("==",title,"(A = scalar, B = lexicographic; d>0 => lexicographic better)")
    ds=[]; pooledA=[]; pooledB=[]
    for i in insts:
        fa,fb=f_sca(i),f_lex(i)
        if not(os.path.exists(fa) and os.path.exists(fb)): print("  missing",i,fa,fb); continue
        A=[r[1000] for r in load(fa)]; B=[r[1000] for r in load(fb)]
        if len(A)<30 or len(B)<30: print("  incomplete",i,len(A),len(B)); continue
        ca=[c(x) for x in A]; cb=[c(x) for x in B]; mn=min(x[0] for x in A+B)
        sp=math.sqrt((st.variance(ca)+st.variance(cb))/2); d=(st.mean(ca)-st.mean(cb))/sp if sp>0 else 0
        dd=[y-x for x,y in zip(ca,cb)]
        pw=stats.wilcoxon(dd).pvalue if any(abs(x)>1e-9 for x in dd) else 1.0
        print(f"  {i:8s} scalar {st.mean(ca):9.1f}±{st.stdev(ca):6.1f} hit={sum(x[0]==mn for x in A):2d} | lex {st.mean(cb):9.1f}±{st.stdev(cb):6.1f} hit={sum(x[0]==mn for x in B):2d} | D(lex-sca)={st.mean(cb)-st.mean(ca):8.1f} d={d:+.2f} lex better {sum(x<0 for x in dd)}/30 wilcoxon p={pw:.3f}")
        if sp>0: ds.append(d)
    if len(ds)>1:
        print(f"  -> n={len(ds)} mean d={st.mean(ds):+.3f} lex better in {sum(x>0 for x in ds)}/{len(ds)}  t two-sided p={stats.ttest_1samp(ds,0).pvalue:.4f}  wilcoxon two-sided p={stats.wilcoxon(ds).pvalue:.4f}")
if __name__=="__main__":
    block("Context 1: current code (Swap + global stagnation)", ALL12, lambda i:f"lx_curlex_{i}.log", cur_scalar)
    block("Context 2: no Swap (commit 5ef47ce)", ["cdp101","cdp103","rcdp104","rdp210","rcdp205","rdp203","rcdp207"], noswap_lex, noswap_scalar)
