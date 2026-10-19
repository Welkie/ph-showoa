import re,statistics as st,math,sys,os
def load(f):
    runs=[];cur={}
    for l in open(f,encoding='utf-8',errors='ignore'):
        m=re.search(r"Gen: (\d+)\..*Best NV (\d+), Best TD ([\d.]+)",l)
        if m: cur[int(m.group(1))]=(int(m.group(2)),float(m.group(3)))
        if "finishes" in l:
            runs.append(cur);cur={}
    return runs
def cost(v): return 2000*v[0]+v[1]
for inst in ["cdp101","rcdp104","rdp210"]:
  for it in [1000,200,100]:
    A=load(f"conv_{inst}_{it}_sa.txt");B=load(f"conv_{inst}_{it}_sa_rcrs_grasp.txt")
    n=min(len(A),len(B))
    if n==0: continue
    A,B=A[:n],B[:n]
    gens=sorted(set(A[0])&set(B[0]))
    pick=[g for g in gens if g in (1,25,50,100,200,500,1000) and g<=it]
    print(f"== {inst} max_iter={it} n={n}")
    for g in pick:
        ca=[cost(r[g]) for r in A];cb=[cost(r[g]) for r in B]
        na=st.mean(r[g][0] for r in A);nb=st.mean(r[g][0] for r in B)
        se=math.sqrt(st.variance(ca)/n+st.variance(cb)/n) if n>1 else float('nan')
        t=(st.mean(cb)-st.mean(ca))/se if se>0 else float('nan')
        print(f" gen {g:4d}: SA NV={na:5.2f} cost={st.mean(ca):9.1f}±{st.stdev(ca):6.1f} | ours NV={nb:5.2f} cost={st.mean(cb):9.1f}±{st.stdev(cb):6.1f} | diff={st.mean(cb)-st.mean(ca):8.1f} t={t:5.2f}")
