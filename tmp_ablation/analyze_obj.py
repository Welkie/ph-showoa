import re,statistics as st,math
def load(f):
    runs=[];cur={};fin=[]
    for l in open(f,encoding='utf-8',errors='ignore'):
        m=re.search(r"Gen: (\d+)\..*Best NV (\d+), Best TD ([\d.]+)",l)
        if m: cur[int(m.group(1))]=(int(m.group(2)),float(m.group(3)))
        if "finishes" in l: runs.append(cur);cur={}
    return runs
c=lambda v:2000*v[0]+v[1]
for inst in ["cdp101","rcdp104","rdp210"]:
  for it in [1000,200]:
    B=load(f"obj_{inst}_{it}.txt");L=load(f"conv_{inst}_{it}_sa_rcrs_grasp.txt");n=min(len(B),len(L));B,L=B[:n],L[:n]
    print(f"== {inst} max_iter={it} n={n}")
    for g in [g for g in (1,25,50,100,200,500,1000) if g<=it]:
        cb=[c(r[g]) for r in B];cl=[c(r[g]) for r in L]
        nb=[r[g][0] for r in B];nl=[r[g][0] for r in L]
        se=math.sqrt(st.variance(cb)/n+st.variance(cl)/n);t=(st.mean(cl)-st.mean(cb))/se if se>0 else float('nan')
        mn=min(nb+nl)
        print(f" gen {g:4d}: baseline NV={st.mean(nb):5.2f}(min {nb.count(mn)}/{n}) cost={st.mean(cb):9.1f}±{st.stdev(cb):6.1f} best={min(cb):9.1f} | lex NV={st.mean(nl):5.2f}(min {nl.count(mn)}/{n}) cost={st.mean(cl):9.1f}±{st.stdev(cl):6.1f} best={min(cl):9.1f} | diff(lex-base)={st.mean(cl)-st.mean(cb):8.1f} t={t:5.2f}")
