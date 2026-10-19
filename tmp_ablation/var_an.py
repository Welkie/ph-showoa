import re, statistics as st
from scipy import stats
names=[("commit","commit 5ef47ce (gốc)"),("swap","+ Inter-route Swap"),("obj","+ hàm vô hướng"),("stag","+ stagnation toàn cục"),("current","code hiện tại (cả 3)")]
def parse(n):
    L=open(f"var_{n}.log",encoding='utf-8',errors='ignore').read()
    runs=[(int(a),float(b)) for a,b in re.findall(r"Run \d+ finishes \| Run Best: NV=(\d+), TD=([\d.]+)",L)]
    w=re.findall(r"wall=(\d+)",L); c=re.findall(r"total consumed (\d+) sec",L)
    gens={}
    cur={}
    for l in L.splitlines():
        m=re.search(r"Gen: (\d+)\..*Best NV (\d+), Best TD ([\d.]+)",l)
        if m: cur[int(m.group(1))]=(int(m.group(2)),float(m.group(3)))
        if "finishes" in l: gens.setdefault('runs',[]).append(cur); cur={}
    return runs, (int(w[-1]) if w else None), (int(c[-1]) if c else None), gens.get('runs',[])
base=None
for n,label in names:
    runs,wall,cons,tr=parse(n)
    td=[b for _,b in runs]
    if not td: print(label,"no data"); continue
    s=f"{label:28s} n={len(runs)} NV={sorted(set(a for a,_ in runs))} TD best={min(td):8.2f} mean={st.mean(td):8.2f} sd={st.stdev(td):6.2f} median={st.median(td):8.2f} | time={cons}s"
    if n=="commit": base=td
    elif base and len(td)==len(base):
        d=[y-x for x,y in zip(base,td)]
        s+=f" | vs commit: meanΔ={st.mean(d):+7.2f} better {sum(x<0 for x in d)}/30 wilcoxon p={stats.wilcoxon(d).pvalue:.3f}"
    print(s)
