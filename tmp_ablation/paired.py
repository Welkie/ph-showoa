import re,sys,math,statistics as st
try:
    from scipy import stats
except Exception: stats=None
def load(f):
    runs=[];cur={}
    for l in open(f,encoding='utf-8',errors='ignore'):
        m=re.search(r"Gen: (\d+)\..*Best NV (\d+), Best TD ([\d.]+)",l)
        if m: cur[int(m.group(1))]=(int(m.group(2)),float(m.group(3)))
        if "finishes" in l: runs.append(cur);cur={}
    return runs
c=lambda v:2000*v[0]+v[1]
def report(name,A,B,g):
    n=min(len(A),len(B));A,B=A[:n],B[:n]
    a=[c(r[g]) for r in A];b=[c(r[g]) for r in B]
    d=[y-x for x,y in zip(a,b)]  # B-A, negative = B better
    md=st.mean(d);sd=st.stdev(d);t=md/(sd/math.sqrt(n)) if sd>0 else float('nan')
    wins=sum(x<-1e-3 for x in d);loss=sum(x>1e-3 for x in d);tie=n-wins-loss
    nva=st.mean(r[g][0] for r in A);nvb=st.mean(r[g][0] for r in B)
    p=''
    if stats:
        try: p=" wilcoxon_p=%.4f"%stats.wilcoxon(d).pvalue
        except Exception: p=" wilcoxon_p=NA"
        sp=stats.ttest_rel(b,a).pvalue; p+=" paired_t_p=%.4f"%sp
        # sign test on non-ties
        k=wins;m=wins+loss
        if m>0: p+=" sign_p=%.4f"%stats.binomtest(k,m,0.5).pvalue
    print(f"{name} gen{g} n={n}: meanD={md:8.1f} t={t:5.2f} B wins/ties/loss={wins}/{tie}/{loss} NV A={nva:.2f} B={nvb:.2f}{p}")
if __name__=="__main__":
    for inst in ["cdp101","rcdp104","rdp210"]:
        for it in (1000,200):
            A=load(f"obj_{inst}_{it}.txt");B=load(f"conv_{inst}_{it}_sa_rcrs_grasp.txt")
            report(f"{inst} it{it}",A,B,it)
