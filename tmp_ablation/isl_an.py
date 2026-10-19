from paired import load,c
from scipy import stats
import statistics as st, math, os
def get(fname,it=1000):
    return [r[it] for r in load(fname)]
sets={}
for inst in ["cdp101","rcdp104","rdp210"]:
    sets[inst]=(f"isl_{inst}_1.txt",f"isl_{inst}_6.txt")
for inst in ["cdp103","cdp201","rcdp101","rcdp202","rcdp205","rcdp207","rdp103","rdp201","rdp203"]:
    sets[inst]=(f"i2_{inst}_n1.txt",f"i2_{inst}_n6.txt")
allA=[];allB=[];rows=[]
for inst,(f1,f6) in sets.items():
    if not(os.path.exists(f1) and os.path.exists(f6)): continue
    A=get(f1);B=get(f6)
    ca=[c(x) for x in A];cb=[c(x) for x in B]
    mn=min(x[0] for x in A+B)
    se=math.sqrt(st.variance(ca)/len(ca)+st.variance(cb)/len(cb))
    p=stats.mannwhitneyu(cb,ca,alternative='less').pvalue
    pt=stats.mannwhitneyu(cb,ca,alternative='greater').pvalue
    print(f"{inst:8s} 1isl {st.mean(ca):9.1f}±{st.stdev(ca):6.1f} NV={st.mean(x[0] for x in A):5.2f} hit={sum(x[0]==mn for x in A):2d} | 6isl {st.mean(cb):9.1f}±{st.stdev(cb):6.1f} NV={st.mean(x[0] for x in B):5.2f} hit={sum(x[0]==mn for x in B):2d} | D={st.mean(cb)-st.mean(ca):7.1f} t={((st.mean(cb)-st.mean(ca))/se if se>0 else 0):5.2f} p(6<1)={p:.3f} p(6>1)={pt:.3f}")
    # z-score normalised pooled
    s=st.stdev(ca+cb) or 1.0
    allA+=[(x-st.mean(ca+cb))/s for x in ca];allB+=[(x-st.mean(ca+cb))/s for x in cb]
    rows.append(st.mean(cb)-st.mean(ca))
print("instances:",len(rows)," 6isl better in",sum(r<0 for r in rows))
print("pooled standardized: mean 6isl=%.3f 1isl=%.3f MWU p(6<1)=%.4f"%(st.mean(allB),st.mean(allA),stats.mannwhitneyu(allB,allA,alternative='less').pvalue))
print("sign test p=%.3f"%stats.binomtest(sum(r<0 for r in rows),len(rows),0.5).pvalue)
