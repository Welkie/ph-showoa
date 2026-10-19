import statistics as st,math
from paired import load,report,c
from scipy import stats
import glob
rows=[]
def pair(inst,it,seed,Af,Bf):
    A=load(Af);B=load(Bf)
    report(f"{inst} it{it} s{seed}",A,B,it)
    n=min(len(A),len(B))
    return [c(r[it]) for r in A[:n]],[c(r[it]) for r in B[:n]]
res={}
for inst in ["cdp103","rcdp101","rdp103","rcdp205"]:
    for it in (1000,300):
        res[(inst,it,42)]=pair(inst,it,42,f"x_{inst}_{it}_A_s42.txt",f"x_{inst}_{it}_B_s42.txt")
res[("rcdp104",1000,777)]=pair("rcdp104",1000,777,"x_rcdp104_1000_A_s777.txt","x_rcdp104_1000_B_s777.txt")
# pool rcdp104 two seeds
a0=load("obj_rcdp104_1000.txt");b0=load("conv_rcdp104_1000_sa_rcrs_grasp.txt")
a=[c(r[1000]) for r in a0]+res[("rcdp104",1000,777)][0];b=[c(r[1000]) for r in b0]+res[("rcdp104",1000,777)][1]
d=[y-x for x,y in zip(a,b)];n=len(d)
print("rcdp104 pooled 60 runs: A=%.1f±%.1f B=%.1f±%.1f meanD=%.1f t=%.2f mannwhitney_p=%.4f"%(st.mean(a),st.stdev(a),st.mean(b),st.stdev(b),st.mean(d),(st.mean(b)-st.mean(a))/math.sqrt(st.variance(a)/n+st.variance(b)/n),stats.mannwhitneyu(b,a,alternative='less').pvalue))
# unpaired summary for all
print()
for k,(a,b) in res.items():
    n=len(a);se=math.sqrt(st.variance(a)/n+st.variance(b)/n)
    print(k,"A=%.1f±%.1f B=%.1f±%.1f D=%.1f t=%.2f MWU_p(B<A)=%.4f"%(st.mean(a),st.stdev(a),st.mean(b),st.stdev(b),st.mean(b)-st.mean(a),(st.mean(b)-st.mean(a))/se,stats.mannwhitneyu(b,a,alternative='less').pvalue))
