import io, os, shutil, subprocess
root=r"D:\AI_Research\PH-SHOWOA"
c5=subprocess.run(["git","-C",root,"show","5ef47ce:src_python_gpu_SA_RCRS_GRASP/gpu_kernels.py"],capture_output=True).stdout.decode('utf-8').replace('\r\n','\n')
bsr=open(os.path.join(root,"tmp_ablation","gpu_kernels.before_stag_revert.py"),encoding='utf-8').read().replace('\r\n','\n')
cur=open(os.path.join(root,"src_python_gpu_SA_RCRS_GRASP","gpu_kernels.py"),encoding='utf-8').read().replace('\r\n','\n')

# swap block = text from "# 5. Inter-route Swap" up to (not incl.) the final "if not improved" of deep LS
def swap_block(src):
    a=src.index("            # 5. Inter-route Swap (exchange one customer between two routes)\n")
    b=src.index("            if not improved:\n                break\n",a)
    return a,b
a,b=swap_block(bsr); SWAP=bsr[a:b]

LEX_BETTER='''        if nr_a < nr_b:
            return True
        if nr_a == nr_b and dist_a < dist_b - 1e-3:
            return True
        return False
'''
SCAL_BETTER='''        # Scalar objective: Cost = 2000*NV + TD
        return (2000.0 * nr_a + dist_a) < (2000.0 * nr_b + dist_b) - 1e-3
'''
LEX_ACC='''        if new_nr < old_nr:
            accepted = True
        elif new_nr == old_nr:
            delta = new_dist - old_dist
            if delta <= 1e-3:
                accepted = True
            else:
                temp = 1.0 - float(iteration) / float(max_iter) if max_iter > 0 else 0.0
                prob = math.exp(-delta / (1e-6 + temp * abs(old_dist)))
                if rand_u01(rng_states, s) < prob:
                    accepted = True
'''
SCAL_ACC='''        # Scalar Cost acceptance: P = exp(-dCost / (1e-6 + T*|Cost|))
        old_c = 2000.0 * old_nr + old_dist
        new_c = 2000.0 * new_nr + new_dist
        delta = new_c - old_c
        if delta <= 1e-3:
            accepted = True
        else:
            temp = 1.0 - float(iteration) / float(max_iter) if max_iter > 0 else 0.0
            prob = math.exp(-delta / (1e-6 + temp * abs(old_c)))
            if rand_u01(rng_states, s) < prob:
                accepted = True
'''
def one(s,old,new):
    assert s.count(old)==1, (s.count(old), old[:60]); return s.replace(old,new)

# V_swap: commit + swap
i=c5.index("            if not improved:\n                break\n", c5.index("            # 4. Inter-route Relocate"))
v_swap=c5[:i]+SWAP+c5[i:]
# V_obj: commit + scalar objective
v_obj=one(one(c5,LEX_BETTER,SCAL_BETTER),LEX_ACC,SCAL_ACC)
# V_stag: current minus swap, minus scalar objective  ( = commit + global stagnation )
a,b=swap_block(cur)
v_stag=cur[:a]+cur[b:]
v_stag=one(one(v_stag,SCAL_BETTER,LEX_BETTER),SCAL_ACC,LEX_ACC)

for name,src in (("v_swap",v_swap),("v_obj",v_obj),("v_stag",v_stag)):
    d=os.path.join(root,"tmp_ablation","var_"+name)
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    shutil.copytree(os.path.join(root,"tmp_ablation","commit_5ef47ce","src_python_gpu_SA_RCRS_GRASP"),os.path.join(d,"src_python_gpu_SA_RCRS_GRASP"),ignore=shutil.ignore_patterns("__pycache__"))
    open(os.path.join(d,"src_python_gpu_SA_RCRS_GRASP","gpu_kernels.py"),"w",encoding='utf-8',newline='\n').write(src)
    print("built",name)
