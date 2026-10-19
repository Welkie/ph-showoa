import subprocess, sys, time, os
env=dict(os.environ, PYTHONPATH=r"D:\AI_Research\PH-SHOWOA")
cfgs=[("current","src_python_gpu_SA_RCRS_GRASP"),("best_only","src_ls_best"),("no_2opt*","src_ls_no2opt")]
for inst in ["rdp210","rcdp104"]:
    for name,pkg in cfgs:
        t={}
        for r in (1,4):
            s=time.perf_counter()
            subprocess.run([sys.executable,"-m",pkg+".main","--problem",f"../dataset/explicit_{inst}.vrpsdptw","--compute_backend","cpu","--init","sa_rcrs_grasp","--paper_flags","--architecture","python_cuda","--runs",str(r),"--max_iter","1000","--pop_size","30","--workers","1","--random_seed","42"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env)
            t[r]=time.perf_counter()-s
        print(f"{inst:8s} {name:10s} 1run={t[1]:6.1f}s 4runs={t[4]:6.1f}s  per-run={(t[4]-t[1])/3:6.2f}s",flush=True)
