import subprocess,sys,time
inst=sys.argv[1]
for pkg,init in [("src_sa_baseline","sa"),("src_python_gpu_SA_RCRS_GRASP","sa_rcrs_grasp")]:
    t={}
    for r in (1,6):
        s=time.perf_counter()
        subprocess.run([sys.executable,"-m",pkg+".main","--problem",f"../dataset/explicit_{inst}.vrpsdptw","--compute_backend","cpu","--init",init,"--paper_flags","--architecture","python_cuda","--objective","lexicographic","--runs",str(r),"--max_iter","1","--pop_size","30","--workers","1"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        t[r]=time.perf_counter()-s
    print(inst,init,"1run=%.1fs 6runs=%.1fs per_extra_run=%.2fs"%(t[1],t[6],(t[6]-t[1])/5),flush=True)
