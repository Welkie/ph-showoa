export PYTHONPATH=/d/AI_Research/PH-SHOWOA
for inst in rcdp101 rcdp202; do
 for cfg in "src_sa_baseline sa" "src_python_gpu_SA_RCRS_GRASP sa_rcrs_grasp"; do
  set -- $cfg
  python -u -m $1.main --problem ../dataset/explicit_$inst.vrpsdptw --compute_backend cpu --init $2 --paper_flags --architecture python_cuda --objective lexicographic --runs 3 --max_iter 1000 --pop_size 30 --workers 1 2>&1 | grep -v "^route\|Worker" > log_${inst}_$2.txt
 done
done
echo DONE > done.txt
