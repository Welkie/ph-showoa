export PYTHONPATH=/d/AI_Research/PH-SHOWOA
inst=$1; it=$2
for cfg in "src_sa_baseline sa" "src_python_gpu_SA_RCRS_GRASP sa_rcrs_grasp"; do
  set -- $cfg
  python -u -m $1.main --problem ../dataset/explicit_$inst.vrpsdptw --compute_backend cpu --init $2 --paper_flags --architecture python_cuda --objective lexicographic --runs 30 --max_iter $it --pop_size 30 --workers 1 2>&1 | grep -E "Gen:|finishes|consumed" > conv_${inst}_${it}_$2.txt
done
echo done > conv_${inst}_${it}.done
