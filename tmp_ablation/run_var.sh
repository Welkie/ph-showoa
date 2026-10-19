name=$1; dir=$2
cd "$dir" && start=$(date +%s) && env -u PYTHONPATH python -u -m src_python_gpu_SA_RCRS_GRASP.main --problem D:/AI_Research/PH-SHOWOA/dataset/explicit_cdp103.vrpsdptw --compute_backend cpu --init sa_rcrs_grasp --paper_flags --architecture python_cuda --runs 30 --max_iter 1000 --pop_size 30 --workers 1 > /d/AI_Research/PH-SHOWOA/tmp_ablation/var_${name}.log 2>&1
end=$(date +%s); echo "wall=$((end-start))" >> /d/AI_Research/PH-SHOWOA/tmp_ablation/var_${name}.log
echo done > /d/AI_Research/PH-SHOWOA/tmp_ablation/var_${name}.done
