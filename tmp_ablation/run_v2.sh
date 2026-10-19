name=$1; dir=$2; inst=$3
cd "$dir" && env -u PYTHONPATH python -u -m src_python_gpu_SA_RCRS_GRASP.main --problem D:/AI_Research/PH-SHOWOA/dataset/explicit_$inst.vrpsdptw --compute_backend cpu --init sa_rcrs_grasp --paper_flags --architecture python_cuda --runs 30 --max_iter 1000 --pop_size 30 --workers 1 --random_seed 42 > /d/AI_Research/PH-SHOWOA/tmp_ablation/lx_${name}_${inst}.log 2>&1
echo done > /d/AI_Research/PH-SHOWOA/tmp_ablation/lx_${name}_${inst}.done
