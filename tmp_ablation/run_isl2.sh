export PYTHONPATH=/d/AI_Research/PH-SHOWOA
inst=$1; tag=$2; shift 2
python -u -m src_python_gpu_SA_RCRS_GRASP.main --problem ../dataset/explicit_$inst.vrpsdptw --compute_backend cpu --init sa_rcrs_grasp --paper_flags --architecture python_cuda --runs 30 --max_iter 1000 --workers 1 --random_seed 42 "$@" > rawi2_${inst}_${tag}.log 2>&1
grep -E "Gen:|finishes|consumed" rawi2_${inst}_${tag}.log > i2_${inst}_${tag}.txt
echo done > i2_${inst}_${tag}.done
