export PYTHONPATH=/d/AI_Research/PH-SHOWOA
inst=$1; isl=$2
python -u -m src_python_gpu_SA_RCRS_GRASP.main --problem ../dataset/explicit_$inst.vrpsdptw --compute_backend cpu --init sa_rcrs_grasp --paper_flags --architecture python_cuda --runs 30 --max_iter 1000 --pop_size 30 --workers 1 --random_seed 42 --num_islands $isl > rawisl_${inst}_$isl.log 2>&1
grep -E "Gen:|finishes|consumed" rawisl_${inst}_$isl.log > isl_${inst}_$isl.txt
echo done > isl_${inst}_$isl.done
