export PYTHONPATH=/d/AI_Research/PH-SHOWOA
inst=$1; isl=$2
python -u -m src_head.main --problem ../dataset/explicit_$inst.vrpsdptw --compute_backend cpu --init sa_rcrs_grasp --paper_flags --architecture python_cuda --runs 30 --max_iter 1000 --pop_size 30 --workers 1 --random_seed 42 --num_islands $isl > rawhead_${inst}_$isl.log 2>&1
grep -E "Gen:|finishes|consumed" rawhead_${inst}_$isl.log > head_${inst}_$isl.txt
echo done > head_${inst}_$isl.done
