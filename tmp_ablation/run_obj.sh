export PYTHONPATH=/d/AI_Research/PH-SHOWOA
inst=$1; it=$2
python -u -m src_obj_baseline.main --problem ../dataset/explicit_$inst.vrpsdptw --compute_backend cpu --init sa_rcrs_grasp --paper_flags --architecture python_cuda --objective lexicographic --runs 30 --max_iter $it --pop_size 30 --workers 1 2>&1 | grep -E "Gen:|finishes|consumed" > obj_${inst}_${it}.txt
echo done > obj_${inst}_${it}.done
