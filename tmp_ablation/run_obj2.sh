export PYTHONPATH=/d/AI_Research/PH-SHOWOA
inst=$1; it=$2; pkg=$3; tag=$4; seed=$5
python -u -m $pkg.main --problem ../dataset/explicit_$inst.vrpsdptw --compute_backend cpu --init sa_rcrs_grasp --paper_flags --architecture python_cuda --objective lexicographic --runs 30 --max_iter $it --pop_size 30 --workers 1 --random_seed $seed 2>&1 | grep -E "Gen:|finishes|consumed" > x_${inst}_${it}_${tag}_s${seed}.txt
echo done > x_${inst}_${it}_${tag}_s${seed}.done
