export PYTHONPATH=/d/AI_Research/PH-SHOWOA
pkg=$1; var=$2; inst=$3; tag=$4; shift 4
python -u -m $pkg.main --problem ../dataset/explicit_$inst.vrpsdptw --compute_backend cpu --init sa_rcrs_grasp --paper_flags --architecture python_cuda --runs 30 --max_iter 1000 --workers 1 --random_seed 42 "$@" > rawst_${var}_${inst}_${tag}.log 2>&1
grep -E "Gen:|finishes|consumed" rawst_${var}_${inst}_${tag}.log > st_${var}_${inst}_${tag}.txt
echo done > st_${var}_${inst}_${tag}.done
