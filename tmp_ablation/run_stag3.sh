export PYTHONPATH=/d/AI_Research/PH-SHOWOA
inst=$1; pkg=$2; tag=$3; cfg=$4; shift 4
python -u -m $pkg.main --problem ../dataset/explicit_$inst.vrpsdptw --compute_backend cpu --init sa_rcrs_grasp --paper_flags --architecture python_cuda --runs 30 --max_iter 1000 --workers 1 --random_seed 42 "$@" > raw_${inst}_${cfg}_${tag}.log 2>&1
grep -E "Gen:|finishes|consumed" raw_${inst}_${cfg}_${tag}.log > s2_${inst}_${cfg}_${tag}.txt
echo done > s3_${inst}_${cfg}_${tag}.done
