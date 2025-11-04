
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Force NumPy to use lower precision to reduce shape mismatches
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0



for problem in cube sphere spiral obelisk; do
# for problem in cube; do
# for problem in obelisk; do
    echo "Testing problem: $problem"
    # taskset -c 0 python test2.py --problem $problem.pickle
    taskset -c 0 python bundle_adjustment.py --problem $problem.pickle
    taskset -c 0 python eval_reconstruction.py --solution $problem-solution.pickle
done
