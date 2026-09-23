# Full-run CUDA Graph

The CUDA backend builds and instantiates one fixed-length graph before the
first run, then calls `cuGraphLaunch` once per run. It does not launch kernels,
calculate search parameters, or read NV/TD from Python between generations.
The CPU reference backend is separate; CUDA errors never fall back to it.

## Execution

1. CPU: read the problem, allocate problem/population/scratch/RNG/log buffers,
   compile and load kernels, capture and instantiate the graph.
2. Before each run: copy one seed scalar to the existing device buffer.
3. GPU: initialize RNG, reset bests and generation, then run RCRS-GRASP and
   the base-aligned configurable SA initialization.
4. GPU, for each generation: calculate parameters, SHO/WOA and acceptance,
   periodic route elimination plus combined local search on the global best,
   stagnation-triggered diversification, retained island migration, best
   tracking, logging and generation increment.
5. CPU, after completion: copy the run best and log, print buffered progress,
   decode, check feasibility and update the best across valid runs.
6. CPU: output and verify the final best.

The bounded generation loop is **unrolled into a DAG during preparation**.
This is not a CUDA conditional `WHILE` graph, and requires neither conditional
graph nodes nor a persistent kernel with grid-wide barriers. During replay,
device predicates check the fixed periodic schedules. A single CUDA stream
orders all graph nodes; it preserves the original order when multiple
scheduled operations coincide. Ping-pong population pointers are wired into
the graph, including odd/even generation parity.

This graph has `4 + 13 * max_iter` kernel nodes and a `max_iter * 4` float64 log
(one reserved row when max_iter is zero). Construction and graph memory scale
with the generation limit. Buffers and graph are reused across runs; changing
the problem, population, iteration limit or schedule requires rebuilding.

## Base-aligned algorithm

The device operators follow `src_python` paper mode for feasibility and repair,
the five-neighbourhood SA initializer, guided SHO crossover and mutation,
route-wise WOA, scalar objective/acceptance, combined local search and
stagnation detection. RCRS-GRASP construction and island migration remain GPU
extensions. GPU evaluation of cosine can differ slightly from the CPU math
library; identical seeds are not a universal bitwise-equivalence guarantee on
different hardware. Compare quality over multiple seeds and measure elapsed
time separately from first-use compilation/capture.

## Requirements and checks

- Python 3.10+ for the maintained CUDA stack; the project's CPU code can still
  use Python 3.9. NVIDIA-maintained `numba-cuda>=0.30.4`, Numba, NVIDIA GPU/driver, and CUDA
  Toolkit usable by Numba. Install the project's `cuda` extra. It constrains
  NumPy to `<2.3`: the newer NumPy in the local environment removed APIs needed
  by the tested CUDA compiler. The notebook installs these dependencies before
  importing Torch/NumPy/Numba; restart a previously used kernel before rerunning.
- The driver must export `cuStreamBeginCapture_v2`, `cuStreamEndCapture`,
  `cuGraphInstantiateWithFlags`, `cuGraphLaunch` and graph cleanup functions
  (CUDA 11.4+ API). No extra Python graph package is needed.
- `--compute_backend cuda` fails if CUDA is unavailable. Explicit `cpu` and
  `auto` retain their reference/backend-selection semantics.
- `NUMBA_ENABLE_CUDASIM=1` is **CPU simulation**, clearly labeled in logs. It
  replays the same steps for logic tests, not actual CUDA Graph capture.
- Progress lines appear after each run, not live during a run.

From the repository root, on a real CUDA GPU:

```bash
python -m pytest tests/test_gpu_search_graph.py -q
```

The same tests can check device logic without a GPU:

```bash
NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_gpu_search_graph.py -q
```

Tests compare the full active population, island/global bests, RNG and NV/TD
trace with an independent host schedule; cover all modes, coincident schedules,
odd/even and zero generations, replay/reset, uint32 seed wrapping, and CPU
decode/feasibility checking. Simulation does not validate driver capture,
device compilation or performance; run the suite on Kaggle before benchmarking.

Local verification: 8 simulator tests passed with NumPy 2.2.6, Numba 0.67.0
and numba-cuda 0.30.4. All 11 graph kernel entry points compiled offline to PTX
for `sm_75` with CUDA 12.9 NVVM. Actual driver graph capture/replay and speed
have not been verified on this machine because no NVIDIA GPU is available.
