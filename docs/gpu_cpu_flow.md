# Flow GPU-CPU Chi Tiết: `src_python_gpu_SA_RCRS_GRASP`

## Tổng Quan Kiến Trúc

Code này có **3 execution path** tùy theo backend được chọn, và chọn tự động theo thứ tự ưu tiên:

```
main.py → search_framework() → run_solver() [gpu_engine.py]
              ↓
        ┌─────────────────────────────────────────────────────┐
        │  Auto-detect backend priority:                      │
        │  1. PyTorch CUDA (TorchComputeBackend)              │
        │     → gpu_pure_tensor_search_framework()            │
        │  2. Numba CUDA (CudaComputeBackend)                 │
        │     → GpuEngine._run_cuda_solve() + CudaSearchGraph │
        │  3. CPU Fallback (BaseComputeBackend)               │
        │     → GpuEngine._run_reference_solve()              │
        └─────────────────────────────────────────────────────┘
```

> [!NOTE]
> `search_framework()` hiện tại **delegate ngay sang `run_solver()`** (GpuEngine), bỏ qua toàn bộ code Python từ line 1935 trở đi trong `search_framework.py`.  
> Code `gpu_pure_tensor_search_framework()` (PyTorch) là dead code - chỉ chạy nếu gọi trực tiếp.

---

## PATH 1: Numba CUDA (Default GPU Path — `GpuEngine`)

Đây là path **chính** khi `--compute_backend cuda` hoặc CUDA available.

### PHASE 0 — CPU: Khởi tạo & Phân bổ

| Bước | Nơi chạy | Chi tiết |
|------|----------|---------|
| Parse args, load problem data | **CPU** | `data.py`, `main.py` |
| `prepare_problem_data()` | **CPU** | Build numpy arrays: `dist_matrix`, `delivery`, `pickup`, `start_tw`, `end_tw`, `service` |
| `_allocate_buffers()` | **CPU → GPU** | Allocate `cuda.device_array(...)` cho `pop_nodes[P,R,L]`, `pop_cost[P]`, `ibest`, `gbest`, `scratch_*` |
| Upload problem data | **CPU → GPU** | `cuda.to_device(delivery)`, `cuda.to_device(dist_matrix)`, ... (`prob_device`) |
| Build kernel bundle | **CPU** | `build_kernel_bundle(is_cuda=True)` — compile Numba CUDA JIT kernels |
| Unroll CUDA Graph | **CPU** | `CudaSearchGraph._make_steps()`: loop `max_iter` lần để build DAG |
| Graph capture | **CPU** | `_capture()` → `cuStreamBeginCapture_v2` → **record** tất cả kernel calls vào graph |
| `cuGraphInstantiateWithFlags` | **CPU** | Compile graph thành executable (1 lần duy nhất) |

### PHASE 1 — GPU: Chạy từng Run

```
graph.prepare_run(seed) ──▶ CPU: copy seed lên device (host_seed → seed_d)
graph.launch()          ──▶ GPU: cuGraphLaunch() — 1 call duy nhất!
graph.synchronize()     ──▶ CPU: chờ GPU xong
```

**Bên trong CUDA Graph (100% GPU), mỗi generation:**

```
┌─ CUDA Graph Replay (GPU only) ─────────────────────────────────────────────┐
│                                                                              │
│  [INIT] init_population_kernel  ← CUDA kernel: GRASP + RCRS init           │
│    • Xorshift128 PRNG trên mỗi thread                                       │
│    • GRASP construction: build routes theo alpha-bounded savings list        │
│    • eval_solution() per thread                                              │
│                                                                              │
│  For gen = 1 → max_iter (UNROLLED, không có loop host nào):                 │
│    ├─ parameters_kernel   (1 block×1 thread): tính a, p_hybrid              │
│    ├─ update_population_kernel  (P threads):                                 │
│    │    • SHO bubble-net / spiral (p_mode > rand)                            │
│    │    • WOA shrinking encircling (p_mode < rand)                           │
│    │    • RCRS crossover: remove + greedy insertion                          │
│    │    • SA acceptance: ΔE, T cooling                                       │
│    │    • eval_solution() per thread                                          │
│    ├─ eliminate_kernel    (P threads): route elimination (NV-first)          │
│    ├─ search_kernel       (P threads): 2-opt / or-opt intra route            │
│    ├─ diversify_kernel    (island blocks): stagnation diversify               │
│    ├─ migrate_kernel      (1 block): ring migration giữa islands             │
│    ├─ update_island_bests (island blocks)                                    │
│    ├─ update_global_best  (1 block)                                          │
│    └─ finish_kernel       (1 block): ghi history[gen]                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### PHASE 2 — CPU: Decode Kết Quả

```
device_best[0..4].copy_to_host(host_best)  ← GPU → CPU (1 lần/run)
graph.read_history()                        ← GPU → CPU (history array)
```

| Bước | Nơi chạy | Chi tiết |
|------|----------|---------|
| `copy_to_host()` | **GPU → CPU** | Copy `gbest_nodes`, `gbest_rlen`, `gbest_nr`, `gbest_dist`, `gbest_cost` |
| Read history | **GPU → CPU** | `history[max_iter, 4]` (a, p_hybrid, NV, TD mỗi gen) |
| Decode solution | **CPU** | Duyệt `h_nodes[0, r, :]` để build `Solution` object |
| `route.update()`, `cal_cost()` | **CPU** | Python-level cost recompute |
| `solution.check()` | **CPU** | CPU verification |
| Best solution compare & update | **CPU** | `state.best_s` update |

---

## PATH 2: PyTorch Tensor (Alternative GPU — `gpu_pure_tensor_search_framework`)

> [!WARNING]
> **Hiện tại là dead code** — `search_framework()` return ngay từ line 1933, không bao giờ reach function này trừ khi gọi trực tiếp.

### PHASE 0 — CPU Setup

| Bước | Nơi chạy | Chi tiết |
|------|----------|---------|
| `TorchComputeBackend.__init__()` | **CPU → GPU** | Upload tất cả matrices lên CUDA VRAM: `delivery_t`, `dist_t`, `time_t`, ... |
| `tensor_rcrs_grasp_init()` | **GPU** | Build population trên CUDA tensor |

### PHASE 1 — GPU Main Loop

**Tất cả operations trong generation loop là pure PyTorch CUDA tensor:**

```
┌─ Generation Loop (P×R×L tensors, GPU resident) ─────────────────────────────┐
│                                                                              │
│  INIT:  tensor_rcrs_grasp_init()     ← GPU: GRASP construction              │
│         tensor_sa_warmup()           ← GPU: SA warm-up                      │
│                                                                              │
│  For gen = 1 → max_iter:                                                    │
│    ├─ tensor_sho_crossover_single()  ← GPU: SHO bubble-net crossover        │
│    ├─ tensor_woa_intensification()   ← GPU: WOA spiral update                │
│    ├─ evaluate_population_tensor()   ← GPU: batch eval (dist+TW+capacity)   │
│    │    ─ dist_t[prev, curr] gather                                          │
│    │    ─ cumsum load check                                                  │
│    │    ─ sequential TW loop (trên từng col j)                               │
│    │    ─ scatter_add_ customer coverage check                               │
│    ├─ SA acceptance                  ← GPU: torch.where()                    │
│    ├─ tensor_deep_local_search_gpu() ← GPU: inter-route relocate/swap       │
│    ├─ Island Ring Migration          ← GPU: pure tensor indexing             │
│    └─ Stagnation Diversify           ← GPU: tensor_generate_offspring_batch  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### PHASE 2 — CPU Decode (Single sync after all runs)

```
"CPU_DECODE: copying final GPU solution for output (Single Host Sync after all runs)"
```

| Bước | Nơi chạy |
|------|----------|
| `global_best_routes_t.tolist()` | **GPU → CPU** |
| `decode_tensor_solution()` | **CPU** |
| `best_s.output()`, `best_s.check()` | **CPU** |

---

## PATH 3: CPU Reference (`_run_reference_solve`)

Fallback khi không có GPU. Kernels **y hệt CUDA kernels** nhưng dùng `@njit` thay `@cuda.jit`:

```
┌─ CPU Reference Loop ────────────────────────────────────────────────────────┐
│                                                                              │
│  build_kernel_bundle(is_cuda=False):                                        │
│    dev_fn = @njit(fastmath=True, nogil=True)                                │
│    k_fn   = @njit(fastmath=True, nogil=True)                                │
│                                                                              │
│  for run in 1..total_runs:                                                  │
│    init_population_kernel(...)     ← Numba CPU (JIT compiled)               │
│    update_island_bests_kernel(...) ← Numba CPU                              │
│    update_global_best_kernel(...)  ← Numba CPU                              │
│                                                                              │
│    for gen in 1..max_iter:                                                  │
│      update_population_kernel(...) ← Numba CPU                              │
│      route_elimination_kernel(...) ← Numba CPU                              │
│      local_search_kernel(...)      ← Numba CPU                              │
│      stagnation_diversify_kernel() ← Numba CPU                              │
│      island_migration_kernel(...)  ← Numba CPU                              │
│      update_island_bests(...)      ← Numba CPU                              │
│      update_global_best(...)       ← Numba CPU                              │
│                                                                              │
│  Decode: Python/CPU (same as GPU path)                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Chi Tiết Các GPU Kernels (`gpu_kernels.py`)

| Kernel | Compile mode | Chức năng |
|--------|-------------|-----------|
| `xorshift128`, `rand_u01`, `randint` | `@cuda.jit(device=True)` | Per-thread PRNG, Xorshift128 |
| `eval_route` | `@cuda.jit(device=True)` | Route feasibility: capacity + time-window + distance |
| `eval_solution` | `@cuda.jit(device=True)` | Aggregate solution cost (NV, TD) |
| `copy_solution` | `@cuda.jit(device=True)` | Deep copy solution buffers |
| `grasp_rcrs_construct` | `@cuda.jit(device=True)` | GRASP construction per thread |
| `sa_acceptance` | `@cuda.jit(device=True)` | Simulated Annealing accept/reject |
| `sho_bubble_move` / `woa_spiral_move` | `@cuda.jit(device=True)` | SHO/WOA position update |
| `init_population_kernel` | `@cuda.jit` (global) | P threads, mỗi thread init 1 solution |
| `update_population_kernel` | `@cuda.jit` (global) | P threads: SHO/WOA + RCRS + SA |
| `route_elimination_kernel` | `@cuda.jit` (global) | P threads: NV-first route elimination |
| `local_search_kernel` | `@cuda.jit` (global) | P threads: 2-opt / or-opt |
| `update_island_bests_kernel` | `@cuda.jit` (global) | `num_islands` threads |
| `update_global_best_kernel` | `@cuda.jit` (global) | 1 thread |
| `island_migration_kernel` | `@cuda.jit` (global) | 1 thread: ring migration |
| `stagnation_diversify_kernel` | `@cuda.jit` (global) | island threads: random removal + insertion |

---

## `compute_backend.py` — Backend Route Evaluation

| Class | Evaluate Route | Evaluate Batch | Evaluate Insertions |
|-------|---------------|---------------|---------------------|
| `BaseComputeBackend` | `@njit` (CPU) | Python loop + `@njit` | `@njit` |
| `CudaComputeBackend` | via batch | `@cuda.jit` batch kernel | `@cuda.jit` insertion kernel |
| `GpuProxyBackend` | IPC → GPU process | Mega-batch via IPC queue | IPC |
| `TorchComputeBackend` | PyTorch CUDA tensor | `evaluate_routes_gpu()` | PyTorch tensor |

**`CudaComputeBackend._ensure_device_cache()`**: upload `delivery`, `pickup`, `dist`, `time`, `start/end/service` lên GPU 1 lần, cache lại.

---

## Sơ Đồ Tổng Hợp: Luồng Dữ Liệu

```
                   ┌──────────────────────────────────────┐
                   │         CPU (Host)                   │
                   │                                      │
  Problem files ──▶│  data.py: parse, build numpy arrays  │
                   │  BackendSnapshot: pack problem data  │
                   │                                      │
                   │  cuda.to_device(delivery) ───────────┼──▶ GPU VRAM (static, 1x upload)
                   │  cuda.to_device(dist_matrix) ────────┼──▶ GPU VRAM
                   │                                      │
                   │  cuda.device_array(pop_nodes P×R×L) ─┼──▶ GPU VRAM (population)
                   │  cuda.device_array(gbest_nodes) ─────┼──▶ GPU VRAM (global best)
                   │                                      │
                   │  CudaSearchGraph._capture() ─────────┼──▶ CUDA Graph (compiled)
                   └──────────────────────────────────────┘
                                    │ graph.launch() (1 call/run)
                                    ▼
                   ┌──────────────────────────────────────┐
                   │         GPU (Device)                 │
                   │                                      │
                   │  INIT: init_population_kernel        │
                   │    ↓  (max_iter generations)         │
                   │  LOOP: update_population_kernel      │
                   │        route_elimination_kernel      │
                   │        local_search_kernel           │
                   │        diversify/migrate kernels     │
                   │        update_island/global_best     │
                   │    ↓                                 │
                   │  gbest_nodes, gbest_cost  (buffered) │
                   └──────────────────────────────────────┘
                                    │ copy_to_host (1x/run)
                                    ▼
                   ┌──────────────────────────────────────┐
                   │         CPU (Host) — Decode          │
                   │                                      │
                   │  h_nodes → Route objects             │
                   │  route.update(), cal_cost()          │
                   │  solution.check() (CPU verify)       │
                   │  Update state.best_s                 │
                   └──────────────────────────────────────┘
```

---

## Tóm Tắt GPU vs CPU

| Giai Đoạn | GPU | CPU |
|-----------|-----|-----|
| Parse problem & build data structures | ✗ | ✅ |
| Upload matrices (delivery, dist, time) | ✅ → VRAM | ✅ source |
| Allocate population buffers | ✅ VRAM | ✗ |
| Build/compile CUDA Graph | ✗ (CPU pre-process) | ✅ |
| GRASP initialization (all P solutions) | ✅ | ✗ |
| SA warm-up | ✅ | ✗ |
| SHO/WOA position update (all P) | ✅ | ✗ |
| RCRS crossover + repair | ✅ | ✗ |
| Route feasibility evaluation | ✅ | ✗ |
| Route elimination (NV-first) | ✅ | ✗ |
| 2-opt / or-opt local search | ✅ | ✗ |
| Island best tracking | ✅ | ✗ |
| Stagnation diversification | ✅ | ✗ |
| Island ring migration | ✅ | ✗ |
| Decode solution from GPU buffers | ✗ | ✅ |
| Solution verification (`check()`) | ✗ | ✅ |
| Update global best (`state.best_s`) | ✗ | ✅ |
| Print logs | ✗ | ✅ |

> [!IMPORTANT]
> **Zero CPU-GPU sync trong search run**: Toàn bộ `max_iter` generations được unroll vào 1 CUDA Graph và launch bằng **1 single `cuGraphLaunch()` call**. CPU không bao giờ đọc intermediate results giữa các generation. Sync chỉ xảy ra **1 lần sau khi toàn bộ run kết thúc**.
