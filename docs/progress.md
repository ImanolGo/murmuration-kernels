# Progress log

## 2026-10-20: step 1, oracle and CUDA C++ baseline

**Done**
- `flock-ref`: CPU oracle (brute-force neighbours), `.flk` file, `gen`, `step`,
  `diff`, `statdiff`, `stats`. 6 unit tests, `cargo clippy -D warnings` clean.
- `cuda/flock.cu`: straightforward port (hash, radix sort with CUB, bucket
  ranges, decide, integrate). Not optimised. Built for sm_75, nvcc 13.4.
- `scripts/parity.sh`: 2000 birds, passes at 1, 100 and 1000 steps (see SPEC.md
  for what each step count checks).
- `compute-sanitizer` memcheck, racecheck and initcheck: 0 errors at 2000
  birds, 20 steps.

**Measured** (GTX 1650 Max-Q, sm_75, driver 615.78.08, CUDA 13.4, seed 1,
density 0.5, 20 warm-up + 20 timed steps, mean ms per step from CUDA events)

| Birds | grid (keys+sort+ranges) | decide | integrate | step total |
|---|---|---|---|---|
| 100,000 | 0.18 | 7.55 | 0.10 | 7.81-7.84 (3 runs) |
| 1,000,000 | 1.40 | 98.9 | 0.99 | 101.3, 101.3, **183.2** (3 runs) |

The third 1M run took 183 ms against 101 ms for the other two. Not
investigated (GPU clock throttling is a guess, not a finding). Use the two
consistent runs with this caveat until it is explained.

`decide` is 96-98 % of the step. That is where the next work goes.

**Not comparable yet**
`docs/bench-gtx1650.txt` in murmuration-gpu reports 14.95 ms per step at 100k
for the wgpu path. Do not set it next to 7.8 ms and claim a speed-up: the wgpu
run includes features this core step omits (escape, random force, 96-byte bird
records), and the scatter differs (see SPEC.md). The honest comparison is the
wgpu backend of *this* repo, once it implements the same core step.

**Open**
- No cross-check against `murmur-ref` (SPEC.md `TODO`).
- Nsight Compute is not installed (`ncu`, `nsys` not found). Needed for the
  roofline and memory-throughput claims. Installing it is a system change; ask first.
- cuTile Rust needs sm_80 or newer: it cannot run on this GPU. cuda-oxide (SIMT,
  nightly) may target sm_75: `TODO: verify`.

**Next**
1. Port the core step to cuda-oxide (SIMT in Rust), same kernel structure.
2. wgpu backend of the same core step for the like-for-like baseline.
3. cutile-rs backend, benchmarked on a rented sm_80+ GPU.
4. Optimisations of `decide` in the CUDA version, each its own commit and measurement.

## 2026-10-20 (later): counting-sort grid, cuda-oxide port started

**Changed the CUDA baseline's grid.** The first baseline sorted with
`cub::DeviceRadixSort`. A Rust kernel cannot call CUB, and the wgpu path uses a
hand-written histogram, scan and scatter, so the CUDA version now does the same
(`grid_histogram`, `grid_scan` in three kernels, `grid_scatter`). Every backend
shares one grid algorithm this way. Parity (1, 100, 1000 steps) and
compute-sanitizer (memcheck, racecheck, initcheck) pass again.

Mean ms per pass, same GPU and protocol as above, three runs each (identical to
the digits shown): 

| Birds | histogram | scan | scatter | decide |
|---|---|---|---|---|
| 100,000 | 0.043 | 0.068 | 0.036 | 7.41 |
| 1,000,000 | 1.81 | 0.47 | 3.21 | 97.2 |

The 183 ms outlier from the CUB runs did not reappear and is still unexplained.
Step totals for this table are not recorded yet (the summary line was cut off
when copying); re-run before quoting them.

**cuda-oxide feasibility (answers the open question above).**
- `cargo oxide doctor` passes on this machine once `CUDA_HOME=/opt/cuda` is set
  (it only probes `/usr/local/cuda*`). The pinned nightly (nightly-2026-10-11)
  was installed by rustup on first use.
- The upstream `vecadd` example **builds and runs correctly on the GTX 1650
  (sm_75)**. cuda-oxide is therefore usable for this project on the laptop.
- Pinned to cuda-rust revision `e027ac324f663e8408c13c116d7dddd58d5e8daf`.
- `ncu` and `nsys` are now installed (ncu 2026.3.1, nsys 2026.3.2).

**State of `oxide/` (work in progress, does not compile yet).** The full port is
written (all kernels and host code). The first build failed on one error:
`cuda_core::CU_EVENT_DEFAULT` does not exist; the event flag has to come from the
`cuda_bindings` re-export (`CUevent_flags_enum_CU_EVENT_DEFAULT`), to be located.
More errors may follow once that one is fixed, because rustc stopped at the first
phase. Nothing in `oxide/` has been run, so no parity or timing claim exists for it.
Build with the `cargo-oxide` binary from a checkout of NVIDIA/cuda-rust
(`cargo-oxide build` from `oxide/`, `CUDA_HOME=/opt/cuda`).

**Next:** fix the event flag, get `oxide/` compiling, run parity against the
oracle, then profile CUDA and oxide with `ncu` on identical inputs.

## 2026-10-21: cuda-oxide port compiles, passes parity, first comparison

**Fixed and run.** The event flag lives at `cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT`
(`cuda_core::sys` re-exports the bindings); after that the whole port built for
sm_75 with no further errors. A second bug was in my host timing code (event
stride), not in the kernels. `/tmp` was cleared by a reboot, so the cuda-rust
checkout was re-cloned at the pinned revision `e027ac3`.

**Correctness.** Against the CPU oracle on the same 2000-bird start the oxide
port gives *the same* errors as the CUDA C++ version: max |dpos| 4.768e-7 after 1
step and 7.629e-6 after 100, zero neighbour-count mismatches, and the 1000-step
aggregates pass (`BACKEND=oxide scripts/parity.sh`). `compute-sanitizer` memcheck: 0 errors.

**Speed** (GTX 1650 Max-Q, sm_75, driver 615.78.08, CUDA 13.4, seed 1, 20 warm-up +
20 timed steps, mean ms from CUDA events, three runs each unless noted):

| Birds | CUDA C++ `decide` | oxide `decide`, default | oxide, `launch_bounds(256, 4)` |
|---|---|---|---|
| 100,000 | 7.41 (7.41, 7.41, 7.42) | 9.17-9.65 (9.65, 9.17, 9.17) | 8.18-8.40 (8.40, 8.19, 8.18) |
| 1,000,000 | 97.2 (3 runs) | 106.3-106.5 (3 runs) | 100.9 (2 runs) |

Step totals, default oxide vs CUDA: 100k 9.41-9.90 vs 7.65-7.66 ms; 1M 112.8-113.0
vs 103.7 ms. The grid kernels (histogram, scatter) and `integrate` take the same
time on both; only `decide` differs.

**Why `decide` differs: registers, shown three ways.**
1. Registers per thread of the kernels *as loaded* (Nsight Systems kernel table,
   no profiler counters needed): `decide` 61 (CUDA) vs 72 (oxide), `integrate`
   46 vs 53. Neither spills by default.
2. On Turing, 61 registers allow 4 blocks of 256 threads per SM (100 %), 72 allow
   3 (75 %). The oxide `decide` also has ~9 % more SASS instructions (1400 vs 1280,
   from `ptxas` on `cargo oxide inspect` PTX; the runtime JIT path may differ).
3. Experiment: the CUDA kernel, unchanged, launched with 20 KB of dynamic shared
   memory so only 3 blocks fit per SM (`--decide-smem 20480`, occupancy printed by
   the program): **9.30 ms at 100k** (oxide 9.17-9.65) and **108.7 ms at 1M** (oxide
   106.4). With occupancy matched, the two compilers' code runs equally fast; the gap
   is the register allocation, not the code.

`#[launch_bounds(256, 4)]` on the oxide kernel (the `__launch_bounds__(256, 4)`
equivalent) recovers about half the gap: it caps registers at 64 by spilling 48 bytes
(`ptxas`: 48 bytes spill stores, 52 loads), so it is still ~10 % slower than CUDA
at 100k and ~4 % at 1M. This is a measured result on one GPU, one workload, one
cuda-oxide revision. It says nothing about newer compilers or other architectures.

**Not yet measured.** Nsight Compute cannot read the GPU counters as a normal user
(`ERR_NVGPUCTRPERM`), so there is no achieved-occupancy, memory-throughput or
roofline data yet. Enabling it is a system change (run `ncu` under `sudo`, or set
the driver option `NVreg_RestrictProfilingToAdminUsers=0` and reboot): ask first.
The occupancy explanation above is supported by the experiment, not by counters.

**Safety, as written** (rough, not a claim about the borrow checker): the
oxide kernels use `DisjointSlice` for every per-thread output and `&[T]` for reads;
the grid scan and scatter need raw pointers. `oxide/src/main.rs` has 18 lines that
mention `unsafe` (comments and SAFETY notes included), `cuda/flock.cu` has none,
because C++ has no equivalent marker. A fair count of unsafe *operations* needs
a proper pass.

**Next:** wgpu backend of the same step (the like-for-like portable baseline),
then cutile-rs on a rented sm_80+ GPU. Open: counters for Nsight Compute.
