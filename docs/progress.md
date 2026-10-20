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
