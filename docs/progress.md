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
