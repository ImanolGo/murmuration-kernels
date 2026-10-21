# flock-four-ways

One starling-flock step, written four ways on the GPU: CUDA C++, cuda-oxide
(SIMT in Rust), cuTile Rust (tile model) and wgpu. The point is to compare the
programming models on the same workload, with a CPU oracle, parity checks and
profiler evidence, and to say where each model helps or gets in the way.

> **Status: work in progress.** The CPU oracle and the CUDA C++, cuda-oxide, wgpu and cuTile
> Rust backends all work and agree with each other. Results exist for two GPUs (a GTX 1650 and
> an RTX 4000 Ada); wgpu was not run on the second. The write-up is not done.

The workload is a reduced version of the model in
[murmuration-gpu](https://github.com/ImanolGo/murmuration-gpu): a spatial-hash
neighbour search, then per-bird separation, cohesion, alignment, roost and
banked flight. [docs/SPEC.md](docs/SPEC.md) lists exactly what is kept and
what is left out.

## Status

| Backend | State | Notes |
|---|---|---|
| CPU oracle (Rust) | done | brute-force neighbours |
| CUDA C++ | baseline done | unoptimised, passes parity |
| cuda-oxide (SIMT, Rust) | done, passes parity | runs on sm_75; `decide` 4-24 % slower depending on tuning, see below |
| cuTile Rust | done, passes parity | needs sm_80+; 2.1-2.5x slower than CUDA C++ on `decide` once tuned, see below |
| wgpu (WGSL) | done, passes parity | within ~7 % of CUDA C++ on this GPU |

## First numbers

GTX 1650 Max-Q (sm_75), driver 615.78.08, CUDA 13.4, seed 1, 20 warm-up and 20
timed steps, mean GPU time of `decide` (96-98 % of the step) in one session:

| Birds | CUDA C++ | cuda-oxide + `launch_bounds` | same, unchecked indexing | wgpu (WGSL) |
|---|---|---|---|---|
| 100,000 | 7.4 ms | 8.2 ms | 7.9 ms | 8.0 ms |
| 1,000,000 | 97 ms | 101 ms | 100 ms | 100-101 ms |

Untuned cuda-oxide (no `launch_bounds`) takes 9.2-9.7 ms at 100k. Nsight Compute shows
why oxide `decide` differs from CUDA: memory behaviour is identical, but oxide used 72
registers against 61 (75 % instead of 100 % occupancy; `launch_bounds` fixes that) and,
even at equal occupancy, executes 77 % more instructions (bounds checks explain about
a fifth; float min/max compile to compare-and-select instead of the hardware
instruction). The kernel is latency-bound, so that costs 14 %, not 77 %. There are no
counters for the wgpu shaders (Nsight Compute does not profile Vulkan compute).
Details, caveats and what is not tested are in [docs/progress.md](docs/progress.md).

## Second GPU: RTX 4000 Ada (sm_89), all four GPU backends but wgpu

Rented VM, driver 580.105, CUDA 13.3, same protocol (20 warm-up + 20 timed steps, 3 runs each).
Mean GPU time of `decide` per step:

| Birds | CUDA C++ | cuda-oxide | cuTile Rust, defaults | cuTile Rust, tuned |
|---|---|---|---|---|
| 100,000 | 1.24 ms | 1.64 ms | 5.57 ms | 2.59 ms |
| 1,000,000 | 9.46 ms | 12.3 ms | 50.6 ms | 23.3 ms |

cuTile's regular passes beat CUDA's (at 1M birds, `integrate` 0.13 vs 0.20 ms and the scan 0.04 vs 0.08 ms), but its
neighbour search is 2-5x slower. Its default build uses 246 registers per thread (16.7 % occupancy); an occupancy hint
of 10 brings that to 48 and halves the time, after which it executes 4.8x the instructions of CUDA. The cuTile tuning
was searched on this input (tile size, warps, occupancy hint), which favours it. Method, every setting tried, the
Nsight Compute table and the caveats are in [docs/progress.md](docs/progress.md); raw logs and reports are in
[docs/results/rtx4000ada-2026-10-21](docs/results/rtx4000ada-2026-10-21/).

## Run it

```
cargo test
scripts/parity.sh                       # CPU oracle vs CUDA, needs an NVIDIA GPU
cargo run --release -p flock-ref -- gen --birds 100000 --seed 1 --out data/s100k.flk
make -C cuda && cuda/build/flock_cuda --in data/s100k.flk --steps 40 --warmup 20
# cuda-oxide: needs `cargo-oxide` from NVIDIA/cuda-rust and its pinned nightly
(cd oxide && CUDA_HOME=/opt/cuda cargo oxide build --arch sm_75)
oxide/target/release/flock-oxide --in data/s100k.flk --steps 40 --warmup 20
BACKEND=oxide scripts/parity.sh
# wgpu
cargo run --release -p flock-wgpu -- --in data/s100k.flk --steps 40 --warmup 20
BACKEND=wgpu scripts/parity.sh
```

Needs Rust 1.95+, a CUDA toolkit (nvcc 13.x) and an NVIDIA GPU. On a hybrid
laptop run the GPU binary under `prime-run`; the program prints the adapter name.
Set `ARCH=sm_86` (or your GPU) for `make`.

## Licence

MIT. The model constants and equations follow Hildenbrandt, Carere & Hemelrijk
(2010) through murmuration-gpu (MIT).
