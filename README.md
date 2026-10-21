# flock-four-ways

One starling-flock step, written four ways on the GPU: CUDA C++, cuda-oxide
(SIMT in Rust), cuTile Rust (tile model) and wgpu. The point is to compare the
programming models on the same workload, with a CPU oracle, parity checks and
profiler evidence, and to say where each model helps or gets in the way.

> **Status: work in progress.** The CPU oracle and the CUDA C++, cuda-oxide and wgpu
> backends work and agree. The cuTile Rust backend is written and compiles to sm_89
> machine code, but has **not run yet** (it needs an sm_80+ GPU), so there is no cuTile
> result and this is not a finished comparison.

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
| cuTile Rust | written, compiles for sm_89, **not run** | needs sm_80+; `scripts/cloud-session.sh` runs it |
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
