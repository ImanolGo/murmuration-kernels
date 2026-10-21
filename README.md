# flock-four-ways

One starling-flock step, written four ways on the GPU: CUDA C++, cuda-oxide
(SIMT in Rust), cuTile Rust (tile model) and wgpu. The point is to compare the
programming models on the same workload, with a CPU oracle, parity checks and
profiler evidence, and to say where each model helps or gets in the way.

> **Status: work in progress.** The CPU oracle, CUDA C++ and cuda-oxide backends
> work and agree. wgpu and cuTile Rust are not started, so this is not a finished
> comparison yet.

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
| cuda-oxide (SIMT, Rust) | done, passes parity | runs on sm_75; `decide` 10-24 % slower, see below |
| cuTile Rust | not started | needs sm_80+ |
| wgpu | not started | |

## First numbers

GTX 1650 Max-Q (sm_75), driver 615.78.08, CUDA 13.4, seed 1, 20 warm-up and 20
timed steps, mean GPU time of `decide` (96-98 % of the step) from CUDA events:

| Birds | CUDA C++ | cuda-oxide | cuda-oxide with `launch_bounds(256, 4)` |
|---|---|---|---|
| 100,000 | 7.4 ms | 9.2-9.7 ms | 8.2-8.4 ms |
| 1,000,000 | 97 ms | 106 ms | 101 ms |

The grid and integrate kernels take the same time in both. The `decide` gap comes
from register allocation: 72 registers against 61, so 3 blocks per SM instead of 4.
Launching the CUDA kernel at the same 75 % occupancy gives 9.3 ms and 109 ms, i.e.
the same speed as oxide. Details, caveats and what is not measured yet (no
profiler counters) are in [docs/progress.md](docs/progress.md).

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
```

Needs Rust 1.95+, a CUDA toolkit (nvcc 13.x) and an NVIDIA GPU. On a hybrid
laptop run the GPU binary under `prime-run`; the program prints the adapter name.
Set `ARCH=sm_86` (or your GPU) for `make`.

## Licence

MIT. The model constants and equations follow Hildenbrandt, Carere & Hemelrijk
(2010) through murmuration-gpu (MIT).
