# flock-four-ways

One starling-flock step, written four ways on the GPU: CUDA C++, cuda-oxide
(SIMT in Rust), cuTile Rust (tile model) and wgpu. The point is to compare the
programming models on the same workload, with a CPU oracle, parity checks and
profiler evidence, and to say where each model helps or gets in the way.

> **Status: work in progress.** Only the CPU oracle and the CUDA C++ baseline
> exist. Nothing here is a finished comparison yet.

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
| cuda-oxide (SIMT, Rust) | not started | |
| cuTile Rust | not started | needs sm_80+ |
| wgpu | not started | |

## First numbers (CUDA C++ baseline)

GTX 1650 Max-Q (sm_75), driver 615.78.08, CUDA 13.4, seed 1, 20 warm-up and 20
timed steps, mean GPU time per step from CUDA events:

| Birds | Step | Of which `decide` |
|---|---|---|
| 100,000 | 7.8 ms | 7.55 ms |
| 1,000,000 | 101 ms (one of three runs: 183 ms, not yet explained) | 99 ms |

See [docs/progress.md](docs/progress.md) for the rest and for what these
numbers can and cannot be compared with.

## Run it

```
cargo test
scripts/parity.sh                       # CPU oracle vs CUDA, needs an NVIDIA GPU
cargo run --release -p flock-ref -- gen --birds 100000 --seed 1 --out data/s100k.flk
make -C cuda && cuda/build/flock_cuda --in data/s100k.flk --steps 40 --warmup 20
```

Needs Rust 1.95+, a CUDA toolkit (nvcc 13.x) and an NVIDIA GPU. On a hybrid
laptop run the GPU binary under `prime-run`; the program prints the adapter name.
Set `ARCH=sm_86` (or your GPU) for `make`.

## Licence

MIT. The model constants and equations follow Hildenbrandt, Carere & Hemelrijk
(2010) through murmuration-gpu (MIT).
