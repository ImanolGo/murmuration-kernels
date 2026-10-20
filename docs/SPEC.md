# Flock core step: specification

Every backend in this repository implements this step and nothing else, so the
comparison is between programming models and not between models of a flock.
The CPU oracle in `crates/flock-ref/src/model.rs` is the source of truth. A GPU
backend mirrors its functions, names and order of operations (the same rule
`murmuration-gpu` follows for its wgpu shaders).

## Relationship to murmuration-gpu

The step is a subset of the `murmur-ref` model in
[ImanolGo/murmuration-gpu](https://github.com/ImanolGo/murmuration-gpu) (MIT).
Constants are copied from `murmur-ref/src/params.rs`. The decide and integrate
code follows `social::decide`, `flight::step_bird` and `World::step`.

`TODO: verify` that this subset gives the same single-step result as
`murmur-ref` with the omitted features switched off. Until a cross-check test
exists, "mirrors murmur-ref" means "written from reading it", not "tested
against it". The oracle here is only tested against the GPU backends.

### Kept

- Spatial hash grid, cell 4 m, `2^k` buckets with an exact cell check (as `neighbours.rs`).
- Per bird: separation (Eq. 4), all-round cohesion scaled by centrality (Eq. 5, 7),
  alignment with the rear blind sector (Eq. 6, 8), vertical roost pull (Eq. 12),
  speed control (Eq. 1), the adaptive radius `R <- R (n_c/|N|)^(s/3)`.
- Banked flight with the base turning model (`flight::step_bird`).
- Staggered decisions: a bird decides when `time >= next_decision`, then
  `next_decision += DU`. Steering is held between decisions.

### Omitted

Falcon, alarm, flash and zigzag; the evening arc; the random force (so runs are
deterministic); the horizontal roost and committed return; the wingbeat; the
inertial turning mode. The benchmark scatter sits inside the roost, so the
horizontal roost would not act on it anyway, but a long run would drift out
and the omission would then matter.

### Order of one step

1. `time += DT` (f32).
2. Rebuild the grid from the current positions.
3. Every bird with `time >= next_decision` decides, reading a snapshot of
   position, velocity and radius. It writes its own steering, radius, neighbour
   count and centrality, and advances `next_decision`.
4. Every bird integrates one `DT` under its held steering.

## Benchmark protocol

Same as `murmur-gpu`'s `--bench` (`crates/murmur-gpu/src/bench.rs`): `n` birds
scattered uniformly in a cube of density 0.5 birds/m^3 centred at z = 100 m,
heading +x at 10 m/s with 0.05 jitter, start radius 1.5 m, decision times
uniform in `[0, DU)`. 20 warm-up steps, then timed steps. Times come from GPU
events (CUDA) or timestamp queries (wgpu), never from wall clocks.

The scatter generator here is SplitMix64, not `murmur-gpu`'s ChaCha, so the
starting positions differ from that benchmark even for the same seed. The
statistics (density, speed, radius) are the same.

## Exchange file `.flk`

Little endian, 4-byte words: `"FLK1"`, `n`, `15`, `time`, then `n` records of
pos xyz, vel xyz, roll, spin, radius, steering xyz, next_decision,
neighbour count (u32), centrality. Every backend reads and writes it.

## Parity criteria

| Steps | Check | Criterion |
|---|---|---|
| 1 and 100 | pointwise `flock-ref diff` | position and velocity within 1e-3, zero neighbour-count mismatches |
| 1000 | aggregate `flock-ref statdiff` | mean speed 0.02, mean neighbours 0.1, mean radius 0.01, polarization 0.005 |

The pointwise check is not kept at 1000 steps on purpose. A bird near its
radius edge flips a neighbour under a different summation order, and the flock
amplifies it: at 1000 steps (5 simulated seconds) 539 of 2000 birds had a
different neighbour count, while every aggregate still agreed. The statistical
tolerances are this project's own choice, not a published criterion.

## Constants

Rust: `crates/flock-ref/src/params.rs`. CUDA: `cuda/params.cuh`. They must stay
identical; there is no automatic check yet (`TODO`).
