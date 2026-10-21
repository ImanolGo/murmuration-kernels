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
   106.4). So the CUDA kernel at oxide's occupancy takes the same time as oxide.
   *Equal times are not equal code*: the counters below show oxide executes far more
   instructions; this kernel is latency-bound enough to hide much of that.

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

## 2026-10-21 (later): Nsight Compute counters

`NVreg_RestrictProfilingToAdminUsers=0` was set (modprobe option, initramfs
rebuilt, rebooted); `ncu` now runs as a normal user. One warmed-up `decide` launch
(launch 23 of 24), 100,000 birds, `--set full`, the raw reports are not published (they embed the machine's host name and paths). ncu locks the GPU clocks, so these
times are higher than the benchmark's; compare columns, not against other tables.

| `decide`, 100k birds | CUDA C++ | CUDA, occupancy capped to 75 % | oxide, `launch_bounds(256, 4)` |
|---|---|---|---|
| duration | 8.28 ms | 10.91 ms | 9.46 ms |
| registers per thread | 61 | 61 | 64 |
| theoretical occupancy | 100 % | 75 % | 100 % |
| achieved active warps | 70.9 % | 50.7 % | 70.6 % |
| instructions executed | 145.9 M | 145.9 M | **258.6 M** (+77 %) |
| issue slots active | 23.2 % | 17.7 % | 36.2 % |
| eligible warps per cycle | 0.29 | 0.20 | 0.50 |
| global-load sectors | 13.60 M | 13.60 M | 13.60 M |
| L1 / L2 hit rate | 42.8 / 25.2 % | 39.8 / 28.8 % | 42.7 / 25.6 % |

What this shows, and what it does not:
- **Memory behaviour is identical** (same sectors, same hit rates). The gap is not memory.
- **Occupancy matters in this kernel**: capping CUDA to 75 % drops achieved warps from
  71 % to 51 % and costs 32 % in time.
- **With `launch_bounds(256, 4)` oxide matches CUDA's occupancy** (70.6 vs 70.9 % active
  warps) but still runs 14 % longer, because it executes **77 % more instructions**.
  The kernel is latency-bound (CUDA issues on 23 % of cycles), so most of the extra
  instructions are hidden. Correction to the note above: the gap is not only registers.

**Where the extra instructions come from (partly tested).**
- *Bounds checks, tested.* Every slice index in Rust is checked. The opt-in
  `--features unchecked` removes them in `decide` (`get_unchecked`): instructions
  258.6 M to 235.4 M (-9 %), time 8.23 to 7.91 ms at 100k (-4 %) and 100.9 to 99.5 ms
  at 1M (-1.4 %). Parity still passes. This explains ~20 % of the extra instructions.
- *Float min/max, tested and refuted as the cause but real as a finding.* CUDA's
  `fmaxf`/`fminf` become the hardware `FMNMX` (20 in `decide`); oxide's SASS has none
  and more `FSETP` (+86), `SEL` (+56), `PLOP3` (+38), `FSEL` (+25). The generated PTX
  has **no `min.f32` / `max.f32` at all** (141 `setp`, 153 `selp`). Replacing
  `libm::fminf/fmaxf` with `f32::min/max` changed nothing (258.58 M vs 258.57 M
  instructions), so both routes lower to compare-and-select. Whether that costs time
  here was not isolated.
- *Saturating float-to-int casts, untested.* `f32 as i32` saturates and maps NaN to 0 in
  Rust. The obvious fix, `f32::to_int_unchecked`, is **rejected by the backend**
  ("invalid input program" in `core/src/convert/num.rs`, build of 2026-10-21), so it
  could not be tried. Likely related to `SEL`/`FRND` (+18) in the opcode mix; not proven.
- 64-bit index arithmetic from `usize` indices is a further candidate. Untested.

Static SASS counts (1280 CUDA vs 1400 oxide) understate the dynamic difference (+77 %):
the extra work sits in the neighbour loops, which run many times per bird.

**Housekeeping.** `cargo-oxide` is now installed in `~/.cargo/bin` from the pinned
revision (`cargo install --git ... --rev e027ac3 cargo-oxide`); `cargo uninstall
cargo-oxide` removes it. `/tmp` is cleared on reboot, so the cuda-rust checkout is
no longer needed to build `oxide/`.

**Upstream status** (everything reproduces without a GPU, see `upstream-repros/`; posted 2026-10-21): the
`constant(false)` miscompile is cuda-rust #1469 with a fix in the open PR #1470, where a confirmation was added; the
generic-helper problem is #1378, where a smaller example was added; new issues are float min/max lowering
([#1594](https://github.com/NVIDIA/cuda-rust/issues/1594)), `to_int_unchecked` ([#1595](https://github.com/NVIDIA/cuda-rust/issues/1595)), the pointer-tile method chain ([#1596](https://github.com/NVIDIA/cuda-rust/issues/1596)) and the
documentation snippet ([#1597](https://github.com/NVIDIA/cuda-rust/issues/1597)).

## 2026-10-21 (evening): wgpu backend

`crates/flock-wgpu` (stable Rust, wgpu 30.0.1, WGSL in `src/flock.wgsl`): the same
eight kernels in the same order. The device is opened with the adapter's own limits
because the shaders bind 14 storage buffers (the WebGPU default is 8), and every
buffer is declared `read_write` because the passes share one bind group layout.
Times are Vulkan timestamp queries per pass. Adapter: GTX 1650 Max-Q on Vulkan,
driver NVIDIA 615.78.08.

**Parity.** Passes at 1, 10, 30 and 1000 steps. At 100 steps it failed the original
criterion (one bird, see the traced threshold flip in SPEC.md) and passes the relaxed
one. CUDA and oxide have 0 birds above tolerance at 100 steps, wgpu has 1.

**Speed**, one session, 20 warm-up + 20 timed steps, mean ms (3 runs; the first CUDA
100k run was 8.17 ms with cold clocks and is excluded from the CUDA figure):

| Birds | CUDA C++ `decide` | oxide + `launch_bounds` | oxide + `launch_bounds` + unchecked | wgpu (WGSL) |
|---|---|---|---|---|
| 100,000 | 7.44 | 8.24 | 7.93 | 7.99-8.03 |
| 1,000,000 | 97.3 | 101.1-101.2 | 99.6-99.7 | 100.3-100.7 |

Step totals, same runs: 100k 7.68 / 8.48 / 8.18 / 8.26-8.29 ms; 1M 103.7-103.8 /
107.5-107.7 / 106.1-106.2 / 106.9-107.3 ms. Relative to CUDA, `decide` is 11 % /
7 % / 7 % slower at 100k and 4 % / 2 % / 3 % at 1M. The untuned oxide kernel
(no `launch_bounds`) is 9.2-9.7 ms at 100k.

What this does and does not show:
- The portable path (WGSL, SPIR-V, the Vulkan driver's compiler) ends up within about
  7 % of CUDA C++ here and level with tuned cuda-oxide. That is one GPU, one driver,
  one workload.
- **Why** wgpu is faster than untuned oxide is not known. Nsight Compute does not
  profile Vulkan compute, so there are no registers, occupancy or instruction counts
  for the wgpu shaders. Do not read the CUDA/oxide counter explanations across.
- wgpu's "step total" is the sum of pass durations; CUDA's is the sum of event spans.
  The wgpu "step span" (first pass begin to last pass end) is 0.06-0.08 ms longer
  than its total; the difference is small next to the gaps being compared.
- All backends share one grid algorithm (histogram, scan, scatter); the wgpu grid
  kernels cost the same as CUDA's within noise (histogram 1.84 vs 1.81 ms, scatter
  3.20 vs 3.21 ms at 1M), scan is slower (0.57 vs 0.47 ms at 1M).

`flock-ref diff` now also prints steering, roll and spin differences (informational)
and supports `--allow-outliers N` / `--outlier-max X`. `BACKEND=wgpu scripts/parity.sh` runs
the wgpu backend.

**Next:** cuTile Rust needs an sm_80+ GPU (rent one; compare against CUDA and
oxide on the same GPU, and re-profile there). Then a write-up. Open: the two possible
upstream issues (min/max lowering; `to_int_unchecked`).

## 2026-10-21 (night): cuTile Rust backend written, not yet run

`cutile/` (standalone crate, cuda-rust pinned at `e027ac3`): eight kernels (`grid_clear`,
`grid_histogram`, `grid_scan_blocks/totals/add`, `grid_scatter`, `decide`, `integrate`) and
a host runner with the same command line and `.flk` files as the other backends.
`decide` is generated from a template (`tools/gen_decide.py`, output committed) because
the DSL cannot share a loop body between the two neighbourhood walks.

**What is verified.** Every kernel lowers from the Rust DSL to Tile IR and to an sm_89 cubin
on the laptop, with no GPU, using `cutile_compiler::compile_api::KernelCompiler` and
`tileiras` (`cargo run --bin check -- --arch sm_89 --cubin-dir out`). The runner builds
clean (`clippy` has no warnings). On the laptop's sm_75 GPU it stops at the first kernel
with `cuda_tile.make_token requires sm_80 or newer`, as expected. A `--check` mode
validates the spatial grid (offsets monotone, `offsets[buckets] = n`, `sorted` a
permutation, each bird in its bucket's range); its checker has unit tests.
**Nothing has executed on a GPU, so there is no parity or timing result.**

**Static resources, sm_89, both compiled by their vendor tools** (registers per thread from
`cuobjdump --dump-resource-usage`; SASS instruction counts not compared because the CUDA
count for sm_89 was not reliably extracted):

| Kernel | CUDA C++ | cuTile Rust |
|---|---|---|
| decide | 62 regs | **246 regs**, 16 B shared, no spills |
| integrate | 40 regs, 32 B stack | 46 regs, 32 B stack |
| grid_histogram / scatter / scan_add / clear | 12-14 regs | 8-12 regs |
| grid_scan_blocks / totals | 20 regs, 1 KB shared | **128 / 124 regs, 8 KB shared** |

The cuTile `decide` is 4,064 SASS instructions (static). The threads per block the
assembler picks (`EIATTR_REQNTID`) was not decoded; the profiler will report it. Register
count and achieved occupancy are the things to look at first when it runs: 246 registers
per thread allows few resident warps.

**How the tile model changed the algorithm** (a finding in itself, not an optimisation):
- A tile of 128 birds is one program. The per-bird neighbour loops have data-dependent trip
  counts, which a tile program cannot express, so each walk iterates to the **largest cell
  span and bucket size in the tile**, with every lane masked. Lanes with fewer candidates
  idle while the busiest finishes. CUDA has the same divergence within a warp of 32; here
  it spans 128 lanes.
- Neighbour data are read with masked pointer gathers (`load_ptr_tko`), one per field and
  candidate; CUDA reads one `float4`. Positions and velocities are separate arrays.
- The scan is simpler: `scan_sum` does the in-block prefix scan in three lines per kernel,
  with no hand-written shared-memory loop. Histogram and scatter use pointer-tile atomics.

**DSL limits met while writing it** (cuda-rust `e027ac3`; each one cost an iteration):
1. No `atan` or `atan2` in the DSL (`sin`, `cos`, `tan`, `exp`, `pow`, `floor`, `sqrt` exist).
   `integrate` needs `atan` for the bank target, so it carries a Cephes polynomial
   (about 1e-7 error, three comparisons and selects). Parity will show whether that is enough.
2. A user device function that is generic over a shape generic (`const S: [i32; 1]`) failed to infer its
   return type when called in argument position ("Failed to determine typeck return type"), so I bound
   such calls to annotated `let`s. A later minimal reproduction (`upstream-repros/cutile-dsl-friction`)
   shows that a *concrete* helper called in argument position compiles; the failure is tied to the generic.
3. **A generic device function called from a generic kernel returns a type with the kernel's generic
   unsubstituted** (`Tile<f32, { [B] }>` against `Tile<f32, { [128] }>`; reproduced, compile-only). A module
   `const` is also not evaluated inside a type `{ [B] }`. The tile size is therefore the literal `128`
   everywhere; changing it means editing the source (the generator does it for `decide`).
4. `constant(x, shape)` needs a literal; a variable needs `broadcast_scalar`.
5. A **method chain** on a pointer tile (`p.reshape(shape![1]).broadcast(shape![128]).offset_tile(off)`) fails
   with "return type is missing a compiled tile type"; the same steps as separate typed `let`s compile
   (reproduced). `PointerTile::broadcast` keeps the rank, so a scalar pointer must be `reshape`d to rank 1
   first, and the reference's snippet (`ptr.broadcast(shape![128])` from a scalar pointer) does not type-check
   (reproduced, `--features docs-snippet`).
6. Mutable tensors, `scan_sum`, `reduce_max`, `tile_to_scalar` loop bounds, nested `for` loops with
   a dozen carried tiles, and masked atomics all compiled without trouble.

**Cloud session, prepared.** `scripts/cloud-session.sh` (after `scripts/remote-setup.sh`)
builds CUDA, oxide and cuTile on the VM, runs the grid check and parity for cuTile
(1, 10, 30, 100, 1000 steps), benchmarks all three with one protocol, profiles each `decide`
kernel with Nsight Compute, and bundles the results. Estimated 25-40 minutes on a
~$0.4/h VM, under $1 including setup. Not yet run: expect the first GPU execution of the
cuTile kernels to need debugging.

## 2026-10-21 (night): cuTile Rust runs on a GPU, passes parity, and is slower

Run on a rented RTX 4000 Ada (sm_89, driver 580.105, CUDA 13.3): `docs/results/rtx4000ada-2026-10-21/`.
The cloud box cost about $0.47 of the $10 credit in total across both attempts and the smoke test.

**Correctness.** The cuTile backend matches the CPU oracle at 1, 10, 30, 100 and 1000 steps (position error
7.6e-6 after 100 steps, the same as CUDA and cuda-oxide; zero neighbour-count mismatches; 0 outlier birds;
aggregates inside tolerance at 1000 steps), for both the default build and the tuned one below. This includes the
hand-written `atan`. Getting there took four fixes, three of them mine:
1. **Launch contract (mine).** An explicit `.grid()` must equal the tile count of the partitioned tensors, and my
   padded buffers disagreed ("launch grid (32,1,1) does not match the inferred partition grid (33,1,1)"). The scan
   now uses separate `block_totals` / `block_starts` buffers and no explicit grid.
2. **`constant(false, shape)` compiles to a tile of TRUE** (`constant <i1: 1>` in the Tile IR). My `band(a, b) =
   select(a, b, constant(false))` therefore computed `a -> b`, which let the bird itself into its own neighbour set:
   every one of 191 deciding birds had exactly one extra neighbour. Found by reading the generated IR, not by guessing.
   `false` now comes from a comparison. **Already reported upstream as NVIDIA/cuda-rust #1469, with a fix proposed in
   the still-open PR #1470** (opened 10 October, before I hit it). I reproduced it with a compile-only program
   (`upstream-repros/cutile-constant-false`) and checked that PR #1470's head (`d23a842`) fixes it.
3. **A sign error in my `body_up` cross product** (`integrate`): small velocity error, only for non-level flight.
4. **A regex bug in my generator** that left one `128` literal in non-128 tiles, which made every smaller tile fail
   parity until fixed. Parity caught it before any time was believed.

**Speed.** RTX 4000 Ada, mean ms per step, `decide` (total step in brackets), 3 runs each, spreads under 3 %:

| Birds | CUDA C++ | cuda-oxide | cuTile, defaults | cuTile, tuned |
|---|---|---|---|---|
| 100,000 | 1.24 (1.29) | 1.64 (1.69) | 5.57 (5.63) | **2.59** (2.65) |
| 1,000,000 | 9.46 (9.92) | 12.30 (12.76) | 50.6 (51.0) | **23.3** (23.6) |

Against CUDA C++ the `decide` kernel is 1.3x (oxide), 4.5x / 5.4x (cuTile defaults) and 2.1x / 2.5x (cuTile tuned) at
100k / 1M birds. The regular passes go the other way: at 1M birds cuTile's `integrate` is 0.126-0.135 ms against
CUDA's 0.200 ms and its scan 0.040 against 0.076 ms; histogram and scatter are level.

**Tuning, with parity checked for every setting** (`decide` at 100k birds):
- Tile size (default hints): 16 -> 18.9 ms, 32 -> 13.8, 64 -> 8.8, **128 -> 5.57**, 256 -> 8.7. Smaller tiles, which I
  expected to cut idle lanes, are slower. 128 is the optimum.
- `num_worker_warps_per_cta`: 2 -> 5.60, default 4 -> 5.56, 8 -> 11.6, 16 -> 11.8.
- **Occupancy hint** (`CompileOptions::occupancy`): none 5.56, 2 -> 5.56, 4 -> 4.45, 6 -> 2.82, 8 -> 2.76,
  **10 -> 2.59**, 12 -> 2.69, 14 -> 3.27, 16 -> 3.26, 24 -> 4.23. Re-checked per tile size: 64 -> 3.9 at best, 256 -> 5.6.
  Best configuration: tile 128, hint 10 (run-to-run noise between 10 and 12 is about 0.05 ms).

**What Nsight Compute shows** (one warmed-up `decide` launch, 100k birds, same input):

| | CUDA C++ | cuda-oxide | cuTile defaults | cuTile tuned |
|---|---|---|---|---|
| duration | 1.26 ms | 1.65 ms | 5.44 ms | 2.56 ms |
| threads per block | 256 | 256 | 128 | 128 |
| registers per thread | 62 | 64 | **246** | **48** |
| theoretical occupancy | 66.7 % | 66.7 % | **16.7 %** | 83.3 % |
| achieved active warps | 43.7 % | 44.2 % | 16.4 % | 67.3 % |
| instructions executed | 143 M | 254 M | **1,077 M** | 692 M |
| issue slots active | 35.9 % | 47.7 % | 50.2 % | 70.2 % |
| global-load sectors | 13.7 M | 13.7 M | **32.5 M** | 32.5 M |
| L1 / L2 hit rate | 43 / 99 % | 43 / 98 % | 20 / 100 % | 19 / 100 % |

- By default the assembler picks 246 registers, so only 2 blocks of 128 threads fit per SM (16.7 % occupancy). The
  occupancy hint makes it use 48 registers instead; that is where the 2.15x comes from.
- Even tuned, cuTile executes **4.8x the instructions** of CUDA (and 2.7x those of oxide) and issues **2.4x the
  global-load sectors**, at 70 % issue utilisation. So it is now bound by instruction count, not by occupancy.
- **What this does not explain.** I did not attribute the extra instructions. Candidates: every lane of a 128-bird
  tile running to the largest cell span and bucket size in the tile; four separate scalar gathers per candidate where
  CUDA does one `float4` load (the 2.4x sector count fits this); full-width masked arithmetic. The tile-size sweep
  argues against idle-lane waste being the whole story, since smaller tiles were slower, but it does not isolate
  anything. Treat these as hypotheses.
- Static register counts from the offline compile (246 for `decide`) matched the runtime profile exactly.

**Caveats.** One GPU model, one driver, one workload, one cuda-rust revision. The tuned cuTile result used a search
over three knobs on this exact input; CUDA and oxide got a far smaller tuning budget (oxide: `launch_bounds` only),
so the comparison favours cuTile if anything. The wgpu backend was not run on this GPU. The Nsight numbers are one
launch per variant.

**Other DSL findings from this session** (cuda-rust `e027ac3`): `async_on` lets a whole step be queued and
synchronised once (needed for event timing); the borrowed partition `(&mut t).partition(..)` writes in place;
`ncu` on a cuTile kernel works unmodified (kernel name `decide_entry`).

**Next:** the write-up; decide what to do with the remaining cuTile instruction-count gap (a gather-layout change, `float4`-style loads, is the obvious
experiment).

## 2026-10-21 (late): wgpu on the RTX 4000 Ada

A second session on the same host (RTX 4000 Ada, driver 580.105.08), `scripts/cloud-wgpu.sh`: `docs/results/rtx4000ada-2026-10-21/wgpu-session/`.
wgpu reported `NVIDIA RTX 4000 Ada Generation on Vulkan (DiscreteGpu)`. The earlier assumption that the VM had no Vulkan was
never tested; it works once `libvulkan1` is installed (the NVIDIA ICD ships with the driver).

**Parity.** Passes at 1, 10, 30, 100 and 1000 steps with results identical to the laptop run, including the same single
outlier bird at 100 steps (`dvel 1.966e-3`, `dpos 2.136e-4`): the wgpu result does not depend on the GPU here.

**Speed**, mean ms per step, 3 runs (first wgpu run at 100k was 7 % slower; medians below), CUDA C++ re-measured in the same session:

| Birds | CUDA C++ `decide` | wgpu `decide` | wgpu / CUDA | CUDA step | wgpu step |
|---|---|---|---|---|---|
| 100,000 | 1.236 | 1.543 | 1.25 | 1.283 | 1.605 |
| 1,000,000 | 9.441 | 12.34 | 1.31 | 9.897 | 13.0 |

CUDA reproduces the first session (1.229-1.238 ms; 9.36-9.52 ms), so the sessions are comparable. The grid passes are also
slower in wgpu at 1M birds (histogram 0.142 vs 0.103 ms, scan 0.116 vs 0.076, scatter 0.206 vs 0.080); `integrate` matches
(0.194 vs 0.199). **This changes the earlier summary**: wgpu is 3-7 % behind CUDA C++ on the Turing laptop GPU but 25-31 % behind
on the Ada GPU, level with cuda-oxide there. There are no counters for the wgpu shaders, so the cause is unknown.

**Process notes.** NVIDIA's apt mirror was mid-sync for several minutes ("File has unexpected size"); `scripts/remote-setup.sh`
now retries `apt-get update`. A remote `setsid nohup ... &` over SSH can leave the local client attached; poll a log instead of
waiting on the command.
