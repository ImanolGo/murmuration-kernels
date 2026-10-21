# The same starling flock on the GPU, four ways

CUDA C++, cuda-oxide, wgpu and cuTile Rust, measured on a laptop GPU and on a rented one.

I wanted to know what writing GPU code in Rust costs today compared with CUDA C++, measured on one workload, with the same algorithm in every language, checked against a reference and timed the same way.

The workload comes from murmuration-gpu, a starling flock simulation I wrote in Rust with wgpu. Each bird follows rules about its nearest neighbours, so every step needs a spatial neighbour search. That search takes 96 to 98 percent of the step. It gathers from all over memory and loops a different number of times for every bird, which is not an obvious fit for a tile-based model. That makes it a hard test for NVIDIA's two Rust routes: cuda-oxide, which compiles SIMT kernels written in Rust, and cuTile Rust, which compiles tile programs. I also kept wgpu, since the simulation already used it, and CUDA C++ as the baseline.

Both NVIDIA projects live in one repository, NVIDIA/cuda-rust, and both call themselves early-stage research software. Everything below is pinned to one revision of it (`e027ac3`, 20 October 2026), so treat the results as a snapshot.

## What I measured

The step is a cut-down version of the murmuration-gpu model, which follows Hildenbrandt, Carere and Hemelrijk (2010). Each bird finds its neighbours through a spatial hash grid, then computes separation, cohesion, alignment, a vertical pull toward a preferred height, speed control and an adaptive perception radius. A banked-flight update moves it. I left out the falcon, the random force and the horizontal roost, so that every backend does the same work and the runs are repeatable. The full list of what is kept and dropped is in `docs/SPEC.md`.

The comparison follows four rules.

A CPU reference in Rust defines the behaviour. Every backend reads the same starting file and has to reproduce the reference. After 1, 10 and 30 steps every bird must agree within 1e-3 and every neighbour count must match. After 100 steps up to two of 2,000 birds may exceed that, because a bird sitting on a cell boundary can land on the other side under different rounding. After 1,000 steps only the averages are compared.

I relaxed that 100-step rule once, and I did it after seeing a failure. The wgpu backend had one bird of 2,000 off by 2e-3. I traced it step by step: the bird was within 3e-7 of the reference until step 73, and then its centrality differed while its neighbour count stayed the same. Centrality uses a search radius twice as wide, and I think a bird sat almost exactly on that boundary and was counted on opposite sides by two implementations that round differently. I did not identify the bird, but it looks like a threshold flip and not a bug in either backend, so the rule now allows a few of them while capping their size. CUDA and cuda-oxide pass the old rule too.

Every backend uses the same neighbour grid: a histogram of cells, a prefix scan, and a scatter. My first CUDA version sorted with CUB, but a Rust kernel cannot call CUB, and comparing a library sort with a hand-written one would have measured the sort. I replaced it.

Times come from GPU events or timestamp queries, never from wall clocks. Each run does 20 warm-up steps and 20 timed ones, and I ran each configuration three times. The birds are scattered at 0.5 per cubic metre, the same protocol as the benchmark in murmuration-gpu.

## Results

The main number is the time for the neighbour search kernel, `decide`, per step. First the laptop GPU, a GTX 1650 Max-Q (sm_75):

| Birds | CUDA C++ | cuda-oxide, default | cuda-oxide, tuned | wgpu |
|---|---|---|---|---|
| 100,000 | 7.4 ms | 9.2 to 9.7 ms | 7.9 ms | 8.0 ms |
| 1,000,000 | 97 ms | 106 ms | 100 ms | 100 to 101 ms |

"Tuned" here means one launch-bounds attribute and removing slice bounds checks. cuTile Rust cannot run on this card, because it needs compute capability 8.0 or higher.

For that I rented a GPU: an RTX 4000 Ada (sm_89) on Vast.ai, for about $0.24 an hour. All the cloud work, which included an earlier smoke test and one host I abandoned because it was slow to boot, cost about $0.47.

| Birds | CUDA C++ | cuda-oxide | cuTile Rust, defaults | cuTile Rust, tuned |
|---|---|---|---|---|
| 100,000 | 1.24 ms | 1.64 ms | 5.57 ms | 2.59 ms |
| 1,000,000 | 9.46 ms | 12.3 ms | 50.6 ms | 23.3 ms |

The three runs of each cell agreed within 3 percent. wgpu is missing from the second table because the rented machine had no Vulkan.

## cuda-oxide: close, and the gap has causes

After fixing one wrong API name in my host code, the whole port built and ran on the Turing laptop card. The grid kernels and `integrate` ran at the same speed as CUDA. `decide` was a different matter. On the laptop the default build was 24 to 30 percent slower at 100,000 birds and 9 percent slower at a million. With the two changes described below it was 7 percent and 3 percent slower. On the Ada card, with the launch-bounds attribute already in place, it was about 30 percent slower at both sizes.

The first cause was registers. The CUDA kernel uses 61 registers per thread, so four blocks fit on each multiprocessor. The Rust one used 72, so only three fit. To check that this was the whole story I launched the CUDA kernel, unchanged, with extra shared memory so that only three blocks fit. It slowed to 9.3 ms, which is where cuda-oxide was. Adding `#[launch_bounds(256, 4)]` to the Rust kernel pulled it back to 64 registers at the price of a small spill, and recovered about half the gap.

The second cause shows up in Nsight Compute once occupancy is equal. Memory traffic was identical to the sector, but the Rust kernel executed 77 percent more instructions (259 million against 146 million on the laptop, 254 million against 143 million on the Ada card). On the laptop the kernel is limited by latency, so most of that was hidden and cost 14 percent in time. On the Ada card the register counts were 64 and 62 and the occupancy was the same, so the whole 30 percent gap is this instruction count. Two parts of it I could pin down. Rust checks every slice index, and removing the checks took 9 percent of the instructions away. And the generated PTX contains no `min.f32` or `max.f32` at all: float min and max compile to compare-and-select, where CUDA uses the hardware instruction. I tried both `libm::fmaxf` and `f32::max`; they compile the same way. The rest of the extra instructions I did not attribute.

I also ran into one thing the backend rejects: `f32::to_int_unchecked` fails with "invalid input program". I have not reduced either of these to a bug report yet.

## wgpu: nearly as fast, and opaque

The same kernels in WGSL, going through SPIR-V and the Vulkan driver, ran about 7 percent behind CUDA C++ at 100,000 birds and 3 percent behind at a million. That is level with tuned cuda-oxide.

I cannot say why. Nsight Compute does not profile Vulkan compute, so I have no register counts or instruction counts for these shaders, and I did not read the counters from the other backends across to them.

## cuTile Rust: a different shape of program

A cuTile kernel is a program over a tile of values, which the compiler maps to threads. That changes how a neighbour search has to be written.

In CUDA each bird loops over its own candidates. In cuTile one program handles 128 birds, and a loop cannot run a different number of times per bird. So each walk runs to the largest cell span and the largest bucket in the whole tile, and every lane is masked. Birds with fewer candidates idle while the busiest one finishes. Neighbour data is fetched with masked pointer gathers, one per field and candidate, where CUDA reads one `float4`.

The regular kernels came out well. The prefix scan is a few lines per kernel with `scan_sum`, instead of a hand-written shared-memory loop. At a million birds cuTile's scan took 0.040 ms against CUDA's 0.076, and `integrate` took 0.13 ms against 0.20. The irregular part is where it lost.

Getting it correct took four fixes, three of them my own mistakes, and one that was not.

- A launch grid has to equal the tile count of the tensors you partition. My padded buffers disagreed, and the first GPU run stopped before any kernel executed.
- A sign error in my own cross product made small velocity errors for any bird not flying level.
- A regular expression in my kernel generator left one literal `128` in place for smaller tiles, so every smaller tile failed the reference check. The check caught it before I believed any speed.
- `constant(false, shape)` compiles to a tile of true. The Tile IR shows `constant <i1: 1>`. My `and` helper was `select(a, b, false)`, which became `select(a, b, true)`, which computes "a implies b". Every bird counted itself as its own neighbour: all 191 birds that decided on the first step had exactly one extra. I found it by reading the generated IR after looking at which birds were wrong. The workaround is to build false from a comparison.

That last one I consider a compiler bug. I am still reducing it to a minimal reproduction.

The DSL also has no `atan`, so the bank-angle target uses a polynomial I wrote, and a generic device function returns a type with its shape unsubstituted, so the tile size is a literal everywhere. The full list is in `docs/progress.md`. One thing worked better than I expected: the compile-only API lowered every kernel to an sm_89 cubin on the laptop with no GPU, so I could write and check all of it before renting anything.

With the defaults, `decide` took 5.57 ms, 4.5 times CUDA's 1.24. Nsight Compute showed why immediately. The compiler chose 246 registers per thread, so only two blocks of 128 threads fit on a multiprocessor, an occupancy of 16.7 percent. The occupancy hint in the compile options changes that. With a hint of 10 the same code uses 48 registers, occupancy goes to 83 percent, and the time falls to 2.59 ms. I swept the hint, the tile size and the number of warps per block, and checked the result against the reference for every setting.

| Setting | `decide` at 100,000 birds |
|---|---|
| tile 16 / 32 / 64 / **128** / 256, default hints | 18.9 / 13.8 / 8.8 / **5.57** / 8.7 ms |
| warps per block 2 / 4 (default) / 8 / 16 | 5.60 / 5.56 / 11.6 / 11.8 ms |
| occupancy hint 4 / 6 / 8 / **10** / 12 / 16 / 24 | 4.4 / 2.8 / 2.8 / **2.6** / 2.7 / 3.3 / 4.2 ms |

I expected smaller tiles to help, because they would cut the idle lanes. They were slower. Past a hint of 12 the time rises again, probably from register spills, though I did not check.

Tuned, cuTile is 2.1 times CUDA at 100,000 birds and 2.5 times at a million. It executes 4.8 times the instructions (692 million against 143 million) and issues 2.4 times the global-load sectors (32.5 million against 13.7 million), at 70 percent issue utilisation. That suggests it is now limited by the amount of work it does.

| One `decide` launch, 100,000 birds | CUDA C++ | cuda-oxide | cuTile defaults | cuTile tuned |
|---|---|---|---|---|
| Registers per thread | 62 | 64 | 246 | 48 |
| Theoretical occupancy | 66.7 % | 66.7 % | 16.7 % | 83.3 % |
| Instructions executed | 143 M | 254 M | 1,077 M | 692 M |
| Global-load sectors | 13.7 M | 13.7 M | 32.5 M | 32.5 M |
| Duration | 1.26 ms | 1.65 ms | 5.44 ms | 2.56 ms |

I have candidate explanations for the extra instructions: lockstep loops to the tile-wide maximum, the four separate gathers, full-width arithmetic on masked lanes. The tile-size results argue against idle lanes being the whole story. None of this is isolated, so I am calling them hypotheses.

## What this does not show

It is one workload, a gather-heavy neighbour search that I picked partly because it is awkward for tiles. I did not test dense kernels such as matrix multiply or attention, which are what cuTile's own examples show. They would probably tell a different story, and the regular kernels here hint at that.

It is two GPUs, one driver family and one revision of an alpha project. A newer compiler may change every number in this article.

The tuning was not even. I searched three settings for cuTile on the exact input I then reported. CUDA and cuda-oxide got a launch-bounds attribute and little else. If anything that favours cuTile.

Each Nsight figure is one launch per variant. The `constant(false)` miscompile, the min and max lowering and the missing `atan` are facts about this revision of an alpha project.

I did not run wgpu on the rented GPU, because the virtual machine had no Vulkan. wgpu's numbers are from the laptop only.

## Reproduce

Everything is in the `flock-four-ways` repository: the CPU reference, one backend per directory, `scripts/parity.sh` for the checks, and `scripts/cloud-session.sh` for the rented-GPU run. The raw logs, the four Nsight reports and a README for them are in `docs/results/rtx4000ada-2026-10-21/`, and `docs/progress.md` has the dated log of every step, including the ones that went wrong. [TODO: add the public link when the repository is published.]
