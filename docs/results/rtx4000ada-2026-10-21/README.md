# RTX 4000 Ada session, 2026-10-21

One rented Vast.ai VM: **NVIDIA RTX 4000 Ada Generation** (sm_89, 20 GB), driver 580.105.08, CUDA
toolkit 13.3 (runtime 13.3 on a CUDA-13.0 driver), Ubuntu 22.04, 5 CPU cores, `ncu` 2026.2.
Seed 1, 20 warm-up + 20 timed steps, mean GPU ms per step from CUDA events, 3 runs each.

- `bench-table-4000ada.txt`: the benchmark table (all runs listed, medians for the small passes).
- `ncu-summary4-4000ada.txt`: Nsight Compute metrics for one warmed-up `decide` launch at 100k birds.
- `profiles/*.ncu-rep.xz`: the four raw reports (`xz -d`, then open with `ncu-ui` or `ncu -i`).
- `logs/`: raw benchmark and parity output.

Which build is which:
- `cuda`: `cuda/flock.cu`, built for sm_89. `oxide`: `oxide/` with `#[launch_bounds(256, 4)]`, bounds-checked.
- `cutile`: the cuTile backend with the compiler's defaults. `cutiletuned`: the same binary run with
  `FLOCK_CUTILE_OCC=10` (occupancy hint 10 for `decide`). Both use a 128-bird tile.
- `wgpu-session/`: a second session on the same host, running the wgpu backend and CUDA C++ again as the reference
  (the Vulkan driver was present: `libvulkan1` and the NVIDIA ICD; wgpu reported `NVIDIA RTX 4000 Ada Generation on
  Vulkan (DiscreteGpu)`). Its CUDA numbers match the first session's. The wgpu step totals are sums of pass durations.

`logs/status.txt` is from the **first** scripted pass, in which every cuTile step failed on a launch-grid
mismatch; the cuTile logs here come from the fixed build run afterwards. `tuned-parity.txt` is the
parity ladder for the tuned configuration. The GPU's identifying `nvidia-smi -q` dump was left out.
