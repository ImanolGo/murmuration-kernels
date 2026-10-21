# cuda-oxide: float min/max compile to compare-and-select

Found while profiling a neighbour-search kernel (`docs/progress.md` in the parent repository). Compile-only: no GPU is
needed to see it.

```
CUDA_HOME=/opt/cuda cargo oxide inspect --arch sm_89     # writes oxide_minmax.ptx
grep -cE 'max\.f32|min\.f32' oxide_minmax.ptx            # 0
grep -cE 'setp|selp' oxide_minmax.ptx                    # 6 setp, 8 selp for four min/max operations
nvcc -ptx -arch=sm_89 reference.cu                       # CUDA C++: max.f32 and min.f32
```

`f32::max`/`f32::min` and `libm::fmaxf`/`libm::fminf` both lower to `setp` + `selp`. In the full kernel the CUDA C++
version has 20 `FMNMX` instructions in its SASS and the cuda-oxide version has none. cuda-rust `e027ac3`
(2026-10-20), nightly-2026-10-11, CUDA 13.3 (also seen with 13.4).
