# cuda-oxide: `f32::to_int_unchecked` fails to compile

Compile-only. The saturating cast `x as i32` builds; the unchecked conversion is rejected:

```
CUDA_HOME=/opt/cuda cargo oxide build --arch sm_89                        # ok
CUDA_HOME=/opt/cuda cargo oxide build --arch sm_89 --features unchecked   # error
```

Result with cuda-rust `e027ac3` (2026-10-20), nightly-2026-10-11, CUDA 13.3:
`Device codegen failed: PTX generation failed: Translation failed: ... core/src/convert/num.rs ... Compilation error: invalid input program.`

Why it matters: Rust's `f32 as i32` saturates and maps NaN to 0, which adds instructions to every float-to-int
conversion (a spatial hash computes several per candidate). `to_int_unchecked` is the usual way to avoid that.
