//! Float min and max in a cuda-oxide kernel. CUDA C++ compiles `fmaxf`/`fminf` to the PTX
//! instructions `max.f32`/`min.f32`; here every route compiles to setp + selp instead.
//! Build the PTX with `cargo oxide inspect --arch sm_89` and read `oxide_minmax.ptx`.

use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// `f32::max` and `f32::min` (the `llvm.maxnum`/`minnum` intrinsics).
    #[kernel]
    pub fn std_minmax(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = a[i].max(b[i]) + a[i].min(b[i]);
        }
    }

    /// `libm::fmaxf` and `libm::fminf`.
    #[kernel]
    pub fn libm_minmax(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = libm::fmaxf(a[i], b[i]) + libm::fminf(a[i], b[i]);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only the PTX matters for the report; this launches both kernels so the crate is a real program.
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let a = DeviceBuffer::from_host(&stream, &[1.0f32, 5.0, -2.0, 7.0])?;
    let b = DeviceBuffer::from_host(&stream, &[3.0f32, 4.0, -1.0, 8.0])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 4)?;
    let module = kernels::load(&ctx)?;
    let cfg = cuda_core::simt::LaunchConfig::for_num_elems(4);
    unsafe { module.std_minmax(stream.as_ref(), cfg, &a, &b, &mut out)? };
    println!("std_minmax: {:?}", out.to_host_vec(&stream)?);
    unsafe { module.libm_minmax(stream.as_ref(), cfg, &a, &b, &mut out)? };
    println!("libm_minmax: {:?}", out.to_host_vec(&stream)?);
    Ok(())
}
