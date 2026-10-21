//! `f32::to_int_unchecked` in a cuda-oxide kernel. The checked cast `x as i32` compiles; the unchecked
//! one is rejected by the backend ("invalid input program" in core/src/convert/num.rs).
//! Build with `CUDA_HOME=/opt/cuda cargo oxide build --arch sm_89`.

use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn cell_index(x: &[f32], mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            // SAFETY: the test inputs are finite and far inside the i32 range.
            #[cfg(feature = "unchecked")]
            {
                *o = unsafe { x[i].to_int_unchecked::<i32>() };
            }
            #[cfg(not(feature = "unchecked"))]
            {
                *o = x[i] as i32;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let x = DeviceBuffer::from_host(&stream, &[1.5f32, -2.5, 100.0, 7.9])?;
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, 4)?;
    let module = kernels::load(&ctx)?;
    unsafe { module.cell_index(stream.as_ref(), cuda_core::simt::LaunchConfig::for_num_elems(4), &x, &mut out)? };
    println!("{:?}", out.to_host_vec(&stream)?);
    Ok(())
}
