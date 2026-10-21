//! Three cuTile Rust DSL problems, each compiled on its own with the compile-only API (no GPU).
//!   cargo run                      -> prints ok/FAIL for each kernel (exit 1 if any fails)
//!   cargo build --features docs-snippet   -> rustc type error from the reference's snippet

use cutile_compiler::compile_api::KernelCompiler;

#[cutile::module]
mod friction {
    use cutile::core::*;

    fn keep<const B: i32>(x: Tile<f32, { [B] }>) -> Tile<f32, { [B] }> {
        let one: Tile<f32, { [B] }> = broadcast_scalar(1.0f32, shape![B]);
        x * one
    }

    // 1a. A user device function called in ARGUMENT position (no typed `let` for its result).
    #[cutile::entry()]
    fn helper_in_argument(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        let t: Tile<f32, { [128] }> = a.load_tile(shape![128], [program_id(0)]);
        out.store(max_tile(keep::<128>(t), t));
    }

    // 1b. Same, with a typed `let`: expected to compile.
    #[cutile::entry()]
    fn helper_in_let(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        let t: Tile<f32, { [128] }> = a.load_tile(shape![128], [program_id(0)]);
        let k: Tile<f32, { [128] }> = keep::<128>(t);
        out.store(max_tile(k, t));
    }

    // 1c. A generic ENTRY that passes its own generic on to a generic device function.
    #[cutile::entry()]
    fn generic_entry<const B: i32>(out: &mut Tensor<f32, { [B] }>, a: &Tensor<f32, { [-1] }>) {
        let t: Tile<f32, { [B] }> = a.load_tile(shape![B], [program_id(0)]);
        let k: Tile<f32, { [B] }> = keep::<B>(t);
        out.store(max_tile(k, t));
    }

    // 2a. A method CHAIN on a pointer tile with no intermediate typed `let`.
    #[cutile::entry()]
    unsafe fn ptr_chain(p: *mut f32, out: &mut Tensor<f32, { [128] }>) {
        let r0: PointerTile<*mut f32, { [] }> = pointer_to_tile(p);
        let off: Tile<i32, { [128] }> = iota(shape![128]);
        let ptrs: PointerTile<*mut f32, { [128] }> =
            r0.reshape(shape![1]).broadcast(shape![128]).offset_tile(off);
        let (v, _t): (Tile<f32, { [128] }>, Token) = unsafe {
            load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>)
        };
        out.store(v);
    }

    // 2b. The same steps as separate typed `let`s with the method forms: expected to compile.
    #[cutile::entry()]
    unsafe fn ptr_steps(p: *mut f32, out: &mut Tensor<f32, { [128] }>) {
        let r0: PointerTile<*mut f32, { [] }> = pointer_to_tile(p);
        let r1: PointerTile<*mut f32, { [1] }> = r0.reshape(shape![1]);
        let rb: PointerTile<*mut f32, { [128] }> = r1.broadcast(shape![128]);
        let off: Tile<i32, { [128] }> = iota(shape![128]);
        let ptrs: PointerTile<*mut f32, { [128] }> = rb.offset_tile(off);
        let (v, _t): (Tile<f32, { [128] }>, Token) = unsafe {
            load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>)
        };
        out.store(v);
    }

    // 2c. The corrected form of the reference snippet (scalar pointer -> rank 1 -> wide -> offsets).
    #[cutile::entry()]
    unsafe fn docs_fixed(raw_ptr: *mut f32, out: &mut Tensor<f32, { [128] }>) {
        let offsets: Tile<i32, { [128] }> = iota(shape![128]);
        let ptr: PointerTile<*mut f32, { [] }> = pointer_to_tile(raw_ptr);
        let ptr1: PointerTile<*mut f32, { [1] }> = reshape_ptr(ptr, shape![1]);
        let wide: PointerTile<*mut f32, { [128] }> = broadcast_ptr(ptr1, shape![128]);
        let offset_ptrs: PointerTile<*mut f32, { [128] }> = wide.offset_tile(offsets);
        let (v, _t): (Tile<f32, { [128] }>, Token) = unsafe {
            load_ptr_tko(offset_ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>)
        };
        out.store(v);
    }

    // --- 3. The snippet in cutile-book/reference/dsl-api.md (PointerTile section) does not type-check. ---
    #[cfg(feature = "docs-snippet")]
    #[cutile::entry()]
    unsafe fn docs_snippet(p: *mut f32, out: &mut Tensor<f32, { [128] }>) {
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(p);
        let offsets: Tile<i32, { [128] }> = iota(shape![128]);
        // as written in the reference: `ptr.broadcast(shape![128]).offset_tile(offsets)`
        let ptrs: PointerTile<*mut f32, { [128] }> = base.broadcast(shape![128]).offset_tile(offsets);
        let (v, _t): (Tile<f32, { [128] }>, Token) = unsafe {
            load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>)
        };
        out.store(v);
    }
}

fn try_compile(name: &str, generics: &[&str], strides: &[(&str, &[i32])]) -> bool {
    match KernelCompiler::new(friction::__module_ast_self, "friction", name)
        .generics(generics.iter().map(|g| g.to_string()).collect())
        .strides(strides)
        .target("sm_89")
        .compile()
    {
        Ok(_) => {
            println!("ok    {name}");
            true
        }
        Err(e) => {
            println!("FAIL  {name}\n      {e}");
            false
        }
    }
}

fn main() {
    let both: &[(&str, &[i32])] = &[("out", &[1]), ("a", &[1])];
    let out_only: &[(&str, &[i32])] = &[("out", &[1])];
    let mut all_ok = true;
    all_ok &= try_compile("helper_in_argument", &[], both);
    all_ok &= try_compile("helper_in_let", &[], both);
    all_ok &= try_compile("generic_entry", &["128"], both);
    all_ok &= try_compile("ptr_chain", &[], out_only);
    all_ok &= try_compile("ptr_steps", &[], out_only);
    all_ok &= try_compile("docs_fixed", &[], out_only);
    std::process::exit(i32::from(!all_ok));
}
