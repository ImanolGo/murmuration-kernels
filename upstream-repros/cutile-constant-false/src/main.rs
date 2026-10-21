//! `constant(false, shape)` inside a `#[cutile::module]` kernel must give a tile of false.
//! This compiles the kernel to Tile IR (no GPU) and checks the `i1` constant.

use cutile_compiler::compile_api::KernelCompiler;

#[cutile::module]
mod repro {
    use cutile::core::*;

    #[cutile::entry()]
    fn mask_kernel(out: &mut Tensor<i32, { [128] }>) {
        let no: Tile<bool, { [128] }> = constant(false, shape![128]);
        let yes: Tile<bool, { [128] }> = constant(true, shape![128]);
        let one: Tile<i32, { [128] }> = constant(1i32, shape![128]);
        let zero: Tile<i32, { [128] }> = constant(0i32, shape![128]);
        // Expected to store 0 in the first half of the program and 1 in the second.
        let a: Tile<i32, { [128] }> = select(no, one, zero);
        let b: Tile<i32, { [128] }> = select(yes, one, zero);
        out.store(a + b);
    }
}

fn main() {
    let art = KernelCompiler::new(repro::__module_ast_self, "repro", "mask_kernel")
        .strides(&[("out", &[1])])
        .target("sm_89")
        .compile()
        .expect("compile");
    let ir = art.ir_text();
    let consts: Vec<&str> = ir.lines().filter(|l| l.contains("constant <i1")).map(str::trim).collect();
    for c in &consts {
        println!("{c}");
    }
    // The kernel holds two bool constants, written `false` then `true`.
    let bits: Vec<&str> = consts.iter().filter_map(|l| l.split("i1: ").nth(1)).map(|t| &t[..1]).collect();
    println!("i1 constant values in source order: {bits:?} (expected [\"0\", \"1\"])");
    if bits != ["0", "1"] {
        println!("BUG: constant(false, ..) was not lowered to 0");
        std::process::exit(1);
    }
    println!("ok");
}
