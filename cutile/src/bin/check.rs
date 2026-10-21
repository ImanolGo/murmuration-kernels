//! Offline compile check: lowers every kernel to Tile IR and then to a cubin with
//! `tileiras`, for a target architecture, with no GPU and no driver.
//!
//!   cargo run --bin check -- [--arch sm_89] [--dump-ir] [--cubin-dir DIR]

use cutile_compiler::compile_api::KernelCompiler;
use flock_cutile::flock_kernels;

/// (kernel name, generic arguments, stride arguments) for every entry point.
type Spec = (&'static str, Vec<String>, Vec<(&'static str, &'static [i32])>);

fn specs() -> Vec<Spec> {
    let b: Vec<String> = vec![];
    vec![
        ("grid_clear", vec![], vec![("counts", &[1])]),
        (
            "grid_histogram",
            vec![],
            vec![("bucket", &[1]), ("px", &[1]), ("py", &[1]), ("pz", &[1])],
        ),
        (
            "grid_scan_blocks",
            vec![],
            vec![("offsets", &[1]), ("block_sums", &[1]), ("counts", &[1])],
        ),
        ("grid_scan_totals", vec![], vec![("block_sums", &[1])]),
        (
            "grid_scan_add",
            vec![],
            vec![("offsets", &[1]), ("counts", &[1]), ("block_sums", &[1])],
        ),
        ("grid_scatter", vec![], vec![("bucket", &[1])]),
        ("decide", vec![], vec![]),
        (
        "integrate",
        b,
        vec![
            ("px", &[1]), ("py", &[1]), ("pz", &[1]), ("vx", &[1]), ("vy", &[1]), ("vz", &[1]),
            ("roll", &[1]), ("spin", &[1]), ("sx", &[1]), ("sy", &[1]), ("sz", &[1]),
        ],
    ),
    ]
}

fn main() {
    let mut arch = "sm_89".to_string();
    let (mut dump_ir, mut cubin_dir) = (false, None);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--arch" => arch = args.next().expect("--arch needs a value"),
            "--dump-ir" => dump_ir = true,
            "--cubin-dir" => cubin_dir = args.next(),
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let mut failed = 0;
    for (name, generics, strides) in specs() {
        let strides: Vec<(&str, &[i32])> = strides.iter().map(|(n, s)| (*n, *s)).collect();
        let result = KernelCompiler::new(flock_kernels::__module_ast_self, "flock_kernels", name)
            .generics(generics.clone())
            .strides(&strides)
            .target(&arch)
            .compile();
        match result {
            Ok(artifacts) => {
                if dump_ir {
                    println!("{}", artifacts.ir_text());
                }
                let bytes = artifacts.bytecode().map(|b| b.len()).unwrap_or(0);
                println!("ok    {name:<20} {generics:?}  Tile IR bytecode {bytes} bytes");
                if let Some(dir) = &cubin_dir {
                    let _ = std::fs::create_dir_all(dir);
                    if let Ok(bc) = artifacts.bytecode() {
                        let path = format!("{dir}/{name}.tilebc");
                        std::fs::write(&path, bc).expect("write bytecode");
                    }
                }
            }
            Err(e) => {
                failed += 1;
                println!("FAIL  {name:<20} {generics:?}\n      {e}");
            }
        }
    }
    std::process::exit(i32::from(failed > 0));
}
