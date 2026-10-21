//! `flock-ref`: generate a starting flock, advance it on the CPU, compare two
//! `.flk` files, and print flock statistics.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::process::ExitCode;

use flock_ref::Flock;
use flock_ref::model::forward;

const USAGE: &str = "usage:
  flock-ref gen   --birds N [--seed S] --out FILE
  flock-ref step  --in FILE --steps K --out FILE     (brute force; use N <= ~5000)
  flock-ref diff  A.flk B.flk [--pos-tol X] [--vel-tol X] [--allow-outliers N] [--outlier-max X]
  flock-ref stats FILE
  flock-ref statdiff A.flk B.flk     (aggregate agreement for long, chaotic runs)";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("flock-ref: {e}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|k| args.get(k + 1))
        .map(String::as_str)
}

fn need<'a>(args: &'a [String], name: &str) -> Result<&'a str, String> {
    opt(args, name).ok_or_else(|| format!("missing {name}"))
}

fn load(path: &str) -> Result<Flock, String> {
    let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    Flock::read(BufReader::new(f)).map_err(|e| format!("{path}: {e}"))
}

fn save(flock: &Flock, path: &str) -> Result<(), String> {
    let f = File::create(path).map_err(|e| format!("{path}: {e}"))?;
    flock
        .write(BufWriter::new(f))
        .map_err(|e| format!("{path}: {e}"))
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    match args.first().map(String::as_str) {
        Some("gen") => {
            let n: usize = need(args, "--birds")?.parse().map_err(|_| "bad --birds")?;
            let seed: u64 = opt(args, "--seed")
                .unwrap_or("1")
                .parse()
                .map_err(|_| "bad --seed")?;
            save(&Flock::scatter(n, seed), need(args, "--out")?)?;
            Ok(ExitCode::SUCCESS)
        }
        Some("step") => {
            let mut flock = load(need(args, "--in")?)?;
            let steps: u32 = need(args, "--steps")?.parse().map_err(|_| "bad --steps")?;
            for _ in 0..steps {
                flock.step();
            }
            save(&flock, need(args, "--out")?)?;
            Ok(ExitCode::SUCCESS)
        }
        Some("stats") => {
            print_stats(&load(args.get(1).ok_or("missing FILE")?)?);
            Ok(ExitCode::SUCCESS)
        }
        Some("diff") => diff(args),
        Some("statdiff") => statdiff(args),
        _ => Err("missing or unknown command".into()),
    }
}

/// Aggregate statistics: mean speed, neighbours, radius and polarization.
fn stats(f: &Flock) -> [f32; 4] {
    let n = f.birds.len() as f32;
    let speed = f.birds.iter().map(|b| b.vel.length()).sum::<f32>() / n;
    let nb = f.birds.iter().map(|b| b.neighbours as f32).sum::<f32>() / n;
    let radius = f.birds.iter().map(|b| b.radius).sum::<f32>() / n;
    let polar = (f.birds.iter().map(|b| forward(b.vel)).sum::<glam::Vec3>() / n).length();
    [speed, nb, radius, polar]
}

fn print_stats(f: &Flock) {
    let [speed, nb, radius, polar] = stats(f);
    println!("birds {}  time {:.3} s", f.birds.len(), f.time);
    println!(
        "mean speed {speed:.4} m/s  mean neighbours {nb:.3}  mean radius {radius:.3} m  polarization {polar:.4}"
    );
}

/// Pointwise agreement is lost after ~100 steps: a bird near its radius edge
/// flips a neighbour differently under a different summation order, and the
/// flock amplifies it. The aggregates must still agree. The tolerances are
/// this project's own choice, not a published criterion (docs/SPEC.md).
fn statdiff(args: &[String]) -> Result<ExitCode, String> {
    let a = load(args.get(1).ok_or("missing A")?)?;
    let b = load(args.get(2).ok_or("missing B")?)?;
    if a.birds.len() != b.birds.len() {
        return Err("files hold different bird counts".into());
    }
    let (sa, sb) = (stats(&a), stats(&b));
    let tol = [0.02, 0.1, 0.01, 0.005];
    let names = [
        "mean speed",
        "mean neighbours",
        "mean radius",
        "polarization",
    ];
    let mut ok = true;
    for k in 0..4 {
        let d = (sa[k] - sb[k]).abs();
        ok &= d <= tol[k];
        println!(
            "{:<16} A {:>9.4}  B {:>9.4}  |d| {:.4}  (tol {})",
            names[k], sa[k], sb[k], d, tol[k]
        );
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

/// Exit code 0 when every compared quantity is inside tolerance, 1 otherwise.
fn diff(args: &[String]) -> Result<ExitCode, String> {
    let a = load(args.get(1).ok_or("missing A")?)?;
    let b = load(args.get(2).ok_or("missing B")?)?;
    if a.birds.len() != b.birds.len() {
        return Err("files hold different bird counts".into());
    }
    let pos_tol: f32 = opt(args, "--pos-tol")
        .unwrap_or("1e-3")
        .parse()
        .map_err(|_| "bad --pos-tol")?;
    let vel_tol: f32 = opt(args, "--vel-tol")
        .unwrap_or("1e-3")
        .parse()
        .map_err(|_| "bad --vel-tol")?;
    let (mut dp, mut dv, mut dr, mut dc) = (0f32, 0f32, 0f32, 0f32);
    let (mut ds, mut droll, mut dspin) = (0f32, 0f32, 0f32);
    let allow: usize = opt(args, "--allow-outliers")
        .unwrap_or("0")
        .parse()
        .map_err(|_| "bad --allow-outliers")?;
    let outlier_max: f32 = opt(args, "--outlier-max")
        .unwrap_or("1e-2")
        .parse()
        .map_err(|_| "bad --outlier-max")?;
    let mut outliers = 0usize;
    let mut nb_mismatch = 0usize;
    for (x, y) in a.birds.iter().zip(&b.birds) {
        let bp = (x.pos - y.pos).abs().max_element();
        let bv = (x.vel - y.vel).abs().max_element();
        dp = dp.max(bp);
        dv = dv.max(bv);
        outliers += usize::from(bp > pos_tol || bv > vel_tol);
        dr = dr.max((x.radius - y.radius).abs());
        dc = dc.max((x.centrality - y.centrality).abs());
        ds = ds.max((x.steering - y.steering).abs().max_element());
        droll = droll.max((x.roll - y.roll).abs());
        dspin = dspin.max((x.spin - y.spin).abs());
        nb_mismatch += usize::from(x.neighbours != y.neighbours);
    }
    println!(
        "birds {}  time A {:.4} B {:.4}",
        a.birds.len(),
        a.time,
        b.time
    );
    println!(
        "max |dpos| {dp:.3e}  max |dvel| {dv:.3e}  max |dradius| {dr:.3e}  max |dcentrality| {dc:.3e}"
    );
    println!(
        "neighbour-count mismatches {nb_mismatch} of {}",
        a.birds.len()
    );
    // Strict by default. With --allow-outliers N, up to N birds may exceed the
    // tolerance (an isolated threshold flip, see docs/SPEC.md) provided none
    // exceeds --outlier-max.
    let ok = if allow == 0 {
        dp <= pos_tol && dv <= vel_tol && nb_mismatch == 0
    } else {
        outliers <= allow && dp <= outlier_max && dv <= outlier_max && nb_mismatch == 0
    };
    println!(
        "(informational) max |dsteering| {ds:.3e}  max |droll| {droll:.3e}  max |dspin| {dspin:.3e}"
    );
    if allow > 0 {
        println!("birds above tolerance: {outliers} (allowed {allow}, each below {outlier_max})");
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}
