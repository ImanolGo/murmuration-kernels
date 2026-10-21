//! Host runner for the cuTile Rust backend of the flock core step (docs/SPEC.md).
//!
//! Same command line and `.flk` files as the other backends:
//!   flock-cutile --in start.flk [--steps K] [--warmup W] [--out end.flk] [--check]
//! `--check` validates the spatial grid built in the last step (see `check_grid`).
//!
//! One step is queued on a single stream (clear, histogram, scan x3, scatter, decide,
//! integrate) and synchronised once; GPU events between the pass groups give the timing.
//! Needs a GPU with compute capability >= 8.0 and CUDA 13.2+.

use cutile::prelude::*;
use flock_cutile::flock_kernels::{
    decide, grid_clear, grid_histogram, grid_scan_add, grid_scan_blocks, grid_scan_totals,
    grid_scatter, integrate,
};
use flock_ref::{Bird, Flock, Vec3};

const TILE: usize = 128;
const SCAN_BLOCK: usize = 2048;
const DT: f32 = 0.005;

fn buckets_for(n: usize) -> usize {
    (n * 2).next_power_of_two().clamp(1 << 16, 1 << 22)
}

fn pad_to(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = None;
    let mut output = None;
    let (mut steps, mut warmup) = (1usize, 0usize);
    let mut check = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--in" => input = args.next(),
            "--out" => output = args.next(),
            "--steps" => steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--warmup" => warmup = args.next().ok_or("--warmup needs a value")?.parse()?,
            "--check" => check = true,
            _ => {
                eprintln!(
                    "usage: flock-cutile --in FILE [--steps K] [--warmup W] [--out FILE] [--check]"
                );
                std::process::exit(2);
            }
        }
    }
    let input = input.ok_or("missing --in")?;
    let warmup = warmup.min(steps);

    let start = Flock::read(std::io::BufReader::new(std::fs::File::open(&input)?))?;
    let n = start.birds.len();
    let n_pad = pad_to(n, TILE);
    let buckets = buckets_for(n);
    let scan_blocks = buckets / SCAN_BLOCK;
    // Bucket-sized arrays carry one extra scan block of padding: the scan-add program past
    // the last bucket tile writes offsets[buckets] = n, and partitions need whole tiles.
    let bucket_len = buckets + SCAN_BLOCK;

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    println!(
        "# flock_cutile: {} (cuda-rust e027ac3, tile {TILE})",
        device.name()?
    );
    println!("# {n} birds, {steps} steps ({warmup} warm-up, not timed)");

    // ---- upload (structure of arrays, padded to a whole tile) ---------------------------
    let col = |f: &dyn Fn(&Bird) -> f32| -> Vec<f32> {
        let mut v: Vec<f32> = start.birds.iter().map(f).collect();
        v.resize(n_pad, 0.0);
        v
    };
    let up = |v: Vec<f32>| -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
        Ok(api::copy_host_vec_to_device(&Arc::new(v)).sync_on(&stream)?)
    };
    let comp = |f: fn(Vec3) -> f32, g: fn(&Bird) -> Vec3| col(&move |b: &Bird| f(g(b)));
    let (mut px, mut py, mut pz) = (
        up(comp(|v| v.x, |b| b.pos))?,
        up(comp(|v| v.y, |b| b.pos))?,
        up(comp(|v| v.z, |b| b.pos))?,
    );
    let (mut vx, mut vy, mut vz) = (
        up(comp(|v| v.x, |b| b.vel))?,
        up(comp(|v| v.y, |b| b.vel))?,
        up(comp(|v| v.z, |b| b.vel))?,
    );
    let (sx, sy, sz) = (
        up(comp(|v| v.x, |b| b.steering))?,
        up(comp(|v| v.y, |b| b.steering))?,
        up(comp(|v| v.z, |b| b.steering))?,
    );
    let mut roll = up(col(&|b| b.roll))?;
    let mut spin = up(col(&|b| b.spin))?;
    let radius = up(col(&|b| b.radius))?;
    let next_decision = up(col(&|b| b.next_decision))?;
    let centrality = up(col(&|b| b.centrality))?;
    let neighbours: Tensor<i32> = {
        let mut v: Vec<i32> = start.birds.iter().map(|b| b.neighbours as i32).collect();
        v.resize(n_pad, 0);
        api::copy_host_vec_to_device(&Arc::new(v)).sync_on(&stream)?
    };
    let mut bucket: Tensor<i32> = api::zeros::<i32>(&[n_pad]).sync_on(&stream)?;
    let sorted: Tensor<i32> = api::zeros::<i32>(&[n_pad]).sync_on(&stream)?;
    let mut counts: Tensor<i32> = api::zeros::<i32>(&[bucket_len]).sync_on(&stream)?;
    let mut offsets: Tensor<i32> = api::zeros::<i32>(&[bucket_len]).sync_on(&stream)?;
    let mut block_sums: Tensor<i32> = api::zeros::<i32>(&[SCAN_BLOCK]).sync_on(&stream)?;

    let (n_i, mask_i) = (n as i32, (buckets - 1) as i32);
    let birds_grid = (n_pad / TILE) as u32;
    let events: Vec<_> = (0..steps * 5 + 1)
        .map(|_| device.new_event())
        .collect::<Result<_, _>>()?;
    let mut time = start.time;

    for s in 0..steps {
        time += DT;
        let e = &events[s * 5..];
        e[0].record(&stream)?;
        // SAFETY: every launch below reads and writes buffers that outlive the step; the
        // stream is synchronised at the end of the step before the host touches anything,
        // and the raw-pointer kernels index within the lengths allocated above.
        unsafe {
            grid_clear((&mut counts).partition([TILE])).async_on(&stream)?;
            grid_histogram(
                (&mut bucket).partition([TILE]),
                &px,
                &py,
                &pz,
                counts.device_pointer(),
                n_i,
                mask_i,
            )
            .async_on(&stream)?;
        }
        e[1].record(&stream)?;
        unsafe {
            grid_scan_blocks(
                (&mut offsets).partition([SCAN_BLOCK]),
                (&mut block_sums).partition([1]),
                &counts,
            )
            .grid((scan_blocks as u32, 1, 1))
            .async_on(&stream)?;
            grid_scan_totals((&mut block_sums).partition([SCAN_BLOCK])).async_on(&stream)?;
            grid_scan_add(
                (&mut offsets).partition([TILE]),
                (&mut counts).partition([TILE]),
                &block_sums,
                n_i,
                buckets as i32,
            )
            .grid(((buckets / TILE + 1) as u32, 1, 1))
            .async_on(&stream)?;
        }
        e[2].record(&stream)?;
        unsafe {
            grid_scatter(
                &bucket,
                counts.device_pointer(),
                sorted.device_pointer(),
                n_i,
            )
            .grid((birds_grid, 1, 1))
            .async_on(&stream)?;
        }
        e[3].record(&stream)?;
        unsafe {
            decide(
                px.device_pointer(),
                py.device_pointer(),
                pz.device_pointer(),
                vx.device_pointer(),
                vy.device_pointer(),
                vz.device_pointer(),
                offsets.device_pointer(),
                sorted.device_pointer(),
                sx.device_pointer(),
                sy.device_pointer(),
                sz.device_pointer(),
                radius.device_pointer(),
                next_decision.device_pointer(),
                centrality.device_pointer(),
                neighbours.device_pointer(),
                n_i,
                mask_i,
                time,
            )
            .grid((birds_grid, 1, 1))
            .async_on(&stream)?;
        }
        e[4].record(&stream)?;
        unsafe {
            integrate(
                (&mut px).partition([TILE]),
                (&mut py).partition([TILE]),
                (&mut pz).partition([TILE]),
                (&mut vx).partition([TILE]),
                (&mut vy).partition([TILE]),
                (&mut vz).partition([TILE]),
                (&mut roll).partition([TILE]),
                (&mut spin).partition([TILE]),
                &sx,
                &sy,
                &sz,
            )
            .async_on(&stream)?;
        }
        events[s * 5 + 5].record(&stream)?;
        // SAFETY: nothing else uses the stream or the buffers; this waits for the whole step.
        unsafe { stream.synchronize()? };
    }

    if check {
        let hb: Vec<i32> = api::dup(&bucket).to_host_vec().sync_on(&stream)?;
        let ho: Vec<i32> = api::dup(&offsets).to_host_vec().sync_on(&stream)?;
        let hs: Vec<i32> = api::dup(&sorted).to_host_vec().sync_on(&stream)?;
        match check_grid(&hb[..n], &ho[..=buckets], &hs[..n], buckets) {
            Ok(()) => println!("# grid check: OK ({n} birds, {buckets} buckets)"),
            Err(e) => {
                println!("# grid check: FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    let timed = steps - warmup;
    if timed > 0 {
        const NAMES: [&str; 5] = [
            "grid_histogram",
            "grid_scan",
            "grid_scatter",
            "decide",
            "integrate",
        ];
        let mut total = [0f64; 5];
        for s in warmup..steps {
            for p in 0..5 {
                total[p] += events[s * 5 + p].elapsed_time(&events[s * 5 + p + 1])? as f64;
            }
        }
        println!("# {timed} timed steps (mean GPU ms per pass, CUDA events)");
        let mut sum = 0.0;
        for (p, name) in NAMES.iter().enumerate() {
            println!("  {name:<16} {:9.4}", total[p] / timed as f64);
            sum += total[p] / timed as f64;
        }
        println!(
            "  {:<16} {sum:9.4}\n  bird-steps/s     {:9.0}",
            "step total",
            n as f64 / (sum / 1000.0)
        );
    }

    if let Some(path) = output {
        let host = |t: &Tensor<f32>| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            // `to_host_vec` consumes its tensor; read a device-side copy.
            Ok(api::dup(t).to_host_vec().sync_on(&stream)?)
        };
        let (hpx, hpy, hpz) = (host(&px)?, host(&py)?, host(&pz)?);
        let (hvx, hvy, hvz) = (host(&vx)?, host(&vy)?, host(&vz)?);
        let (hsx, hsy, hsz) = (host(&sx)?, host(&sy)?, host(&sz)?);
        let (hroll, hspin, hrad) = (host(&roll)?, host(&spin)?, host(&radius)?);
        let (hnext, hcen) = (host(&next_decision)?, host(&centrality)?);
        let hnb: Vec<i32> = api::dup(&neighbours).to_host_vec().sync_on(&stream)?;
        let birds = (0..n)
            .map(|i| Bird {
                pos: Vec3::new(hpx[i], hpy[i], hpz[i]),
                vel: Vec3::new(hvx[i], hvy[i], hvz[i]),
                roll: hroll[i],
                spin: hspin[i],
                radius: hrad[i],
                steering: Vec3::new(hsx[i], hsy[i], hsz[i]),
                next_decision: hnext[i],
                neighbours: hnb[i] as u32,
                centrality: hcen[i],
            })
            .collect();
        Flock { time, birds }.write(std::io::BufWriter::new(std::fs::File::create(path)?))?;
    }
    Ok(())
}

/// Validates the grid built in a step: `bucket[i]` is bird i's bucket, `offsets` is the
/// exclusive scan of the bucket counts (`buckets + 1` entries) and `sorted` lists the birds
/// ordered by bucket. Reports the first violation, which names the kernel at fault.
fn check_grid(
    bucket: &[i32],
    offsets: &[i32],
    sorted: &[i32],
    buckets: usize,
) -> Result<(), String> {
    let n = bucket.len();
    if bucket.iter().any(|&b| b < 0 || b as usize >= buckets) {
        return Err("a bucket index is outside 0..buckets (grid_histogram)".into());
    }
    if offsets[0] != 0 {
        return Err(format!(
            "offsets[0] = {} (grid_scan_blocks/totals)",
            offsets[0]
        ));
    }
    if offsets[buckets] as usize != n {
        return Err(format!(
            "offsets[buckets] = {} but n = {n} (grid_scan_add)",
            offsets[buckets]
        ));
    }
    if let Some(b) = (0..buckets).find(|&b| offsets[b + 1] < offsets[b]) {
        return Err(format!("offsets decreases at bucket {b} (grid_scan_*)"));
    }
    let mut count = vec![0i32; buckets];
    for &b in bucket {
        count[b as usize] += 1;
    }
    if let Some(b) = (0..buckets).find(|&b| offsets[b + 1] - offsets[b] != count[b]) {
        return Err(format!(
            "bucket {b} holds {} birds but its range is {} (grid_histogram vs scan)",
            count[b],
            offsets[b + 1] - offsets[b]
        ));
    }
    let mut seen = vec![false; n];
    for b in 0..buckets {
        for s in offsets[b]..offsets[b + 1] {
            let j = sorted[s as usize];
            if j < 0 || j as usize >= n || seen[j as usize] {
                return Err(format!(
                    "sorted[{s}] = {j} is out of range or repeated (grid_scatter)"
                ));
            }
            seen[j as usize] = true;
            if bucket[j as usize] as usize != b {
                return Err(format!(
                    "bird {j} sits in bucket {b}'s range but belongs to {} (grid_scatter)",
                    bucket[j as usize]
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_grid;

    #[test]
    fn accepts_a_correct_grid() {
        // 5 birds, 4 buckets: bucket 0 holds birds 1 and 4, bucket 2 bird 0, bucket 3 birds 2 and 3.
        let bucket = [2, 0, 3, 3, 0];
        let offsets = [0, 2, 2, 3, 5];
        let sorted = [1, 4, 0, 2, 3];
        assert!(check_grid(&bucket, &offsets, &sorted, 4).is_ok());
    }

    #[test]
    fn rejects_a_wrong_total() {
        let bucket = [2, 0, 3, 3, 0];
        let offsets = [0, 2, 2, 3, 4];
        let sorted = [1, 4, 0, 2, 3];
        let e = check_grid(&bucket, &offsets, &sorted, 4).unwrap_err();
        assert!(e.contains("grid_scan_add"), "{e}");
    }

    #[test]
    fn rejects_a_bird_in_the_wrong_range() {
        let bucket = [2, 0, 3, 3, 0];
        let offsets = [0, 2, 2, 3, 5];
        let sorted = [1, 4, 2, 0, 3]; // birds 2 and 0 swapped
        let e = check_grid(&bucket, &offsets, &sorted, 4).unwrap_err();
        assert!(e.contains("grid_scatter"), "{e}");
    }
}
