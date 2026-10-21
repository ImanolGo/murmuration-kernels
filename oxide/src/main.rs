//! The flock core step (docs/SPEC.md) as cuda-oxide SIMT kernels.
//!
//! Same passes, same names and same order of operations as `cuda/flock.cu`:
//! histogram, two-level scan, scatter, decide, integrate. Where each thread
//! owns exactly one output element the kernels use `DisjointSlice`, which the
//! compiler checks; the scatter-style grid writes need raw pointers.
//!
//! Run: `cargo oxide run -- --in start.flk --steps 100 --out end.flk`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, thread};
use cuda_host::cuda_module;

use flock_ref::params::*;
use flock_ref::{Bird, Flock};

/// A `float4`: 16-byte aligned so it loads as one 128-bit access.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct F4(pub [f32; 4]);

// SAFETY: four f32 with no padding; every bit pattern is a valid value.
unsafe impl DeviceCopy for F4 {}

const SCAN_WG: usize = 256;
const SCAN_EPT: usize = 8;
const SCAN_BLOCK: usize = SCAN_WG * SCAN_EPT;
const INV_CELL: f32 = 1.0 / GRID_CELL;

// ---- device-side helpers ----------------------------------------------------

#[derive(Clone, Copy)]
pub struct V3 {
    x: f32,
    y: f32,
    z: f32,
}

impl V3 {
    #[inline(always)]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    #[inline(always)]
    fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    #[inline(always)]
    fn from4(v: F4) -> Self {
        Self::new(v.0[0], v.0[1], v.0[2])
    }
    #[inline(always)]
    fn to4(self) -> F4 {
        F4([self.x, self.y, self.z, 0.0])
    }
    #[inline(always)]
    fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline(always)]
    fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    #[inline(always)]
    fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
    #[inline(always)]
    fn div(self, s: f32) -> Self {
        Self::new(self.x / s, self.y / s, self.z / s)
    }
}

#[inline(always)]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    libm::fminf(libm::fmaxf(v, lo), hi)
}

#[inline(always)]
fn normalize_or_zero(v: V3) -> V3 {
    let l2 = v.dot(v);
    if l2 > 0.0 {
        v.scale(1.0 / l2.sqrt())
    } else {
        V3::zero()
    }
}

/// flock-ref `forward`: unit direction with a stable fallback.
#[inline(always)]
fn forward_of(v: V3) -> V3 {
    let l2 = v.dot(v);
    if l2 > 1e-12 {
        v.scale(1.0 / l2.sqrt())
    } else {
        V3::new(1.0, 0.0, 0.0)
    }
}

/// flock-ref `side_of`: normalize(f x z) with a fallback.
#[inline(always)]
fn side_of(f: V3) -> V3 {
    let s = f.cross(V3::new(0.0, 0.0, 1.0));
    let l2 = s.dot(s);
    if l2 > 1e-8 {
        s.scale(1.0 / l2.sqrt())
    } else {
        V3::new(1.0, 0.0, 0.0)
    }
}

#[inline(always)]
fn body_up(f: V3, beta: f32) -> V3 {
    let side = side_of(f);
    let up0 = side.cross(f);
    up0.scale(libm::cosf(beta)).add(side.scale(libm::sinf(beta)))
}

#[inline(always)]
fn cell_of(p: V3) -> (i32, i32, i32) {
    (
        libm::floorf(p.x * INV_CELL) as i32,
        libm::floorf(p.y * INV_CELL) as i32,
        libm::floorf(p.z * INV_CELL) as i32,
    )
}

#[inline(always)]
fn bucket_of(c: (i32, i32, i32), mask: u32) -> u32 {
    let h = (c.0 as u32).wrapping_mul(73856093)
        ^ (c.1 as u32).wrapping_mul(19349663)
        ^ (c.2 as u32).wrapping_mul(83492791);
    h & mask
}

/// Skip cells the query sphere cannot touch (as murmur-gpu's sim.wgsl does).
#[inline(always)]
fn cell_intersects_sphere(c: (i32, i32, i32), p: V3, r2: f32) -> bool {
    let lo_x = c.0 as f32 * GRID_CELL;
    let lo_y = c.1 as f32 * GRID_CELL;
    let lo_z = c.2 as f32 * GRID_CELL;
    let dx = clampf(p.x, lo_x, lo_x + GRID_CELL) - p.x;
    let dy = clampf(p.y, lo_y, lo_y + GRID_CELL) - p.y;
    let dz = clampf(p.z, lo_z, lo_z + GRID_CELL) - p.z;
    dx * dx + dy * dy + dz * dz <= r2 + 1e-4
}

/// `radius_step` from flock-ref params.rs.
#[inline(always)]
fn radius_step(r: f32, n: u32) -> f32 {
    let nn = if n > 1 { n } else { 1 } as f32;
    let factor = libm::powf(N_C / nn, RADIUS_LERP / 3.0);
    clampf(r * factor, R_H, R_MAX)
}

// Level-1 and level-2 scans share this body; the shared array must be declared
// inside the kernel that uses it, so it is a macro and not a function.
// Leaves `$run` = this thread's exclusive prefix and `$total` = the block total.
macro_rules! block_exclusive_scan {
    ($sum:expr, $run:ident, $total:ident) => {
        static mut SH: SharedArray<u32, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        unsafe {
            SH[t] = $sum;
        }
        thread::sync_threads();
        let mut off = 1usize;
        while off < SCAN_WG {
            let x = if t >= off { unsafe { SH[t - off] } } else { 0u32 };
            thread::sync_threads();
            unsafe {
                SH[t] += x;
            }
            thread::sync_threads();
            off <<= 1;
        }
        let $total = unsafe { SH[SCAN_WG - 1] };
        let mut $run = unsafe { SH[t] } - $sum;
    };
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Zeroes the bucket histogram.
    #[kernel]
    pub fn grid_clear(mut counts: DisjointSlice<u32>) {
        if let Some(c) = counts.get_mut(thread::index_1d()) {
            *c = 0;
        }
    }

    /// Records each bird's bucket and counts one per bucket.
    #[kernel]
    pub fn grid_histogram(pos: &[F4], mut bucket: DisjointSlice<u32>, counts: *mut u32, mask: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(b) = bucket.get_mut(idx) {
            let k = bucket_of(cell_of(V3::from4(pos[i])), mask);
            *b = k;
            // SAFETY: `counts` has `mask + 1` entries and `k <= mask`.
            unsafe { DeviceAtomicU32::from_ptr(counts.add(k as usize)) }
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    /// Level 1: each block scans SCAN_BLOCK buckets and writes its total.
    #[kernel]
    pub fn grid_scan_blocks(counts: &[u32], offsets: *mut u32, block_sums: *mut u32) {
        let base = thread::blockIdx_x() as usize * SCAN_BLOCK + thread::threadIdx_x() as usize * SCAN_EPT;
        let mut c = [0u32; SCAN_EPT];
        let mut sum = 0u32;
        for k in 0..SCAN_EPT {
            c[k] = counts[base + k];
            sum += c[k];
        }
        block_exclusive_scan!(sum, run, total);
        for k in 0..SCAN_EPT {
            // SAFETY: `base + k < buckets`; buckets is a multiple of SCAN_BLOCK.
            unsafe { *offsets.add(base + k) = run };
            run += c[k];
        }
        if thread::threadIdx_x() == 0 {
            // SAFETY: one block total per block.
            unsafe { *block_sums.add(thread::blockIdx_x() as usize) = total };
        }
    }

    /// Level 2: one block scans the block totals in place (at most 2048).
    #[kernel]
    pub fn grid_scan_totals(block_sums: *mut u32, nblocks: u32) {
        let base = thread::threadIdx_x() as usize * SCAN_EPT;
        let mut c = [0u32; SCAN_EPT];
        let mut sum = 0u32;
        for k in 0..SCAN_EPT {
            // SAFETY: guarded by `nblocks`.
            c[k] = if base + k < nblocks as usize { unsafe { *block_sums.add(base + k) } } else { 0 };
            sum += c[k];
        }
        block_exclusive_scan!(sum, run, total);
        let _ = total;
        for k in 0..SCAN_EPT {
            if base + k < nblocks as usize {
                // SAFETY: guarded by `nblocks`; each thread owns its own slots.
                unsafe { *block_sums.add(base + k) = run };
            }
            run += c[k];
        }
    }

    /// Level 3: adds each block's start offset and re-seeds counts as the cursor.
    #[kernel]
    pub fn grid_scan_add(
        block_sums: &[u32],
        mut offsets: DisjointSlice<u32>,
        mut counts: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let b = idx.get();
        if let (Some(o), Some(c)) = (offsets.get_mut(idx), counts.get_mut(thread::index_1d())) {
            let v = *o + block_sums[b / SCAN_BLOCK];
            *o = v;
            *c = v;
        }
    }

    /// Writes each bird's index into its bucket's range.
    #[kernel]
    pub fn grid_scatter(bucket: &[u32], counts: *mut u32, sorted: *mut u32) {
        let i = thread::index_1d().get();
        if i < bucket.len() {
            // SAFETY: `counts` is the scatter cursor; slot < n by construction.
            let slot = unsafe { DeviceAtomicU32::from_ptr(counts.add(bucket[i] as usize)) }
                .fetch_add(1, AtomicOrdering::Relaxed);
            unsafe { *sorted.add(slot as usize) = i as u32 };
        }
    }

    /// Mirrors flock-ref `model::decide`. Reads positions and velocities (only
    /// `integrate` writes them) and updates only this bird's own decision fields.
    #[kernel]
    #[launch_bounds(256, 4)]
    pub fn decide(
        pos: &[F4],
        vel: &[F4],
        offsets: &[u32],
        sorted: &[u32],
        mut steer: DisjointSlice<F4>,
        mut radius: DisjointSlice<f32>,
        mut next_decision: DisjointSlice<f32>,
        mut centrality: DisjointSlice<f32>,
        mut neighbours: DisjointSlice<u32>,
        mask: u32,
        time: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let (Some(next), Some(rad)) = (next_decision.get_mut(idx), radius.get_mut(thread::index_1d())) else {
            return;
        };
        if time < *next {
            return;
        }
        let p = V3::from4(pos[i]);
        let v_i = V3::from4(vel[i]);
        let r = *rad;
        let f_i = forward_of(v_i);

        let mut sep = V3::zero();
        let mut coh = V3::zero();
        let mut align = V3::zero();
        let mut n_all = 0u32;
        let mut n_star = 0u32;
        let mut n_align = 0u32;

        // Neighbourhood walk (Eq. 3-6): separation, cohesion, alignment.
        {
            let lo = cell_of(V3::new(p.x - r, p.y - r, p.z - r));
            let hi = cell_of(V3::new(p.x + r, p.y + r, p.z + r));
            let r2 = r * r;
            for cz in lo.2..=hi.2 {
                for cy in lo.1..=hi.1 {
                    for cx in lo.0..=hi.0 {
                        let cell = (cx, cy, cz);
                        if !cell_intersects_sphere(cell, p, r2) {
                            continue;
                        }
                        let b = bucket_of(cell, mask) as usize;
                        let (s0, s1) = (offsets[b] as usize, offsets[b + 1] as usize);
                        for s in s0..s1 {
                            let j = sorted[s] as usize;
                            if j == i {
                                continue;
                            }
                            let q = V3::from4(pos[j]);
                            if cell_of(q) != cell {
                                continue;
                            }
                            let d = q.sub(p);
                            let d2 = d.dot(d);
                            if d2 > r2 {
                                continue;
                            }
                            let dist = d2.sqrt();
                            n_all += 1;
                            let dhat = d.div(libm::fmaxf(dist, 1e-6));
                            let g = if dist <= R_H {
                                1.0
                            } else {
                                libm::expf(-((dist - R_H) * (dist - R_H)) / (2.0 * SIGMA * SIGMA))
                            };
                            sep = sep.sub(dhat.scale(g));
                            if dist > R_H {
                                coh = coh.add(dhat);
                                n_star += 1;
                            }
                            if f_i.dot(dhat) >= BLIND_COS {
                                align = align.add(forward_of(V3::from4(vel[j])));
                                n_align += 1;
                            }
                        }
                    }
                }
            }
        }

        let mut force = V3::zero();
        if n_all > 0 {
            force = force.add(sep.scale(W_SEP / n_all as f32));
        }

        // Centrality over 2R_i (Eq. 7).
        let mut csum = V3::zero();
        let mut n_g = 0u32;
        {
            let r2 = 4.0 * r * r;
            let rr = 2.0 * r;
            let lo = cell_of(V3::new(p.x - rr, p.y - rr, p.z - rr));
            let hi = cell_of(V3::new(p.x + rr, p.y + rr, p.z + rr));
            for cz in lo.2..=hi.2 {
                for cy in lo.1..=hi.1 {
                    for cx in lo.0..=hi.0 {
                        let cell = (cx, cy, cz);
                        if !cell_intersects_sphere(cell, p, r2) {
                            continue;
                        }
                        let b = bucket_of(cell, mask) as usize;
                        let (s0, s1) = (offsets[b] as usize, offsets[b + 1] as usize);
                        for s in s0..s1 {
                            let j = sorted[s] as usize;
                            if j == i {
                                continue;
                            }
                            let q = V3::from4(pos[j]);
                            if cell_of(q) != cell {
                                continue;
                            }
                            let d = q.sub(p);
                            let d2 = d.dot(d);
                            if d2 > r2 {
                                continue;
                            }
                            csum = csum.add(d.div(libm::fmaxf(d2.sqrt(), 1e-6)));
                            n_g += 1;
                        }
                    }
                }
            }
        }
        let cen = if n_g > 0 { csum.dot(csum).sqrt() / n_g as f32 } else { 0.0 };

        if n_star > 0 {
            force = force.add(normalize_or_zero(coh).scale(W_COH * cen));
        }
        let a = if n_align > 0 { align.div(n_align as f32).sub(f_i) } else { V3::zero() };
        force = force.add(a.scale(W_ALIGN));

        // Vertical roost pull (Eq. 12), capped.
        force = force.add(V3::new(0.0, 0.0, W_ROOST_V * clampf((Z0 - p.z) / H_ROOST_V, -1.0, 1.0)));
        // Speed control (Eq. 1).
        let speed = v_i.dot(v_i).sqrt();
        force = force.add(f_i.scale((MASS / TAU) * (V0 - speed)));

        if let Some(s) = steer.get_mut(thread::index_1d()) {
            *s = force.to4();
        }
        if let Some(c) = centrality.get_mut(thread::index_1d()) {
            *c = cen;
        }
        if let Some(n) = neighbours.get_mut(thread::index_1d()) {
            *n = n_all;
        }
        *rad = radius_step(r, n_all);
        *next += DU;
    }

    /// Mirrors flock-ref `model::step_bird` (base turning model, no forced roll).
    #[kernel]
    pub fn integrate(
        steer: &[F4],
        mut pos: DisjointSlice<F4>,
        mut vel: DisjointSlice<F4>,
        mut roll: DisjointSlice<f32>,
        mut spin: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let (Some(p4), Some(v4), Some(rl), Some(sp)) = (
            pos.get_mut(idx),
            vel.get_mut(thread::index_1d()),
            roll.get_mut(thread::index_1d()),
            spin.get_mut(thread::index_1d()),
        ) else {
            return;
        };
        let mut p = V3::from4(*p4);
        let mut v = V3::from4(*v4);
        let steering = V3::from4(steer[i]);

        let fwd = forward_of(v);
        let side = side_of(fwd);
        let speed = libm::fmaxf(v.dot(v).sqrt(), 1e-6);

        let a_lat = steering.dot(side) / MASS;
        let target = clampf(libm::atanf(a_lat / G), -MAX_BANK, MAX_BANK);
        let rate = clampf((target - *rl) / TAU_ROLL, -MAX_ROLL_RATE, MAX_ROLL_RATE);
        let beta = clampf(*rl + rate * DT, -MAX_BANK, MAX_BANK);
        *sp = rate;
        *rl = beta;

        let up = body_up(fwd, beta);
        let cos_beta = libm::fmaxf(libm::cosf(beta), 0.2);
        let sr = speed / V0;
        let lift = sr * sr * L0 / cos_beta;
        let drag = sr * sr * D0;
        let flight_force = up
            .scale(lift)
            .sub(fwd.scale(drag))
            .add(fwd.scale(D0))
            .sub(V3::new(0.0, 0.0, MASS * G));

        let control = steering.sub(side.scale(steering.dot(side)));
        v = v.add(control.add(flight_force).div(MASS).scale(DT));
        let s = v.dot(v).sqrt();
        if s > V_MAX {
            v = v.scale(V_MAX / s);
        } else if s < V_MIN && s > 1e-6 {
            v = v.scale(V_MIN / s);
        }
        p = p.add(v.scale(DT));

        *p4 = p.to4();
        *v4 = v.to4();
    }
}

// ---- host -------------------------------------------------------------------

fn buckets_for(n: usize) -> u32 {
    let want = (n as u64 * 2).next_power_of_two();
    want.clamp(1 << 16, 1 << 22) as u32
}

/// Raw device address for the kernels that take pointers. Never dereferenced on
/// the host; the buffer stays alive and the stream is synchronised before reuse.
fn raw<T>(buffer: &mut DeviceBuffer<T>) -> *mut T {
    buffer.cu_deviceptr() as *mut T
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = None;
    let mut output = None;
    let (mut steps, mut warmup) = (1usize, 0usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--in" => input = args.next(),
            "--out" => output = args.next(),
            "--steps" => steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--warmup" => warmup = args.next().ok_or("--warmup needs a value")?.parse()?,
            _ => {
                eprintln!("usage: flock-oxide --in FILE [--steps K] [--warmup W] [--out FILE]");
                std::process::exit(2);
            }
        }
    }
    let input = input.ok_or("missing --in")?;
    let warmup = warmup.min(steps);

    let start = Flock::read(std::io::BufReader::new(std::fs::File::open(&input)?))?;
    let n = start.birds.len();
    let buckets = buckets_for(n);
    let scan_blocks = buckets as usize / SCAN_BLOCK;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    println!("# flock_oxide: device 0 (sm_{major}{minor}), cuda-oxide e027ac3");
    println!("# {n} birds, {steps} steps ({warmup} warm-up, not timed)");

    let f4 = |f: &dyn Fn(&Bird) -> [f32; 3]| -> Vec<F4> {
        start.birds.iter().map(|b| { let v = f(b); F4([v[0], v[1], v[2], 0.0]) }).collect()
    };
    let mut pos = DeviceBuffer::from_host(&stream, &f4(&|b| b.pos.to_array()))?;
    let mut vel = DeviceBuffer::from_host(&stream, &f4(&|b| b.vel.to_array()))?;
    let mut steer = DeviceBuffer::from_host(&stream, &f4(&|b| b.steering.to_array()))?;
    let col = |f: &dyn Fn(&Bird) -> f32| -> Vec<f32> { start.birds.iter().map(f).collect() };
    let mut roll = DeviceBuffer::from_host(&stream, &col(&|b| b.roll))?;
    let mut spin = DeviceBuffer::from_host(&stream, &col(&|b| b.spin))?;
    let mut radius = DeviceBuffer::from_host(&stream, &col(&|b| b.radius))?;
    let mut next_decision = DeviceBuffer::from_host(&stream, &col(&|b| b.next_decision))?;
    let mut centrality = DeviceBuffer::from_host(&stream, &col(&|b| b.centrality))?;
    let nb: Vec<u32> = start.birds.iter().map(|b| b.neighbours).collect();
    let mut neighbours = DeviceBuffer::from_host(&stream, &nb)?;

    let mut bucket = DeviceBuffer::<u32>::zeroed(&stream, n)?;
    let mut sorted = DeviceBuffer::<u32>::zeroed(&stream, n)?;
    let mut counts = DeviceBuffer::<u32>::zeroed(&stream, buckets as usize)?;
    let mut offsets = DeviceBuffer::<u32>::zeroed(&stream, buckets as usize + 1)?;
    let mut block_sums = DeviceBuffer::<u32>::zeroed(&stream, scan_blocks)?;

    let module = kernels::load(&ctx)?;
    let tb = 256u32;
    let cfg = |threads: usize| LaunchConfig::for_num_elems(threads as u32);
    let cfg_blocks = |blocks: usize, threads: u32| LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let _ = tb;

    const PASSES: [&str; 5] = ["grid_histogram", "grid_scan", "grid_scatter", "decide", "integrate"];
    let timing = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let mut events = Vec::with_capacity(steps * PASSES.len() + 1);
    let mut time = start.time;
    let mask = buckets - 1;

    for _ in 0..steps {
        time += DT;
        events.push(stream.record_event(timing)?);
        // SAFETY (all launches below): shapes cover every buffer access, buffers
        // outlive the stream work, and kernels that take raw pointers get the
        // allocations they index with the bounds noted at each kernel.
        unsafe {
            module.grid_clear(stream.as_ref(), cfg(buckets as usize), &mut counts)?;
            module.grid_histogram(stream.as_ref(), cfg(n), &pos, &mut bucket, raw(&mut counts), mask)?;
        }
        events.push(stream.record_event(timing)?);
        unsafe {
            module.grid_scan_blocks(stream.as_ref(), cfg_blocks(scan_blocks, SCAN_WG as u32), &counts, raw(&mut offsets), raw(&mut block_sums))?;
            module.grid_scan_totals(stream.as_ref(), cfg_blocks(1, SCAN_WG as u32), raw(&mut block_sums), scan_blocks as u32)?;
            module.grid_scan_add(stream.as_ref(), cfg(buckets as usize), &block_sums, &mut offsets, &mut counts)?;
        }
        events.push(stream.record_event(timing)?);
        unsafe {
            module.grid_scatter(stream.as_ref(), cfg(n), &bucket, raw(&mut counts), raw(&mut sorted))?;
        }
        events.push(stream.record_event(timing)?);
        unsafe {
            module.decide(
                stream.as_ref(), cfg(n), &pos, &vel, &offsets, &sorted, &mut steer, &mut radius,
                &mut next_decision, &mut centrality, &mut neighbours, mask, time,
            )?;
        }
        events.push(stream.record_event(timing)?);
        unsafe {
            module.integrate(stream.as_ref(), cfg(n), &steer, &mut pos, &mut vel, &mut roll, &mut spin)?;
        }
    }
    events.push(stream.record_event(timing)?);
    stream.synchronize()?;

    let timed = steps - warmup;
    if timed > 0 {
        let mut total = [0f64; 5];
        for s in warmup..steps {
            // Step s owns events s*5 ..= s*5+5; the last one is the next step's first (or the final) event.
            let e = &events[s * PASSES.len()..=(s + 1) * PASSES.len()];
            for p in 0..PASSES.len() {
                total[p] += e[p].elapsed_ms(&e[p + 1])? as f64;
            }
        }
        println!("# {timed} timed steps (mean GPU ms per pass, CUDA events)");
        let mut sum = 0.0;
        for (p, name) in PASSES.iter().enumerate() {
            println!("  {name:<16} {:9.4}", total[p] / timed as f64);
            sum += total[p] / timed as f64;
        }
        println!("  {:<16} {sum:9.4}\n  bird-steps/s     {:9.0}", "step total", n as f64 / (sum / 1000.0));
    }

    if let Some(path) = output {
        let (p, v, st) = (pos.to_host_vec(&stream)?, vel.to_host_vec(&stream)?, steer.to_host_vec(&stream)?);
        let (rl, sp, rd) = (roll.to_host_vec(&stream)?, spin.to_host_vec(&stream)?, radius.to_host_vec(&stream)?);
        let (nd, ce, nbv) = (next_decision.to_host_vec(&stream)?, centrality.to_host_vec(&stream)?, neighbours.to_host_vec(&stream)?);
        let v3 = |a: &F4| flock_ref_vec(a.0[0], a.0[1], a.0[2]);
        let birds = (0..n)
            .map(|i| Bird {
                pos: v3(&p[i]),
                vel: v3(&v[i]),
                roll: rl[i],
                spin: sp[i],
                radius: rd[i],
                steering: v3(&st[i]),
                next_decision: nd[i],
                neighbours: nbv[i],
                centrality: ce[i],
            })
            .collect();
        let out = Flock { time, birds };
        out.write(std::io::BufWriter::new(std::fs::File::create(path)?))?;
    }
    Ok(())
}

fn flock_ref_vec(x: f32, y: f32, z: f32) -> flock_ref::Vec3 {
    flock_ref::Vec3::new(x, y, z)
}
