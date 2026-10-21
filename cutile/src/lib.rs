//! The flock core step (docs/SPEC.md) as cuTile Rust kernels.
//!
//! Same passes and order of operations as `cuda/flock.cu`. Differences forced by
//! the tile model are listed in docs/progress.md; `bin/check.rs` compiles every
//! kernel for a target architecture without a GPU.

// The kernels are `unsafe fn` entries that call unsafe pointer helpers.
#![allow(unsafe_op_in_unsafe_fn)]

// GENERATED-TILE-BEGIN
/// Birds per tile of the `decide` kernel; written by tools/gen_decide.py (DECIDE_TILE).
pub const DECIDE_TILE: usize = 128;
// GENERATED-TILE-END

#[cutile::module]
pub mod flock_kernels {
    use cutile::core::*;

    // Birds per tile (one program handles this many) is the literal 128 below. It is not a
    // generic parameter or a const: a generic device function's return type keeps its
    // unsubstituted `B`, and a const is not evaluated inside a type, in this DSL version.

    // ---- constants (flock-ref params.rs; cuda/params.cuh repeats them) ----------
    const DT: f32 = 0.005f32;
    const V0: f32 = 10.0f32;
    const MASS: f32 = 0.08f32;
    const G: f32 = 9.81f32;
    const L0: f32 = 0.7848f32; // MASS * G
    const D0: f32 = 0.23781818f32; // L0 / CL_CD, CL_CD = 3.3
    const TAU_ROLL: f32 = 0.15f32;
    const MAX_ROLL_RATE: f32 = 6.0f32;
    const MAX_BANK: f32 = 1.3962634f32; // 80 degrees
    const V_MIN: f32 = 5.0f32;
    const V_MAX: f32 = 20.0f32;

    // The DSL infers a user function's return type only from a typed `let`, never from
    // an argument position, so every helper call below is bound to an annotated `let`.

    /// `atan` is not in the DSL (sin, cos, tan, exp, pow, floor, sqrt are). This is the
    /// Cephes `atanf` polynomial, accurate to about 1e-7; see docs/progress.md.
    /// Constants are passed in as tiles because nothing can build one in argument position.
    fn atanf(
        a: Tile<f32, { [128] }>,
        zero: Tile<f32, { [128] }>,
        one: Tile<f32, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let neg: Tile<bool, { [128] }> = lt_tile(a, zero);
        let na: Tile<f32, { [128] }> = negf(a);
        let x0: Tile<f32, { [128] }> = select(neg, na, a);
        let t3: Tile<f32, { [128] }> = broadcast_scalar(2.4142135f32, shape![128]);
        let t1: Tile<f32, { [128] }> = broadcast_scalar(0.41421357f32, shape![128]);
        let big: Tile<bool, { [128] }> = gt_tile(x0, t3);
        let mid: Tile<bool, { [128] }> = gt_tile(x0, t1);
        let r_big: Tile<f32, { [128] }> = negf(one / x0);
        let r_mid: Tile<f32, { [128] }> = (x0 - one) / (x0 + one);
        let x1m: Tile<f32, { [128] }> = select(mid, r_mid, x0);
        let x1: Tile<f32, { [128] }> = select(big, r_big, x1m);
        let pi2: Tile<f32, { [128] }> = broadcast_scalar(1.5707964f32, shape![128]);
        let pi4: Tile<f32, { [128] }> = broadcast_scalar(0.7853982f32, shape![128]);
        let y0m: Tile<f32, { [128] }> = select(mid, pi4, zero);
        let y0: Tile<f32, { [128] }> = select(big, pi2, y0m);
        let z: Tile<f32, { [128] }> = x1 * x1;
        let c3: Tile<f32, { [128] }> = broadcast_scalar(8.05374449538e-2f32, shape![128]);
        let c2: Tile<f32, { [128] }> = broadcast_scalar(1.38776856032e-1f32, shape![128]);
        let c1: Tile<f32, { [128] }> = broadcast_scalar(1.99777106478e-1f32, shape![128]);
        let c0: Tile<f32, { [128] }> = broadcast_scalar(3.33329491539e-1f32, shape![128]);
        let poly: Tile<f32, { [128] }> = ((c3 * z - c2) * z + c1) * z - c0;
        let y: Tile<f32, { [128] }> = y0 + poly * z * x1 + x1;
        let ny: Tile<f32, { [128] }> = negf(y);
        select(neg, ny, y)
    }

    /// `base + offsets` as a tile of 128 pointers (scalar pointer -> rank 1 -> 128 wide).
    fn offset_ptrs(
        base: PointerTile<*mut i32, { [] }>,
        offsets: Tile<i32, { [128] }>,
    ) -> PointerTile<*mut i32, { [128] }> {
        let p1: PointerTile<*mut i32, { [1] }> = reshape_ptr(base, shape![1]);
        let pb: PointerTile<*mut i32, { [128] }> = broadcast_ptr(p1, shape![128]);
        let out: PointerTile<*mut i32, { [128] }> = pb.offset_tile(offsets);
        out
    }

    // ---- grid (the passes of murmur-gpu's grid.wgsl) --------------------------------
    // Bucket arrays are padded to multiples of the tile; `n` and `buckets` arrive as
    // scalars. The scan works on 2048-wide tiles (SCAN_BLOCK), one program per block.

    /// Zeroes the bucket histogram.
    #[cutile::entry()]
    pub fn grid_clear(counts: &mut Tensor<i32, { [128] }>) {
        let z: Tile<i32, { [128] }> = constant(0i32, shape![128]);
        counts.store(z);
    }

    /// Records each bird's bucket and counts one per bucket.
    #[cutile::entry()]
    pub unsafe fn grid_histogram(
        bucket: &mut Tensor<i32, { [128] }>,
        px: &Tensor<f32, { [-1] }>,
        py: &Tensor<f32, { [-1] }>,
        pz: &Tensor<f32, { [-1] }>,
        counts: *mut i32,
        n: i32,
        mask: i32,
    ) {
        let pid = program_id(0);
        let s: Shape<{ [128] }> = shape![128];
        let x: Tile<f32, { [128] }> = px.load_tile(s, [pid]);
        let y: Tile<f32, { [128] }> = py.load_tile(s, [pid]);
        let z: Tile<f32, { [128] }> = pz.load_tile(s, [pid]);
        let inv: Tile<f32, { [128] }> = constant(0.25f32, shape![128]); // 1 / GRID_CELL
        let fx: Tile<f32, { [128] }> = floor(x * inv);
        let fy: Tile<f32, { [128] }> = floor(y * inv);
        let fz: Tile<f32, { [128] }> = floor(z * inv);
        let cx: Tile<i32, { [128] }> = convert_tile(fx);
        let cy: Tile<i32, { [128] }> = convert_tile(fy);
        let cz: Tile<i32, { [128] }> = convert_tile(fz);
        let m1: Tile<i32, { [128] }> = constant(73856093i32, shape![128]);
        let m2: Tile<i32, { [128] }> = constant(19349663i32, shape![128]);
        let m3: Tile<i32, { [128] }> = constant(83492791i32, shape![128]);
        let h: Tile<i32, { [128] }> = xori(xori(cx * m1, cy * m2), cz * m3);
        let mk: Tile<i32, { [128] }> = broadcast_scalar(mask, shape![128]);
        let b: Tile<i32, { [128] }> = andi(h, mk);
        bucket.store(b);

        let idx0: Tile<i32, { [128] }> = iota(shape![128]);
        let base: Tile<i32, { [128] }> = broadcast_scalar(pid * 128i32, shape![128]);
        let idx: Tile<i32, { [128] }> = idx0 + base;
        let nn: Tile<i32, { [128] }> = broadcast_scalar(n, shape![128]);
        let valid: Tile<bool, { [128] }> = lt_tile(idx, nn);
        let root: PointerTile<*mut i32, { [] }> = pointer_to_tile(counts);
        let ptrs: PointerTile<*mut i32, { [128] }> = offset_ptrs(root, b);
        let one: Tile<i32, { [128] }> = constant(1i32, shape![128]);
        let (_old, _t): (Tile<i32, { [128] }>, Token) = unsafe {
            atomic_rmw_tko(
                ptrs,
                one,
                atomic::Add,
                ordering::Relaxed,
                scope::Device,
                Some(valid),
                None,
            )
        };
    }

    /// Level 1: each program scans a 2048-bucket block (exclusive) and writes its total.
    /// `offsets` and `block_totals` must hold the same number of tiles (the launch grid is
    /// inferred from the partitions): `buckets / 2048 + 1` here, the last one over padding.
    #[cutile::entry()]
    pub fn grid_scan_blocks(
        offsets: &mut Tensor<i32, { [2048] }>,
        block_totals: &mut Tensor<i32, { [1] }>,
        counts: &Tensor<i32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let c: Tile<i32, { [2048] }> = counts.load_tile(shape![2048], [pid]);
        let incl: Tile<i32, { [2048] }> = scan_sum(c, 0i32, reverse::Forward, 0i32);
        offsets.store(incl - c);
        let total: Tile<i32, { [] }> = reduce_sum(c, 0i32);
        let t1: Tile<i32, { [1] }> = total.reshape(shape![1]);
        block_totals.store(t1);
    }

    /// Level 2: one program turns the block totals (zero-padded to 2048) into block starts.
    #[cutile::entry()]
    pub fn grid_scan_totals(
        block_starts: &mut Tensor<i32, { [2048] }>,
        block_totals: &Tensor<i32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let c: Tile<i32, { [2048] }> = block_totals.load_tile(shape![2048], [pid]);
        let incl: Tile<i32, { [2048] }> = scan_sum(c, 0i32, reverse::Forward, 0i32);
        block_starts.store(incl - c);
    }

    /// Level 3: add each block's start offset, re-seed counts as the scatter cursor, and
    /// write offsets[buckets] = n (it lands in the padding tile past the last bucket).
    #[cutile::entry()]
    pub fn grid_scan_add(
        offsets: &mut Tensor<i32, { [128] }>,
        counts: &mut Tensor<i32, { [128] }>,
        block_starts: &Tensor<i32, { [-1] }>,
        n: i32,
        buckets: i32,
    ) {
        let pid = program_id(0);
        let o: Tile<i32, { [128] }> = offsets.load();
        let blk: Tile<i32, { [1] }> = block_starts.load_tile(shape![1], [pid / 16i32]);
        let bs: Tile<i32, { [128] }> = blk.broadcast(shape![128]);
        let v0: Tile<i32, { [128] }> = o + bs;
        let idx0: Tile<i32, { [128] }> = iota(shape![128]);
        let base: Tile<i32, { [128] }> = broadcast_scalar(pid * 128i32, shape![128]);
        let idx: Tile<i32, { [128] }> = idx0 + base;
        let nb: Tile<i32, { [128] }> = broadcast_scalar(buckets, shape![128]);
        let nn: Tile<i32, { [128] }> = broadcast_scalar(n, shape![128]);
        let last: Tile<bool, { [128] }> = eq_tile(idx, nb);
        let v: Tile<i32, { [128] }> = select(last, nn, v0);
        offsets.store(v);
        counts.store(v);
    }

    /// Writes each bird's index into its bucket's range.
    #[cutile::entry()]
    pub unsafe fn grid_scatter(
        bucket: &Tensor<i32, { [-1] }>,
        counts: *mut i32,
        sorted: *mut i32,
        n: i32,
    ) {
        let pid = program_id(0);
        let b: Tile<i32, { [128] }> = bucket.load_tile(shape![128], [pid]);
        let idx0: Tile<i32, { [128] }> = iota(shape![128]);
        let base: Tile<i32, { [128] }> = broadcast_scalar(pid * 128i32, shape![128]);
        let idx: Tile<i32, { [128] }> = idx0 + base;
        let nn: Tile<i32, { [128] }> = broadcast_scalar(n, shape![128]);
        let valid: Tile<bool, { [128] }> = lt_tile(idx, nn);
        let croot: PointerTile<*mut i32, { [] }> = pointer_to_tile(counts);
        let cptrs: PointerTile<*mut i32, { [128] }> = offset_ptrs(croot, b);
        let one: Tile<i32, { [128] }> = constant(1i32, shape![128]);
        let (slot, _t): (Tile<i32, { [128] }>, Token) = unsafe {
            atomic_rmw_tko(
                cptrs,
                one,
                atomic::Add,
                ordering::Relaxed,
                scope::Device,
                Some(valid),
                None,
            )
        };
        let sroot: PointerTile<*mut i32, { [] }> = pointer_to_tile(sorted);
        let sptrs: PointerTile<*mut i32, { [128] }> = offset_ptrs(sroot, slot);
        unsafe {
            store_ptr_tko(
                sptrs,
                idx,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                None,
                Latency::<0>,
            );
        }
    }

    /// Mirrors flock-ref `model::step_bird` (base turning model, no forced roll),
    /// one tile of B birds per program. Arrays are padded to a multiple of B.
    #[cutile::entry()]
    pub fn integrate(
        px: &mut Tensor<f32, { [128] }>,
        py: &mut Tensor<f32, { [128] }>,
        pz: &mut Tensor<f32, { [128] }>,
        vx: &mut Tensor<f32, { [128] }>,
        vy: &mut Tensor<f32, { [128] }>,
        vz: &mut Tensor<f32, { [128] }>,
        roll: &mut Tensor<f32, { [128] }>,
        spin: &mut Tensor<f32, { [128] }>,
        sx: &Tensor<f32, { [-1] }>,
        sy: &Tensor<f32, { [-1] }>,
        sz: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let s: Shape<{ [128] }> = shape![128];
        let zero: Tile<f32, { [128] }> = constant(0.0f32, shape![128]);
        let one: Tile<f32, { [128] }> = constant(1.0f32, shape![128]);
        let eps: Tile<f32, { [128] }> = constant(1e-6f32, shape![128]);
        let eps2: Tile<f32, { [128] }> = constant(1e-12f32, shape![128]);
        let eps8: Tile<f32, { [128] }> = constant(1e-8f32, shape![128]);

        let x: Tile<f32, { [128] }> = px.load();
        let y: Tile<f32, { [128] }> = py.load();
        let z: Tile<f32, { [128] }> = pz.load();
        let mut ux: Tile<f32, { [128] }> = vx.load();
        let mut uy: Tile<f32, { [128] }> = vy.load();
        let mut uz: Tile<f32, { [128] }> = vz.load();
        let rl: Tile<f32, { [128] }> = roll.load();
        let fx0: Tile<f32, { [128] }> = sx.load_tile(s, [pid]);
        let fy0: Tile<f32, { [128] }> = sy.load_tile(s, [pid]);
        let fz0: Tile<f32, { [128] }> = sz.load_tile(s, [pid]);

        // forward_of(v): unit velocity, +x when the speed is ~0.
        let v2: Tile<f32, { [128] }> = ux * ux + uy * uy + uz * uz;
        let ok: Tile<bool, { [128] }> = gt_tile(v2, eps2);
        let v2c: Tile<f32, { [128] }> = max_tile(v2, eps2);
        let rt: Tile<f32, { [128] }> = sqrt(v2c, rounding::NearestEven, ftz::Disabled);
        let inv: Tile<f32, { [128] }> = one / rt;
        let ax: Tile<f32, { [128] }> = ux * inv;
        let ay: Tile<f32, { [128] }> = uy * inv;
        let az: Tile<f32, { [128] }> = uz * inv;
        let fx: Tile<f32, { [128] }> = select(ok, ax, one);
        let fy: Tile<f32, { [128] }> = select(ok, ay, zero);
        let fz: Tile<f32, { [128] }> = select(ok, az, zero);

        // side_of(f) = normalize(f x z) = (fy, -fx, 0), +x when f is near vertical.
        let cx: Tile<f32, { [128] }> = fy;
        let cy: Tile<f32, { [128] }> = negf(fx);
        let c2: Tile<f32, { [128] }> = cx * cx + cy * cy;
        let cok: Tile<bool, { [128] }> = gt_tile(c2, eps8);
        let c2c: Tile<f32, { [128] }> = max_tile(c2, eps8);
        let crt: Tile<f32, { [128] }> = sqrt(c2c, rounding::NearestEven, ftz::Disabled);
        let cinv: Tile<f32, { [128] }> = one / crt;
        let nx: Tile<f32, { [128] }> = cx * cinv;
        let ny: Tile<f32, { [128] }> = cy * cinv;
        let sdx: Tile<f32, { [128] }> = select(cok, nx, one);
        let sdy: Tile<f32, { [128] }> = select(cok, ny, zero);

        let speed0: Tile<f32, { [128] }> = sqrt(v2, rounding::NearestEven, ftz::Disabled);
        let speed: Tile<f32, { [128] }> = max_tile(speed0, eps);

        let mass: Tile<f32, { [128] }> = constant(0.08f32, shape![128]);
        let grav: Tile<f32, { [128] }> = constant(9.81f32, shape![128]);
        let slat: Tile<f32, { [128] }> = fx0 * sdx + fy0 * sdy; // side has no z component
        let a_lat: Tile<f32, { [128] }> = slat / mass;
        let ratio: Tile<f32, { [128] }> = a_lat / grav;
        let at = atanf(ratio, zero, one);
        let max_bank: Tile<f32, { [128] }> = constant(1.3962634f32, shape![128]);
        let min_bank: Tile<f32, { [128] }> = constant(-1.3962634f32, shape![128]);
        let tgt: Tile<f32, { [128] }> = min_tile(max_tile(at, min_bank), max_bank);
        let tau_roll: Tile<f32, { [128] }> = constant(0.15f32, shape![128]);
        let max_rate: Tile<f32, { [128] }> = constant(6.0f32, shape![128]);
        let min_rate: Tile<f32, { [128] }> = constant(-6.0f32, shape![128]);
        let rate_raw: Tile<f32, { [128] }> = (tgt - rl) / tau_roll;
        let rate: Tile<f32, { [128] }> = min_tile(max_tile(rate_raw, min_rate), max_rate);
        let dt: Tile<f32, { [128] }> = constant(0.005f32, shape![128]);
        let beta_raw: Tile<f32, { [128] }> = rl + rate * dt;
        let beta: Tile<f32, { [128] }> = min_tile(max_tile(beta_raw, min_bank), max_bank);
        spin.store(rate);
        roll.store(beta);

        // body_up(f, beta) = (side x f) cos(beta) + side sin(beta), side = (sdx, sdy, 0);
        // side x f = (sdy fz, -sdx fz, sdx fy - sdy fx).
        let u0x: Tile<f32, { [128] }> = sdy * fz;
        let u0y: Tile<f32, { [128] }> = negf(sdx * fz);
        let u0z: Tile<f32, { [128] }> = sdx * fy - sdy * fx;
        let cb: Tile<f32, { [128] }> = cos(beta);
        let sb: Tile<f32, { [128] }> = sin(beta);
        let upx: Tile<f32, { [128] }> = u0x * cb + sdx * sb;
        let upy: Tile<f32, { [128] }> = u0y * cb + sdy * sb;
        let upz: Tile<f32, { [128] }> = u0z * cb;

        let cos_floor: Tile<f32, { [128] }> = constant(0.2f32, shape![128]);
        let cos_beta: Tile<f32, { [128] }> = max_tile(cb, cos_floor);
        let v0: Tile<f32, { [128] }> = constant(10.0f32, shape![128]);
        let l0: Tile<f32, { [128] }> = constant(0.7848f32, shape![128]);
        let d0: Tile<f32, { [128] }> = constant(0.23781818f32, shape![128]);
        let sr: Tile<f32, { [128] }> = speed / v0;
        let lift: Tile<f32, { [128] }> = sr * sr * l0 / cos_beta;
        let drag: Tile<f32, { [128] }> = sr * sr * d0;
        let weight: Tile<f32, { [128] }> = constant(0.7848f32, shape![128]); // MASS * G
        let gx: Tile<f32, { [128] }> = upx * lift - fx * drag + fx * d0;
        let gy: Tile<f32, { [128] }> = upy * lift - fy * drag + fy * d0;
        let gz: Tile<f32, { [128] }> = upz * lift - fz * drag + fz * d0 - weight;

        let sdot: Tile<f32, { [128] }> = fx0 * sdx + fy0 * sdy;
        let kx: Tile<f32, { [128] }> = fx0 - sdx * sdot;
        let ky: Tile<f32, { [128] }> = fy0 - sdy * sdot;
        let kz: Tile<f32, { [128] }> = fz0;
        let dt_m: Tile<f32, { [128] }> = constant(0.0625f32, shape![128]); // DT / MASS
        ux = ux + (kx + gx) * dt_m;
        uy = uy + (ky + gy) * dt_m;
        uz = uz + (kz + gz) * dt_m;

        let sp: Tile<f32, { [128] }> = sqrt(
            ux * ux + uy * uy + uz * uz,
            rounding::NearestEven,
            ftz::Disabled,
        );
        let vmax: Tile<f32, { [128] }> = constant(20.0f32, shape![128]);
        let vmin: Tile<f32, { [128] }> = constant(5.0f32, shape![128]);
        let too_fast: Tile<bool, { [128] }> = gt_tile(sp, vmax);
        let too_slow: Tile<bool, { [128] }> = lt_tile(sp, vmin);
        let can: Tile<bool, { [128] }> = gt_tile(sp, eps);
        let spc: Tile<f32, { [128] }> = max_tile(sp, eps);
        let up_scale: Tile<f32, { [128] }> = vmax / spc;
        let dn_scale: Tile<f32, { [128] }> = vmin / spc;
        let slow_scale: Tile<f32, { [128] }> = select(can, dn_scale, one);
        let mid_scale: Tile<f32, { [128] }> = select(too_slow, slow_scale, one);
        let scale: Tile<f32, { [128] }> = select(too_fast, up_scale, mid_scale);
        ux = ux * scale;
        uy = uy * scale;
        uz = uz * scale;

        vx.store(ux);
        vy.store(uy);
        vz.store(uz);
        px.store(x + ux * dt);
        py.store(y + uy * dt);
        pz.store(z + uz * dt);
    }

    // GENERATED-BEGIN (tools/gen_decide.py)

    fn ptrs_f32(base: *mut f32, off: Tile<i32, { [128] }>) -> PointerTile<*mut f32, { [128] }> {
        let r0: PointerTile<*mut f32, { [] }> = pointer_to_tile(base);
        let p1: PointerTile<*mut f32, { [1] }> = reshape_ptr(r0, shape![1]);
        let pb: PointerTile<*mut f32, { [128] }> = broadcast_ptr(p1, shape![128]);
        let out: PointerTile<*mut f32, { [128] }> = pb.offset_tile(off);
        out
    }

    fn ptrs_i32(base: *mut i32, off: Tile<i32, { [128] }>) -> PointerTile<*mut i32, { [128] }> {
        let r0: PointerTile<*mut i32, { [] }> = pointer_to_tile(base);
        let p1: PointerTile<*mut i32, { [1] }> = reshape_ptr(r0, shape![1]);
        let pb: PointerTile<*mut i32, { [128] }> = broadcast_ptr(p1, shape![128]);
        let out: PointerTile<*mut i32, { [128] }> = pb.offset_tile(off);
        out
    }

    unsafe fn gather_f32(base: *mut f32, idx: Tile<i32, { [128] }>, live: Tile<bool, { [128] }>) -> Tile<f32, { [128] }> {
        let ptrs: PointerTile<*mut f32, { [128] }> = ptrs_f32(base, idx);
        let (v, _t): (Tile<f32, { [128] }>, Token) = unsafe {
            load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, Some(live), Some(0.0f32), None, Latency::<0>)
        };
        v
    }

    unsafe fn gather_i32(base: *mut i32, idx: Tile<i32, { [128] }>, live: Tile<bool, { [128] }>) -> Tile<i32, { [128] }> {
        let ptrs: PointerTile<*mut i32, { [128] }> = ptrs_i32(base, idx);
        let (v, _t): (Tile<i32, { [128] }>, Token) = unsafe {
            load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, Some(live), Some(0i32), None, Latency::<0>)
        };
        v
    }

    unsafe fn scatter_f32(base: *mut f32, idx: Tile<i32, { [128] }>, value: Tile<f32, { [128] }>, live: Tile<bool, { [128] }>) {
        let ptrs: PointerTile<*mut f32, { [128] }> = ptrs_f32(base, idx);
        let _t: Token = unsafe {
            store_ptr_tko(ptrs, value, ordering::Weak, None::<scope::TileBlock>, Some(live), None, Latency::<0>)
        };
    }

    unsafe fn scatter_i32(base: *mut i32, idx: Tile<i32, { [128] }>, value: Tile<i32, { [128] }>, live: Tile<bool, { [128] }>) {
        let ptrs: PointerTile<*mut i32, { [128] }> = ptrs_i32(base, idx);
        let _t: Token = unsafe {
            store_ptr_tko(ptrs, value, ordering::Weak, None::<scope::TileBlock>, Some(live), None, Latency::<0>)
        };
    }

    /// `a && b` on bool tiles. The "false" tile comes from a comparison: `constant(false,
    /// shape)` compiles to a tile of TRUE in this DSL version (`constant <i1: 1>` in the Tile IR),
    /// which silently turned this into `a -> b`.
    fn band(a: Tile<bool, { [128] }>, b: Tile<bool, { [128] }>) -> Tile<bool, { [128] }> {
        let z: Tile<i32, { [128] }> = constant(0i32, shape![128]);
        let no: Tile<bool, { [128] }> = lt_tile(z, z);
        select(a, b, no)
    }

    /// Mirrors flock-ref `model::decide` for a tile of 128 birds. A bird that is not due
    /// (or is padding) takes part in no loop and stores nothing. Both neighbourhood walks
    /// iterate to the largest cell span and bucket size in the tile, masked per lane.
    #[cutile::entry()]
    pub unsafe fn decide(
        px: *mut f32, py: *mut f32, pz: *mut f32,
        vx: *mut f32, vy: *mut f32, vz: *mut f32,
        offsets: *mut i32, sorted: *mut i32,
        sx: *mut f32, sy: *mut f32, sz: *mut f32,
        radius: *mut f32, next_decision: *mut f32, centrality: *mut f32, neighbours: *mut i32,
        n: i32, mask: i32, time: f32,
    ) {
        let pid = program_id(0);
        let i0: Tile<i32, { [128] }> = iota(shape![128]);
        let base: Tile<i32, { [128] }> = broadcast_scalar(pid * 128i32, shape![128]);
        let idx: Tile<i32, { [128] }> = i0 + base;
        let nn: Tile<i32, { [128] }> = broadcast_scalar(n, shape![128]);
        let valid: Tile<bool, { [128] }> = lt_tile(idx, nn);
        let tm: Tile<f32, { [128] }> = broadcast_scalar(time, shape![128]);

        let zero: Tile<f32, { [128] }> = constant(0.0f32, shape![128]);
        let one: Tile<f32, { [128] }> = constant(1.0f32, shape![128]);
        let four: Tile<f32, { [128] }> = constant(4.0f32, shape![128]);
        let eps: Tile<f32, { [128] }> = constant(1e-6f32, shape![128]);
        let tiny: Tile<f32, { [128] }> = constant(1e-4f32, shape![128]);
        let tiny2: Tile<f32, { [128] }> = constant(1e-12f32, shape![128]);
        let tiny8: Tile<f32, { [128] }> = constant(1e-8f32, shape![128]);
        let inv_cell: Tile<f32, { [128] }> = constant(0.25f32, shape![128]);
        let rh: Tile<f32, { [128] }> = constant(0.2f32, shape![128]);
        let two_sigma2: Tile<f32, { [128] }> = constant(3.7538f32, shape![128]); // 2 * SIGMA^2
        let blind: Tile<f32, { [128] }> = constant(-0.70710678f32, shape![128]);
        let zero_i: Tile<i32, { [128] }> = constant(0i32, shape![128]);
        let one_i: Tile<i32, { [128] }> = constant(1i32, shape![128]);
        let m1: Tile<i32, { [128] }> = constant(73856093i32, shape![128]);
        let m2: Tile<i32, { [128] }> = constant(19349663i32, shape![128]);
        let m3: Tile<i32, { [128] }> = constant(83492791i32, shape![128]);
        let mk: Tile<i32, { [128] }> = broadcast_scalar(mask, shape![128]);

        let nd: Tile<f32, { [128] }> = gather_f32(next_decision, idx, valid);
        let due: Tile<bool, { [128] }> = band(valid, ge_tile(tm, nd));
        let x: Tile<f32, { [128] }> = gather_f32(px, idx, valid);
        let y: Tile<f32, { [128] }> = gather_f32(py, idx, valid);
        let z: Tile<f32, { [128] }> = gather_f32(pz, idx, valid);
        let ux: Tile<f32, { [128] }> = gather_f32(vx, idx, valid);
        let uy: Tile<f32, { [128] }> = gather_f32(vy, idx, valid);
        let uz: Tile<f32, { [128] }> = gather_f32(vz, idx, valid);
        let r: Tile<f32, { [128] }> = gather_f32(radius, idx, valid);
        let r2: Tile<f32, { [128] }> = r * r;
        let rr: Tile<f32, { [128] }> = r + r;
        let rr2: Tile<f32, { [128] }> = rr * rr;

        // forward_of(v)
        let v2: Tile<f32, { [128] }> = ux * ux + uy * uy + uz * uz;
        let vok: Tile<bool, { [128] }> = gt_tile(v2, tiny2);
        let vrt: Tile<f32, { [128] }> = sqrt(max_tile(v2, tiny2), rounding::NearestEven, ftz::Disabled);
        let vinv: Tile<f32, { [128] }> = one / vrt;
        let fx: Tile<f32, { [128] }> = select(vok, ux * vinv, one);
        let fy: Tile<f32, { [128] }> = select(vok, uy * vinv, zero);
        let fz: Tile<f32, { [128] }> = select(vok, uz * vinv, zero);

        // ---- accumulators, shared by the first walk ----
        let mut sep_x: Tile<f32, { [128] }> = zero;
        let mut sep_y: Tile<f32, { [128] }> = zero;
        let mut sep_z: Tile<f32, { [128] }> = zero;
        let mut coh_x: Tile<f32, { [128] }> = zero;
        let mut coh_y: Tile<f32, { [128] }> = zero;
        let mut coh_z: Tile<f32, { [128] }> = zero;
        let mut al_x: Tile<f32, { [128] }> = zero;
        let mut al_y: Tile<f32, { [128] }> = zero;
        let mut al_z: Tile<f32, { [128] }> = zero;
        let mut n_all: Tile<i32, { [128] }> = zero_i;
        let mut n_star: Tile<i32, { [128] }> = zero_i;
        let mut n_align: Tile<i32, { [128] }> = zero_i;

        // ---- walk over the box of radius r (cells lo..hi, then the birds in each) ----
        let box_lo_x: Tile<i32, { [128] }> = convert_tile(floor((x - r) * inv_cell));
        let box_lo_y: Tile<i32, { [128] }> = convert_tile(floor((y - r) * inv_cell));
        let box_lo_z: Tile<i32, { [128] }> = convert_tile(floor((z - r) * inv_cell));
        let box_hi_x: Tile<i32, { [128] }> = convert_tile(floor((x + r) * inv_cell));
        let box_hi_y: Tile<i32, { [128] }> = convert_tile(floor((y + r) * inv_cell));
        let box_hi_z: Tile<i32, { [128] }> = convert_tile(floor((z + r) * inv_cell));
        let span_x: Tile<i32, { [128] }> = select(due, box_hi_x - box_lo_x, zero_i);
        let span_y: Tile<i32, { [128] }> = select(due, box_hi_y - box_lo_y, zero_i);
        let span_z: Tile<i32, { [128] }> = select(due, box_hi_z - box_lo_z, zero_i);
        let max_x: Tile<i32, { [] }> = reduce_max(span_x, 0i32);
        let max_y: Tile<i32, { [] }> = reduce_max(span_y, 0i32);
        let max_z: Tile<i32, { [] }> = reduce_max(span_z, 0i32);
        let nx0: i32 = tile_to_scalar(max_x);
        let nx: i32 = nx0 + 1i32;
        let ny0: i32 = tile_to_scalar(max_y);
        let ny: i32 = ny0 + 1i32;
        let nz0: i32 = tile_to_scalar(max_z);
        let nz: i32 = nz0 + 1i32;
        for oz in 0i32..nz {
            for oy in 0i32..ny {
                for ox in 0i32..nx {
                    let cell_x: Tile<i32, { [128] }> = box_lo_x + broadcast_scalar(ox, shape![128]);
                    let cell_y: Tile<i32, { [128] }> = box_lo_y + broadcast_scalar(oy, shape![128]);
                    let cell_z: Tile<i32, { [128] }> = box_lo_z + broadcast_scalar(oz, shape![128]);
                    let in_x: Tile<bool, { [128] }> = le_tile(cell_x, box_hi_x);
                    let in_y: Tile<bool, { [128] }> = le_tile(cell_y, box_hi_y);
                    let in_z: Tile<bool, { [128] }> = le_tile(cell_z, box_hi_z);
                    let in_xy: Tile<bool, { [128] }> = band(in_x, in_y);
                    let in_xyz: Tile<bool, { [128] }> = band(in_xy, in_z);
                    let in_box: Tile<bool, { [128] }> = band(in_xyz, due);
                    // cell_intersects_sphere
                    let lo_fx: Tile<f32, { [128] }> = convert_tile(cell_x) * four;
                    let lo_fy: Tile<f32, { [128] }> = convert_tile(cell_y) * four;
                    let lo_fz: Tile<f32, { [128] }> = convert_tile(cell_z) * four;
                    let nxd: Tile<f32, { [128] }> = min_tile(max_tile(x, lo_fx), lo_fx + four) - x;
                    let nyd: Tile<f32, { [128] }> = min_tile(max_tile(y, lo_fy), lo_fy + four) - y;
                    let nzd: Tile<f32, { [128] }> = min_tile(max_tile(z, lo_fz), lo_fz + four) - z;
                    let near: Tile<bool, { [128] }> = le_tile(nxd * nxd + nyd * nyd + nzd * nzd, r2 + tiny);
                    let act: Tile<bool, { [128] }> = band(in_box, near);
                    // bucket_of(cell) and its range in the sorted array
                    let hh: Tile<i32, { [128] }> = xori(xori(cell_x * m1, cell_y * m2), cell_z * m3);
                    let bk: Tile<i32, { [128] }> = andi(hh, mk);
                    let st: Tile<i32, { [128] }> = gather_i32(offsets, bk, act);
                    let en: Tile<i32, { [128] }> = gather_i32(offsets, bk + one_i, act);
                    let cnt: Tile<i32, { [128] }> = en - st;
                    let max_cnt: Tile<i32, { [] }> = reduce_max(cnt, 0i32);
                    let mc: i32 = tile_to_scalar(max_cnt);
                    for k in 0i32..mc {
                        let kk: Tile<i32, { [128] }> = broadcast_scalar(k, shape![128]);
                        let live: Tile<bool, { [128] }> = lt_tile(kk, cnt);
                        let j: Tile<i32, { [128] }> = gather_i32(sorted, st + kk, live);
                        let qx: Tile<f32, { [128] }> = gather_f32(px, j, live);
                        let qy: Tile<f32, { [128] }> = gather_f32(py, j, live);
                        let qz: Tile<f32, { [128] }> = gather_f32(pz, j, live);
                        // the exact cell check: only birds really in this cell, not just this bucket
                        let qcx: Tile<i32, { [128] }> = convert_tile(floor(qx * inv_cell));
                        let qcy: Tile<i32, { [128] }> = convert_tile(floor(qy * inv_cell));
                        let qcz: Tile<i32, { [128] }> = convert_tile(floor(qz * inv_cell));
                        let same_x: Tile<bool, { [128] }> = eq_tile(qcx, cell_x);
                        let same_y: Tile<bool, { [128] }> = eq_tile(qcy, cell_y);
                        let same_z: Tile<bool, { [128] }> = eq_tile(qcz, cell_z);
                        let same_xy: Tile<bool, { [128] }> = band(same_x, same_y);
                        let same: Tile<bool, { [128] }> = band(same_xy, same_z);
                        let not_self: Tile<bool, { [128] }> = ne_tile(j, idx);
                        let dx: Tile<f32, { [128] }> = qx - x;
                        let dy: Tile<f32, { [128] }> = qy - y;
                        let dz: Tile<f32, { [128] }> = qz - z;
                        let d2: Tile<f32, { [128] }> = dx * dx + dy * dy + dz * dz;
                        let within: Tile<bool, { [128] }> = le_tile(d2, r2);
                        let m0: Tile<bool, { [128] }> = band(live, same);
                        let m1b: Tile<bool, { [128] }> = band(m0, not_self);
                        let m: Tile<bool, { [128] }> = band(m1b, within);
                        let dist: Tile<f32, { [128] }> = sqrt(d2, rounding::NearestEven, ftz::Disabled);
                        let distc: Tile<f32, { [128] }> = max_tile(dist, eps);
                        let dhx: Tile<f32, { [128] }> = dx / distc;
                        let dhy: Tile<f32, { [128] }> = dy / distc;
                        let dhz: Tile<f32, { [128] }> = dz / distc;
                        let dm: Tile<f32, { [128] }> = dist - rh;
                        let gauss: Tile<f32, { [128] }> = exp(negf(dm * dm) / two_sigma2);
                        let g: Tile<f32, { [128] }> = select(le_tile(dist, rh), one, gauss);
                        let mg: Tile<f32, { [128] }> = select(m, g, zero);
                        sep_x = sep_x - dhx * mg;
                        sep_y = sep_y - dhy * mg;
                        sep_z = sep_z - dhz * mg;
                        n_all = n_all + select(m, one_i, zero_i);
                        let far: Tile<bool, { [128] }> = band(m, gt_tile(dist, rh));
                        coh_x = coh_x + select(far, dhx, zero);
                        coh_y = coh_y + select(far, dhy, zero);
                        coh_z = coh_z + select(far, dhz, zero);
                        n_star = n_star + select(far, one_i, zero_i);
                        let dotf: Tile<f32, { [128] }> = fx * dhx + fy * dhy + fz * dhz;
                        let aligned: Tile<bool, { [128] }> = band(m, ge_tile(dotf, blind));
                        let wx: Tile<f32, { [128] }> = gather_f32(vx, j, live);
                        let wy: Tile<f32, { [128] }> = gather_f32(vy, j, live);
                        let wz: Tile<f32, { [128] }> = gather_f32(vz, j, live);
                        let w2: Tile<f32, { [128] }> = wx * wx + wy * wy + wz * wz;
                        let wok: Tile<bool, { [128] }> = gt_tile(w2, tiny2);
                        let wrt: Tile<f32, { [128] }> = sqrt(max_tile(w2, tiny2), rounding::NearestEven, ftz::Disabled);
                        let winv: Tile<f32, { [128] }> = one / wrt;
                        let fjx: Tile<f32, { [128] }> = select(wok, wx * winv, one);
                        let fjy: Tile<f32, { [128] }> = select(wok, wy * winv, zero);
                        let fjz: Tile<f32, { [128] }> = select(wok, wz * winv, zero);
                        al_x = al_x + select(aligned, fjx, zero);
                        al_y = al_y + select(aligned, fjy, zero);
                        al_z = al_z + select(aligned, fjz, zero);
                        n_align = n_align + select(aligned, one_i, zero_i);
                    }
                }
            }
        }


        let mut cs_x: Tile<f32, { [128] }> = zero;
        let mut cs_y: Tile<f32, { [128] }> = zero;
        let mut cs_z: Tile<f32, { [128] }> = zero;
        let mut n_g: Tile<i32, { [128] }> = zero_i;

        // ---- walk over the box of radius rr (cells lo..hi, then the birds in each) ----
        let box_lo_x: Tile<i32, { [128] }> = convert_tile(floor((x - rr) * inv_cell));
        let box_lo_y: Tile<i32, { [128] }> = convert_tile(floor((y - rr) * inv_cell));
        let box_lo_z: Tile<i32, { [128] }> = convert_tile(floor((z - rr) * inv_cell));
        let box_hi_x: Tile<i32, { [128] }> = convert_tile(floor((x + rr) * inv_cell));
        let box_hi_y: Tile<i32, { [128] }> = convert_tile(floor((y + rr) * inv_cell));
        let box_hi_z: Tile<i32, { [128] }> = convert_tile(floor((z + rr) * inv_cell));
        let span_x: Tile<i32, { [128] }> = select(due, box_hi_x - box_lo_x, zero_i);
        let span_y: Tile<i32, { [128] }> = select(due, box_hi_y - box_lo_y, zero_i);
        let span_z: Tile<i32, { [128] }> = select(due, box_hi_z - box_lo_z, zero_i);
        let max_x: Tile<i32, { [] }> = reduce_max(span_x, 0i32);
        let max_y: Tile<i32, { [] }> = reduce_max(span_y, 0i32);
        let max_z: Tile<i32, { [] }> = reduce_max(span_z, 0i32);
        let nx0: i32 = tile_to_scalar(max_x);
        let nx: i32 = nx0 + 1i32;
        let ny0: i32 = tile_to_scalar(max_y);
        let ny: i32 = ny0 + 1i32;
        let nz0: i32 = tile_to_scalar(max_z);
        let nz: i32 = nz0 + 1i32;
        for oz in 0i32..nz {
            for oy in 0i32..ny {
                for ox in 0i32..nx {
                    let cell_x: Tile<i32, { [128] }> = box_lo_x + broadcast_scalar(ox, shape![128]);
                    let cell_y: Tile<i32, { [128] }> = box_lo_y + broadcast_scalar(oy, shape![128]);
                    let cell_z: Tile<i32, { [128] }> = box_lo_z + broadcast_scalar(oz, shape![128]);
                    let in_x: Tile<bool, { [128] }> = le_tile(cell_x, box_hi_x);
                    let in_y: Tile<bool, { [128] }> = le_tile(cell_y, box_hi_y);
                    let in_z: Tile<bool, { [128] }> = le_tile(cell_z, box_hi_z);
                    let in_xy: Tile<bool, { [128] }> = band(in_x, in_y);
                    let in_xyz: Tile<bool, { [128] }> = band(in_xy, in_z);
                    let in_box: Tile<bool, { [128] }> = band(in_xyz, due);
                    // cell_intersects_sphere
                    let lo_fx: Tile<f32, { [128] }> = convert_tile(cell_x) * four;
                    let lo_fy: Tile<f32, { [128] }> = convert_tile(cell_y) * four;
                    let lo_fz: Tile<f32, { [128] }> = convert_tile(cell_z) * four;
                    let nxd: Tile<f32, { [128] }> = min_tile(max_tile(x, lo_fx), lo_fx + four) - x;
                    let nyd: Tile<f32, { [128] }> = min_tile(max_tile(y, lo_fy), lo_fy + four) - y;
                    let nzd: Tile<f32, { [128] }> = min_tile(max_tile(z, lo_fz), lo_fz + four) - z;
                    let near: Tile<bool, { [128] }> = le_tile(nxd * nxd + nyd * nyd + nzd * nzd, rr2 + tiny);
                    let act: Tile<bool, { [128] }> = band(in_box, near);
                    // bucket_of(cell) and its range in the sorted array
                    let hh: Tile<i32, { [128] }> = xori(xori(cell_x * m1, cell_y * m2), cell_z * m3);
                    let bk: Tile<i32, { [128] }> = andi(hh, mk);
                    let st: Tile<i32, { [128] }> = gather_i32(offsets, bk, act);
                    let en: Tile<i32, { [128] }> = gather_i32(offsets, bk + one_i, act);
                    let cnt: Tile<i32, { [128] }> = en - st;
                    let max_cnt: Tile<i32, { [] }> = reduce_max(cnt, 0i32);
                    let mc: i32 = tile_to_scalar(max_cnt);
                    for k in 0i32..mc {
                        let kk: Tile<i32, { [128] }> = broadcast_scalar(k, shape![128]);
                        let live: Tile<bool, { [128] }> = lt_tile(kk, cnt);
                        let j: Tile<i32, { [128] }> = gather_i32(sorted, st + kk, live);
                        let qx: Tile<f32, { [128] }> = gather_f32(px, j, live);
                        let qy: Tile<f32, { [128] }> = gather_f32(py, j, live);
                        let qz: Tile<f32, { [128] }> = gather_f32(pz, j, live);
                        // the exact cell check: only birds really in this cell, not just this bucket
                        let qcx: Tile<i32, { [128] }> = convert_tile(floor(qx * inv_cell));
                        let qcy: Tile<i32, { [128] }> = convert_tile(floor(qy * inv_cell));
                        let qcz: Tile<i32, { [128] }> = convert_tile(floor(qz * inv_cell));
                        let same_x: Tile<bool, { [128] }> = eq_tile(qcx, cell_x);
                        let same_y: Tile<bool, { [128] }> = eq_tile(qcy, cell_y);
                        let same_z: Tile<bool, { [128] }> = eq_tile(qcz, cell_z);
                        let same_xy: Tile<bool, { [128] }> = band(same_x, same_y);
                        let same: Tile<bool, { [128] }> = band(same_xy, same_z);
                        let not_self: Tile<bool, { [128] }> = ne_tile(j, idx);
                        let dx: Tile<f32, { [128] }> = qx - x;
                        let dy: Tile<f32, { [128] }> = qy - y;
                        let dz: Tile<f32, { [128] }> = qz - z;
                        let d2: Tile<f32, { [128] }> = dx * dx + dy * dy + dz * dz;
                        let within: Tile<bool, { [128] }> = le_tile(d2, rr2);
                        let m0: Tile<bool, { [128] }> = band(live, same);
                        let m1b: Tile<bool, { [128] }> = band(m0, not_self);
                        let m: Tile<bool, { [128] }> = band(m1b, within);
                        let dist: Tile<f32, { [128] }> = sqrt(d2, rounding::NearestEven, ftz::Disabled);
                        let dnorm: Tile<f32, { [128] }> = max_tile(dist, eps);
                        cs_x = cs_x + select(m, dx / dnorm, zero);
                        cs_y = cs_y + select(m, dy / dnorm, zero);
                        cs_z = cs_z + select(m, dz / dnorm, zero);
                        n_g = n_g + select(m, one_i, zero_i);
                    }
                }
            }
        }


        // ---- combine (flock-ref model::decide) ----
        let nall_f: Tile<f32, { [128] }> = convert_tile(n_all);
        let has_n: Tile<bool, { [128] }> = gt_tile(nall_f, zero);
        let inv_n: Tile<f32, { [128] }> = select(has_n, one / max_tile(nall_f, one), zero);
        let mut fx_f: Tile<f32, { [128] }> = sep_x * inv_n;
        let mut fy_f: Tile<f32, { [128] }> = sep_y * inv_n;
        let mut fz_f: Tile<f32, { [128] }> = sep_z * inv_n;

        let ng_f: Tile<f32, { [128] }> = convert_tile(n_g);
        let has_g: Tile<bool, { [128] }> = gt_tile(ng_f, zero);
        let cs2: Tile<f32, { [128] }> = cs_x * cs_x + cs_y * cs_y + cs_z * cs_z;
        let cen_raw: Tile<f32, { [128] }> = sqrt(cs2, rounding::NearestEven, ftz::Disabled) / max_tile(ng_f, one);
        let cen: Tile<f32, { [128] }> = select(has_g, cen_raw, zero);

        let coh2: Tile<f32, { [128] }> = coh_x * coh_x + coh_y * coh_y + coh_z * coh_z;
        let coh_ok: Tile<bool, { [128] }> = gt_tile(coh2, zero);
        let coh_inv: Tile<f32, { [128] }> = one / sqrt(max_tile(coh2, tiny8), rounding::NearestEven, ftz::Disabled);
        let w_coh: Tile<f32, { [128] }> = constant(0.65f32, shape![128]);
        let coh_k: Tile<f32, { [128] }> = select(coh_ok, coh_inv * w_coh * cen, zero);
        fx_f = fx_f + coh_x * coh_k;
        fy_f = fy_f + coh_y * coh_k;
        fz_f = fz_f + coh_z * coh_k;

        let nal_f: Tile<f32, { [128] }> = convert_tile(n_align);
        let has_a: Tile<bool, { [128] }> = gt_tile(nal_f, zero);
        let inv_a: Tile<f32, { [128] }> = select(has_a, one / max_tile(nal_f, one), zero);
        let w_align: Tile<f32, { [128] }> = constant(0.5f32, shape![128]);
        let ax: Tile<f32, { [128] }> = select(has_a, al_x * inv_a - fx, zero);
        let ay: Tile<f32, { [128] }> = select(has_a, al_y * inv_a - fy, zero);
        let az: Tile<f32, { [128] }> = select(has_a, al_z * inv_a - fz, zero);
        fx_f = fx_f + ax * w_align;
        fy_f = fy_f + ay * w_align;
        fz_f = fz_f + az * w_align;

        // vertical roost pull (Eq. 12), capped
        let z0: Tile<f32, { [128] }> = constant(100.0f32, shape![128]);
        let hroost: Tile<f32, { [128] }> = constant(40.0f32, shape![128]);
        let w_roost: Tile<f32, { [128] }> = constant(0.2f32, shape![128]);
        let neg_one: Tile<f32, { [128] }> = constant(-1.0f32, shape![128]);
        let pull: Tile<f32, { [128] }> = min_tile(max_tile((z0 - z) / hroost, neg_one), one);
        fz_f = fz_f + w_roost * pull;

        // speed control (Eq. 1): f_i * ((MASS / TAU) * (V0 - speed))
        let speed: Tile<f32, { [128] }> = sqrt(v2, rounding::NearestEven, ftz::Disabled);
        let ten: Tile<f32, { [128] }> = constant(10.0f32, shape![128]);
        let mass_tau: Tile<f32, { [128] }> = constant(0.08f32, shape![128]);
        let sc: Tile<f32, { [128] }> = mass_tau * (ten - speed);
        fx_f = fx_f + fx * sc;
        fy_f = fy_f + fy * sc;
        fz_f = fz_f + fz * sc;

        // radius_step: r * (N_C / max(n, 1))^(RADIUS_LERP / 3), clamped
        let n_c: Tile<f32, { [128] }> = constant(6.5f32, shape![128]);
        let expo: Tile<f32, { [128] }> = constant(0.033333335f32, shape![128]);
        let fac: Tile<f32, { [128] }> = pow(n_c / max_tile(nall_f, one), expo);
        let r_max: Tile<f32, { [128] }> = constant(100.0f32, shape![128]);
        let r_new: Tile<f32, { [128] }> = min_tile(max_tile(r * fac, rh), r_max);

        // stores, only for birds that decided this step
        let du: Tile<f32, { [128] }> = constant(0.05f32, shape![128]);
        scatter_f32(sx, idx, fx_f, due);
        scatter_f32(sy, idx, fy_f, due);
        scatter_f32(sz, idx, fz_f, due);
        scatter_f32(radius, idx, r_new, due);
        scatter_f32(next_decision, idx, nd + du, due);
        scatter_f32(centrality, idx, cen, due);
        scatter_i32(neighbours, idx, n_all, due);
    }
    // GENERATED-END
}
