//! The flock core step (docs/SPEC.md) as cuTile Rust kernels.
//!
//! Same passes and order of operations as `cuda/flock.cu`. Differences forced by
//! the tile model are listed in docs/progress.md; `bin/check.rs` compiles every
//! kernel for a target architecture without a GPU.

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

        // body_up(f, beta) = (side x f) cos(beta) + side sin(beta), side = (sdx, sdy, 0).
        let u0x: Tile<f32, { [128] }> = negf(sdy * fz);
        let u0y: Tile<f32, { [128] }> = sdx * fz;
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

        let sp: Tile<f32, { [128] }> = sqrt(ux * ux + uy * uy + uz * uz, rounding::NearestEven, ftz::Disabled);
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
}
