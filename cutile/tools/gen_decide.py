#!/usr/bin/env python3
"""Generates the `decide` kernel (and its pointer helpers) for src/lib.rs.

The cuTile Rust DSL cannot share a loop body between the two neighbourhood walks
(radius R and 2R) through a function, so the walk is a template here and the output
is committed as ordinary source between the GENERATED markers in src/lib.rs.
    python3 tools/gen_decide.py        (rewrites the marked block in place)
"""
import re, sys, pathlib

F, I, B = "Tile<f32, { [128] }>", "Tile<i32, { [128] }>", "Tile<bool, { [128] }>"

HELPERS = '''
    fn ptrs_f32(base: *mut f32, off: @I) -> PointerTile<*mut f32, { [128] }> {
        let r0: PointerTile<*mut f32, { [] }> = pointer_to_tile(base);
        let p1: PointerTile<*mut f32, { [1] }> = reshape_ptr(r0, shape![1]);
        let pb: PointerTile<*mut f32, { [128] }> = broadcast_ptr(p1, shape![128]);
        let out: PointerTile<*mut f32, { [128] }> = pb.offset_tile(off);
        out
    }

    fn ptrs_i32(base: *mut i32, off: @I) -> PointerTile<*mut i32, { [128] }> {
        let r0: PointerTile<*mut i32, { [] }> = pointer_to_tile(base);
        offset_ptrs(r0, off)
    }

    unsafe fn gather_f32(base: *mut f32, idx: @I, live: @B) -> @F {
        let ptrs: PointerTile<*mut f32, { [128] }> = ptrs_f32(base, idx);
        let (v, _t): (@F, Token) = unsafe {
            load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, Some(live), Some(0.0f32), None, Latency::<0>)
        };
        v
    }

    unsafe fn gather_i32(base: *mut i32, idx: @I, live: @B) -> @I {
        let ptrs: PointerTile<*mut i32, { [128] }> = ptrs_i32(base, idx);
        let (v, _t): (@I, Token) = unsafe {
            load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, Some(live), Some(0i32), None, Latency::<0>)
        };
        v
    }

    unsafe fn scatter_f32(base: *mut f32, idx: @I, value: @F, live: @B) {
        let ptrs: PointerTile<*mut f32, { [128] }> = ptrs_f32(base, idx);
        let _t: Token = unsafe {
            store_ptr_tko(ptrs, value, ordering::Weak, None::<scope::TileBlock>, Some(live), None, Latency::<0>)
        };
    }

    unsafe fn scatter_i32(base: *mut i32, idx: @I, value: @I, live: @B) {
        let ptrs: PointerTile<*mut i32, { [128] }> = ptrs_i32(base, idx);
        let _t: Token = unsafe {
            store_ptr_tko(ptrs, value, ordering::Weak, None::<scope::TileBlock>, Some(live), None, Latency::<0>)
        };
    }

    /// `a && b` on bool tiles.
    fn band(a: @B, b: @B) -> @B {
        let no: @B = constant(false, shape![128]);
        select(a, b, no)
    }
'''

# One neighbourhood walk. @RADIUS names the tile holding the box radius (r or 2r), @R2 its
# square. @ACCUM is the per-candidate accumulation, which differs between the two walks.
WALK = '''
        // ---- walk over the box of radius @RADIUS (cells lo..hi, then the birds in each) ----
        let box_lo_x: @I = convert_tile(floor((x - @RADIUS) * inv_cell));
        let box_lo_y: @I = convert_tile(floor((y - @RADIUS) * inv_cell));
        let box_lo_z: @I = convert_tile(floor((z - @RADIUS) * inv_cell));
        let box_hi_x: @I = convert_tile(floor((x + @RADIUS) * inv_cell));
        let box_hi_y: @I = convert_tile(floor((y + @RADIUS) * inv_cell));
        let box_hi_z: @I = convert_tile(floor((z + @RADIUS) * inv_cell));
        let span_x: @I = select(due, box_hi_x - box_lo_x, zero_i);
        let span_y: @I = select(due, box_hi_y - box_lo_y, zero_i);
        let span_z: @I = select(due, box_hi_z - box_lo_z, zero_i);
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
                    let cell_x: @I = box_lo_x + broadcast_scalar(ox, shape![128]);
                    let cell_y: @I = box_lo_y + broadcast_scalar(oy, shape![128]);
                    let cell_z: @I = box_lo_z + broadcast_scalar(oz, shape![128]);
                    let in_x: @B = le_tile(cell_x, box_hi_x);
                    let in_y: @B = le_tile(cell_y, box_hi_y);
                    let in_z: @B = le_tile(cell_z, box_hi_z);
                    let in_xy: @B = band(in_x, in_y);
                    let in_xyz: @B = band(in_xy, in_z);
                    let in_box: @B = band(in_xyz, due);
                    // cell_intersects_sphere
                    let lo_fx: @F = convert_tile(cell_x) * four;
                    let lo_fy: @F = convert_tile(cell_y) * four;
                    let lo_fz: @F = convert_tile(cell_z) * four;
                    let nxd: @F = min_tile(max_tile(x, lo_fx), lo_fx + four) - x;
                    let nyd: @F = min_tile(max_tile(y, lo_fy), lo_fy + four) - y;
                    let nzd: @F = min_tile(max_tile(z, lo_fz), lo_fz + four) - z;
                    let near: @B = le_tile(nxd * nxd + nyd * nyd + nzd * nzd, @R2 + tiny);
                    let act: @B = band(in_box, near);
                    // bucket_of(cell) and its range in the sorted array
                    let hh: @I = xori(xori(cell_x * m1, cell_y * m2), cell_z * m3);
                    let bk: @I = andi(hh, mk);
                    let st: @I = gather_i32(offsets, bk, act);
                    let en: @I = gather_i32(offsets, bk + one_i, act);
                    let cnt: @I = en - st;
                    let max_cnt: Tile<i32, { [] }> = reduce_max(cnt, 0i32);
                    let mc: i32 = tile_to_scalar(max_cnt);
                    for k in 0i32..mc {
                        let kk: @I = broadcast_scalar(k, shape![128]);
                        let live: @B = lt_tile(kk, cnt);
                        let j: @I = gather_i32(sorted, st + kk, live);
                        let qx: @F = gather_f32(px, j, live);
                        let qy: @F = gather_f32(py, j, live);
                        let qz: @F = gather_f32(pz, j, live);
                        // the exact cell check: only birds really in this cell, not just this bucket
                        let qcx: @I = convert_tile(floor(qx * inv_cell));
                        let qcy: @I = convert_tile(floor(qy * inv_cell));
                        let qcz: @I = convert_tile(floor(qz * inv_cell));
                        let same_x: @B = eq_tile(qcx, cell_x);
                        let same_y: @B = eq_tile(qcy, cell_y);
                        let same_z: @B = eq_tile(qcz, cell_z);
                        let same_xy: @B = band(same_x, same_y);
                        let same: @B = band(same_xy, same_z);
                        let not_self: @B = ne_tile(j, idx);
                        let dx: @F = qx - x;
                        let dy: @F = qy - y;
                        let dz: @F = qz - z;
                        let d2: @F = dx * dx + dy * dy + dz * dz;
                        let within: @B = le_tile(d2, @R2);
                        let m0: @B = band(live, same);
                        let m1b: @B = band(m0, not_self);
                        let m: @B = band(m1b, within);
                        let dist: @F = sqrt(d2, rounding::NearestEven, ftz::Disabled);
@ACCUM
                    }
                }
            }
        }
'''

ACCUM_MAIN = '''                        let distc: @F = max_tile(dist, eps);
                        let dhx: @F = dx / distc;
                        let dhy: @F = dy / distc;
                        let dhz: @F = dz / distc;
                        let dm: @F = dist - rh;
                        let gauss: @F = exp(negf(dm * dm) / two_sigma2);
                        let g: @F = select(le_tile(dist, rh), one, gauss);
                        let mg: @F = select(m, g, zero);
                        sep_x = sep_x - dhx * mg;
                        sep_y = sep_y - dhy * mg;
                        sep_z = sep_z - dhz * mg;
                        n_all = n_all + select(m, one_i, zero_i);
                        let far: @B = band(m, gt_tile(dist, rh));
                        coh_x = coh_x + select(far, dhx, zero);
                        coh_y = coh_y + select(far, dhy, zero);
                        coh_z = coh_z + select(far, dhz, zero);
                        n_star = n_star + select(far, one_i, zero_i);
                        let dotf: @F = fx * dhx + fy * dhy + fz * dhz;
                        let aligned: @B = band(m, ge_tile(dotf, blind));
                        let wx: @F = gather_f32(vx, j, live);
                        let wy: @F = gather_f32(vy, j, live);
                        let wz: @F = gather_f32(vz, j, live);
                        let w2: @F = wx * wx + wy * wy + wz * wz;
                        let wok: @B = gt_tile(w2, tiny2);
                        let wrt: @F = sqrt(max_tile(w2, tiny2), rounding::NearestEven, ftz::Disabled);
                        let winv: @F = one / wrt;
                        let fjx: @F = select(wok, wx * winv, one);
                        let fjy: @F = select(wok, wy * winv, zero);
                        let fjz: @F = select(wok, wz * winv, zero);
                        al_x = al_x + select(aligned, fjx, zero);
                        al_y = al_y + select(aligned, fjy, zero);
                        al_z = al_z + select(aligned, fjz, zero);
                        n_align = n_align + select(aligned, one_i, zero_i);'''

ACCUM_CEN = '''                        let dnorm: @F = max_tile(dist, eps);
                        cs_x = cs_x + select(m, dx / dnorm, zero);
                        cs_y = cs_y + select(m, dy / dnorm, zero);
                        cs_z = cs_z + select(m, dz / dnorm, zero);
                        n_g = n_g + select(m, one_i, zero_i);'''

DECIDE = '''
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
        let i0: @I = iota(shape![128]);
        let base: @I = broadcast_scalar(pid * 128i32, shape![128]);
        let idx: @I = i0 + base;
        let nn: @I = broadcast_scalar(n, shape![128]);
        let valid: @B = lt_tile(idx, nn);
        let tm: @F = broadcast_scalar(time, shape![128]);

        let zero: @F = constant(0.0f32, shape![128]);
        let one: @F = constant(1.0f32, shape![128]);
        let four: @F = constant(4.0f32, shape![128]);
        let eps: @F = constant(1e-6f32, shape![128]);
        let tiny: @F = constant(1e-4f32, shape![128]);
        let tiny2: @F = constant(1e-12f32, shape![128]);
        let tiny8: @F = constant(1e-8f32, shape![128]);
        let inv_cell: @F = constant(0.25f32, shape![128]);
        let rh: @F = constant(0.2f32, shape![128]);
        let two_sigma2: @F = constant(3.7538f32, shape![128]); // 2 * SIGMA^2
        let blind: @F = constant(-0.70710678f32, shape![128]);
        let zero_i: @I = constant(0i32, shape![128]);
        let one_i: @I = constant(1i32, shape![128]);
        let m1: @I = constant(73856093i32, shape![128]);
        let m2: @I = constant(19349663i32, shape![128]);
        let m3: @I = constant(83492791i32, shape![128]);
        let mk: @I = broadcast_scalar(mask, shape![128]);

        let nd: @F = gather_f32(next_decision, idx, valid);
        let due: @B = band(valid, ge_tile(tm, nd));
        let x: @F = gather_f32(px, idx, valid);
        let y: @F = gather_f32(py, idx, valid);
        let z: @F = gather_f32(pz, idx, valid);
        let ux: @F = gather_f32(vx, idx, valid);
        let uy: @F = gather_f32(vy, idx, valid);
        let uz: @F = gather_f32(vz, idx, valid);
        let r: @F = gather_f32(radius, idx, valid);
        let r2: @F = r * r;
        let rr: @F = r + r;
        let rr2: @F = rr * rr;

        // forward_of(v)
        let v2: @F = ux * ux + uy * uy + uz * uz;
        let vok: @B = gt_tile(v2, tiny2);
        let vrt: @F = sqrt(max_tile(v2, tiny2), rounding::NearestEven, ftz::Disabled);
        let vinv: @F = one / vrt;
        let fx: @F = select(vok, ux * vinv, one);
        let fy: @F = select(vok, uy * vinv, zero);
        let fz: @F = select(vok, uz * vinv, zero);

        // ---- accumulators, shared by the first walk ----
        let mut sep_x: @F = zero;
        let mut sep_y: @F = zero;
        let mut sep_z: @F = zero;
        let mut coh_x: @F = zero;
        let mut coh_y: @F = zero;
        let mut coh_z: @F = zero;
        let mut al_x: @F = zero;
        let mut al_y: @F = zero;
        let mut al_z: @F = zero;
        let mut n_all: @I = zero_i;
        let mut n_star: @I = zero_i;
        let mut n_align: @I = zero_i;
@WALK1

        let mut cs_x: @F = zero;
        let mut cs_y: @F = zero;
        let mut cs_z: @F = zero;
        let mut n_g: @I = zero_i;
@WALK2

        // ---- combine (flock-ref model::decide) ----
        let nall_f: @F = convert_tile(n_all);
        let has_n: @B = gt_tile(nall_f, zero);
        let inv_n: @F = select(has_n, one / max_tile(nall_f, one), zero);
        let mut fx_f: @F = sep_x * inv_n;
        let mut fy_f: @F = sep_y * inv_n;
        let mut fz_f: @F = sep_z * inv_n;

        let ng_f: @F = convert_tile(n_g);
        let has_g: @B = gt_tile(ng_f, zero);
        let cs2: @F = cs_x * cs_x + cs_y * cs_y + cs_z * cs_z;
        let cen_raw: @F = sqrt(cs2, rounding::NearestEven, ftz::Disabled) / max_tile(ng_f, one);
        let cen: @F = select(has_g, cen_raw, zero);

        let coh2: @F = coh_x * coh_x + coh_y * coh_y + coh_z * coh_z;
        let coh_ok: @B = gt_tile(coh2, zero);
        let coh_inv: @F = one / sqrt(max_tile(coh2, tiny8), rounding::NearestEven, ftz::Disabled);
        let w_coh: @F = constant(0.65f32, shape![128]);
        let coh_k: @F = select(coh_ok, coh_inv * w_coh * cen, zero);
        fx_f = fx_f + coh_x * coh_k;
        fy_f = fy_f + coh_y * coh_k;
        fz_f = fz_f + coh_z * coh_k;

        let nal_f: @F = convert_tile(n_align);
        let has_a: @B = gt_tile(nal_f, zero);
        let inv_a: @F = select(has_a, one / max_tile(nal_f, one), zero);
        let w_align: @F = constant(0.5f32, shape![128]);
        let ax: @F = select(has_a, al_x * inv_a - fx, zero);
        let ay: @F = select(has_a, al_y * inv_a - fy, zero);
        let az: @F = select(has_a, al_z * inv_a - fz, zero);
        fx_f = fx_f + ax * w_align;
        fy_f = fy_f + ay * w_align;
        fz_f = fz_f + az * w_align;

        // vertical roost pull (Eq. 12), capped
        let z0: @F = constant(100.0f32, shape![128]);
        let hroost: @F = constant(40.0f32, shape![128]);
        let w_roost: @F = constant(0.2f32, shape![128]);
        let neg_one: @F = constant(-1.0f32, shape![128]);
        let pull: @F = min_tile(max_tile((z0 - z) / hroost, neg_one), one);
        fz_f = fz_f + w_roost * pull;

        // speed control (Eq. 1): f_i * ((MASS / TAU) * (V0 - speed))
        let speed: @F = sqrt(v2, rounding::NearestEven, ftz::Disabled);
        let ten: @F = constant(10.0f32, shape![128]);
        let mass_tau: @F = constant(0.08f32, shape![128]);
        let sc: @F = mass_tau * (ten - speed);
        fx_f = fx_f + fx * sc;
        fy_f = fy_f + fy * sc;
        fz_f = fz_f + fz * sc;

        // radius_step: r * (N_C / max(n, 1))^(RADIUS_LERP / 3), clamped
        let n_c: @F = constant(6.5f32, shape![128]);
        let expo: @F = constant(0.033333335f32, shape![128]);
        let fac: @F = pow(n_c / max_tile(nall_f, one), expo);
        let r_max: @F = constant(100.0f32, shape![128]);
        let r_new: @F = min_tile(max_tile(r * fac, rh), r_max);

        // stores, only for birds that decided this step
        let du: @F = constant(0.05f32, shape![128]);
        scatter_f32(sx, idx, fx_f, due);
        scatter_f32(sy, idx, fy_f, due);
        scatter_f32(sz, idx, fz_f, due);
        scatter_f32(radius, idx, r_new, due);
        scatter_f32(next_decision, idx, nd + du, due);
        scatter_f32(centrality, idx, cen, due);
        scatter_i32(neighbours, idx, n_all, due);
    }
'''

def sub_types(t):
    return t.replace("@F", F).replace("@I", I).replace("@B", B)

def build():
    w1 = WALK.replace("@RADIUS", "r").replace("@R2", "r2").replace("@ACCUM", ACCUM_MAIN)
    w2 = WALK.replace("@RADIUS", "rr").replace("@R2", "rr2").replace("@ACCUM", ACCUM_CEN)
    body = DECIDE.replace("@WALK1", w1).replace("@WALK2", w2)
    return sub_types(HELPERS) + sub_types(body)

if __name__ == "__main__":
    path = pathlib.Path(__file__).resolve().parent.parent / "src" / "lib.rs"
    s = path.read_text()
    begin, end = "    // GENERATED-BEGIN (tools/gen_decide.py)\n", "    // GENERATED-END\n"
    block = begin + build() + end
    if begin in s:
        s = re.sub(re.escape(begin) + r".*?" + re.escape(end), lambda m: block, s, flags=re.S)
    else:
        i = s.rindex("}")  # closing brace of the module
        s = s[:i] + "\n" + block + s[i:]
    path.write_text(s)
    print("rewrote", path)
