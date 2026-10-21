// The flock core step (docs/SPEC.md) as WGSL compute shaders.
//
// Same passes, names and order of operations as cuda/flock.cu and
// oxide/src/main.rs: clear, histogram, two-level scan, scatter, decide, integrate.
// Constants repeat crates/flock-ref/src/params.rs.
//
// Every storage buffer is declared read_write because all entry points share one
// bind group layout (WGSL access must match the layout entry). The device is opened
// with the adapter's own limits, which allows the 14 storage buffers used here.

const DT: f32 = 0.005;
const DU: f32 = 0.05;
const V0: f32 = 10.0;
const MASS: f32 = 0.08;
const CL_CD: f32 = 3.3;
const G: f32 = 9.81;
const L0: f32 = MASS * G;
const D0: f32 = L0 / CL_CD;
const TAU: f32 = 1.0;
const TAU_ROLL: f32 = 0.15;
const MAX_ROLL_RATE: f32 = 6.0;
const PI_F: f32 = 3.14159265358979323846;
const MAX_BANK: f32 = 80.0 * PI_F / 180.0;
const V_MIN: f32 = 0.5 * V0;
const V_MAX: f32 = 2.0 * V0;
const N_C: f32 = 6.5;
const R_MAX: f32 = 100.0;
const RADIUS_LERP: f32 = 0.1;
const R_H: f32 = 0.2;
const SIGMA: f32 = 1.37;
const W_SEP: f32 = 1.0;
const W_ALIGN: f32 = 0.5;
const W_COH: f32 = 0.65;
const BLIND_COS: f32 = -0.70710678118654752440;
const W_ROOST_V: f32 = 0.2;
const H_ROOST_V: f32 = 40.0;
const Z0: f32 = 100.0;
const GRID_CELL: f32 = 4.0;
const INV_CELL: f32 = 1.0 / GRID_CELL;

const SCAN_WG: u32 = 256u;
const SCAN_EPT: u32 = 8u;
const SCAN_BLOCK: u32 = 2048u; // SCAN_WG * SCAN_EPT

struct Params {
    n: u32,
    buckets: u32,
    mask: u32,
    time: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> steer: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> roll: array<f32>;
@group(0) @binding(5) var<storage, read_write> spin: array<f32>;
@group(0) @binding(6) var<storage, read_write> radius: array<f32>;
@group(0) @binding(7) var<storage, read_write> next_decision: array<f32>;
@group(0) @binding(8) var<storage, read_write> centrality: array<f32>;
@group(0) @binding(9) var<storage, read_write> neighbours: array<u32>;
@group(0) @binding(10) var<storage, read_write> bucket: array<u32>;
@group(0) @binding(11) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(13) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(14) var<storage, read_write> sorted: array<u32>;

// ---- helpers (flock-ref model.rs) -----------------------------------------

fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if l2 > 0.0 {
        return v * (1.0 / sqrt(l2));
    }
    return vec3<f32>(0.0);
}

// flock-ref `forward`: unit direction with a stable fallback.
fn forward_of(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if l2 > 1e-12 {
        return v * (1.0 / sqrt(l2));
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

// flock-ref `side_of`: normalize(f x z) with a fallback.
fn side_of(f: vec3<f32>) -> vec3<f32> {
    let s = cross(f, vec3<f32>(0.0, 0.0, 1.0));
    let l2 = dot(s, s);
    if l2 > 1e-8 {
        return s * (1.0 / sqrt(l2));
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

fn body_up(f: vec3<f32>, beta: f32) -> vec3<f32> {
    let side = side_of(f);
    let up0 = cross(side, f);
    return up0 * cos(beta) + side * sin(beta);
}

fn cell_of(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(p * INV_CELL));
}

fn bucket_of(c: vec3<i32>) -> u32 {
    let h = (bitcast<u32>(c.x) * 73856093u) ^ (bitcast<u32>(c.y) * 19349663u) ^ (bitcast<u32>(c.z) * 83492791u);
    return h & P.mask;
}

// Skip cells the query sphere cannot touch (as murmur-gpu's sim.wgsl does).
fn cell_intersects_sphere(c: vec3<i32>, p: vec3<f32>, r2: f32) -> bool {
    let lo = vec3<f32>(c) * GRID_CELL;
    let nearest = clamp(p, lo, lo + vec3<f32>(GRID_CELL));
    let d = nearest - p;
    return dot(d, d) <= r2 + 1e-4;
}

// radius_step from flock-ref params.rs.
fn radius_step(r: f32, n: u32) -> f32 {
    let nn = f32(max(n, 1u));
    let factor = pow(N_C / nn, RADIUS_LERP / 3.0);
    return clamp(r * factor, R_H, R_MAX);
}

// ---- grid -------------------------------------------------------------------

@compute @workgroup_size(256)
fn grid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.buckets {
        return;
    }
    atomicStore(&counts[gid.x], 0u);
}

// Records each bird's bucket and counts one per bucket.
@compute @workgroup_size(256)
fn grid_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    let b = bucket_of(cell_of(pos[i].xyz));
    bucket[i] = b;
    atomicAdd(&counts[b], 1u);
}

var<workgroup> sh: array<u32, 256>;

// Exclusive scan of 256 per-thread sums (Hillis-Steele): x = this thread's
// exclusive prefix, y = the block total. Must be called from uniform control flow.
fn block_exclusive_scan(t: u32, v: u32) -> vec2<u32> {
    sh[t] = v;
    workgroupBarrier();
    for (var off = 1u; off < SCAN_WG; off = off << 1u) {
        var x = 0u;
        if t >= off {
            x = sh[t - off];
        }
        workgroupBarrier();
        sh[t] = sh[t] + x;
        workgroupBarrier();
    }
    let incl = sh[t];
    let total = sh[SCAN_WG - 1u];
    return vec2<u32>(incl - v, total);
}

// Level 1: each workgroup scans SCAN_BLOCK buckets and writes its total.
@compute @workgroup_size(256)
fn grid_scan_blocks(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let base = wid.x * SCAN_BLOCK + t * SCAN_EPT;
    var c: array<u32, 8>;
    var sum = 0u;
    for (var k = 0u; k < SCAN_EPT; k++) {
        c[k] = atomicLoad(&counts[base + k]);
        sum += c[k];
    }
    let r = block_exclusive_scan(t, sum);
    var run = r.x;
    for (var k = 0u; k < SCAN_EPT; k++) {
        offsets[base + k] = run;
        run += c[k];
    }
    if t == 0u {
        block_sums[wid.x] = r.y;
    }
}

// Level 2: one workgroup scans the block totals in place (at most 2048).
@compute @workgroup_size(256)
fn grid_scan_totals(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let nblocks = P.buckets / SCAN_BLOCK;
    let base = t * SCAN_EPT;
    var c: array<u32, 8>;
    var sum = 0u;
    for (var k = 0u; k < SCAN_EPT; k++) {
        c[k] = 0u;
        if base + k < nblocks {
            c[k] = block_sums[base + k];
        }
        sum += c[k];
    }
    let r = block_exclusive_scan(t, sum);
    var run = r.x;
    for (var k = 0u; k < SCAN_EPT; k++) {
        if base + k < nblocks {
            block_sums[base + k] = run;
        }
        run += c[k];
    }
}

// Level 3: add each block's start offset, re-seed counts as the scatter cursor.
@compute @workgroup_size(256)
fn grid_scan_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= P.buckets {
        return;
    }
    let o = offsets[b] + block_sums[b / SCAN_BLOCK];
    offsets[b] = o;
    atomicStore(&counts[b], o);
    if b == 0u {
        offsets[P.buckets] = P.n;
    }
}

// Writes each bird's index into its bucket's range.
@compute @workgroup_size(256)
fn grid_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    let slot = atomicAdd(&counts[bucket[i]], 1u);
    sorted[slot] = i;
}

// ---- decide (flock-ref model::decide) ---------------------------------------

@compute @workgroup_size(256)
fn decide(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    if P.time < next_decision[i] {
        return;
    }
    let p = pos[i].xyz;
    let v_i = vel[i].xyz;
    let r = radius[i];
    let f_i = forward_of(v_i);

    var sep = vec3<f32>(0.0);
    var coh = vec3<f32>(0.0);
    var align = vec3<f32>(0.0);
    var n_all = 0u;
    var n_star = 0u;
    var n_align = 0u;

    // Neighbourhood walk (Eq. 3-6): separation, cohesion, alignment.
    {
        let lo = cell_of(p - vec3<f32>(r));
        let hi = cell_of(p + vec3<f32>(r));
        let r2 = r * r;
        for (var cz = lo.z; cz <= hi.z; cz++) {
            for (var cy = lo.y; cy <= hi.y; cy++) {
                for (var cx = lo.x; cx <= hi.x; cx++) {
                    let cell = vec3<i32>(cx, cy, cz);
                    if !cell_intersects_sphere(cell, p, r2) {
                        continue;
                    }
                    let b = bucket_of(cell);
                    let e = offsets[b + 1u];
                    for (var s = offsets[b]; s < e; s++) {
                        let j = sorted[s];
                        if j == i {
                            continue;
                        }
                        let q = pos[j].xyz;
                        if any(cell_of(q) != cell) {
                            continue;
                        }
                        let d = q - p;
                        let d2 = dot(d, d);
                        if d2 > r2 {
                            continue;
                        }
                        let dist = sqrt(d2);
                        n_all++;
                        let dhat = d / max(dist, 1e-6);
                        var g = 1.0;
                        if dist > R_H {
                            g = exp(-((dist - R_H) * (dist - R_H)) / (2.0 * SIGMA * SIGMA));
                        }
                        sep -= dhat * g;
                        if dist > R_H {
                            coh += dhat;
                            n_star++;
                        }
                        if dot(f_i, dhat) >= BLIND_COS {
                            align += forward_of(vel[j].xyz);
                            n_align++;
                        }
                    }
                }
            }
        }
    }

    var force = vec3<f32>(0.0);
    if n_all > 0u {
        force += sep * (W_SEP / f32(n_all));
    }

    // Centrality over 2R_i (Eq. 7).
    var csum = vec3<f32>(0.0);
    var n_g = 0u;
    {
        let r2 = 4.0 * r * r;
        let rr = 2.0 * r;
        let lo = cell_of(p - vec3<f32>(rr));
        let hi = cell_of(p + vec3<f32>(rr));
        for (var cz = lo.z; cz <= hi.z; cz++) {
            for (var cy = lo.y; cy <= hi.y; cy++) {
                for (var cx = lo.x; cx <= hi.x; cx++) {
                    let cell = vec3<i32>(cx, cy, cz);
                    if !cell_intersects_sphere(cell, p, r2) {
                        continue;
                    }
                    let b = bucket_of(cell);
                    let e = offsets[b + 1u];
                    for (var s = offsets[b]; s < e; s++) {
                        let j = sorted[s];
                        if j == i {
                            continue;
                        }
                        let q = pos[j].xyz;
                        if any(cell_of(q) != cell) {
                            continue;
                        }
                        let d = q - p;
                        let d2 = dot(d, d);
                        if d2 > r2 {
                            continue;
                        }
                        csum += d / max(sqrt(d2), 1e-6);
                        n_g++;
                    }
                }
            }
        }
    }
    var cen = 0.0;
    if n_g > 0u {
        cen = sqrt(dot(csum, csum)) / f32(n_g);
    }

    if n_star > 0u {
        force += normalize_or_zero(coh) * (W_COH * cen);
    }
    var a = vec3<f32>(0.0);
    if n_align > 0u {
        a = align / f32(n_align) - f_i;
    }
    force += a * W_ALIGN;

    // Vertical roost pull (Eq. 12), capped.
    force += vec3<f32>(0.0, 0.0, W_ROOST_V * clamp((Z0 - p.z) / H_ROOST_V, -1.0, 1.0));
    // Speed control (Eq. 1).
    let speed = sqrt(dot(v_i, v_i));
    force += f_i * ((MASS / TAU) * (V0 - speed));

    steer[i] = vec4<f32>(force, 0.0);
    radius[i] = radius_step(r, n_all);
    neighbours[i] = n_all;
    centrality[i] = cen;
    next_decision[i] = next_decision[i] + DU;
}

// ---- integrate (flock-ref model::step_bird, base turning, no forced roll) ----

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    var p = pos[i].xyz;
    var v = vel[i].xyz;
    let steering = steer[i].xyz;
    let rl = roll[i];

    let fwd = forward_of(v);
    let side = side_of(fwd);
    let speed = max(sqrt(dot(v, v)), 1e-6);

    let a_lat = dot(steering, side) / MASS;
    let target_roll = clamp(atan(a_lat / G), -MAX_BANK, MAX_BANK);
    let rate = clamp((target_roll - rl) / TAU_ROLL, -MAX_ROLL_RATE, MAX_ROLL_RATE);
    let beta = clamp(rl + rate * DT, -MAX_BANK, MAX_BANK);
    spin[i] = rate;
    roll[i] = beta;

    let up = body_up(fwd, beta);
    let cos_beta = max(cos(beta), 0.2);
    let sr = speed / V0;
    let lift = sr * sr * L0 / cos_beta;
    let drag = sr * sr * D0;
    let flight_force = up * lift - fwd * drag + fwd * D0 - vec3<f32>(0.0, 0.0, MASS * G);

    let control = steering - side * dot(steering, side);
    v += (control + flight_force) / MASS * DT;
    let sp = sqrt(dot(v, v));
    if sp > V_MAX {
        v = v * (V_MAX / sp);
    } else if sp < V_MIN && sp > 1e-6 {
        v = v * (V_MIN / sp);
    }
    p += v * DT;

    pos[i] = vec4<f32>(p, 0.0);
    vel[i] = vec4<f32>(v, 0.0);
}
