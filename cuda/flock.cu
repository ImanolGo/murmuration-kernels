// CUDA C++ baseline for the flock core step (docs/SPEC.md).
//
// A straightforward port of the passes murmur-gpu runs with wgpu: hash each bird
// into a cell bucket, counting-sort the birds by bucket (histogram, two-level
// scan, scatter), then decide and integrate. The counting sort is hand-written
// on purpose: every backend in this repo uses the same grid algorithm, and a
// Rust kernel cannot call CUB. (The first baseline used cub::DeviceRadixSort;
// see docs/progress.md for its numbers.) Mirrors crates/flock-ref/src/model.rs operation by
// operation. This is the unoptimised reference point; optimisations go into
// separate, named commits so each one can be measured.
//
// Build: make -C cuda          Run: cuda/build/flock_cuda --in start.flk --steps 100 --out end.flk

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "params.cuh"

#define CK(call)                                                                      \
    do {                                                                              \
        cudaError_t e_ = (call);                                                      \
        if (e_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

// ---- float3 helpers ---------------------------------------------------------

__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 operator/(float3 a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }
__device__ __forceinline__ float3& operator+=(float3& a, float3 b) { a = a + b; return a; }
__device__ __forceinline__ float3& operator-=(float3& a, float3 b) { a = a - b; return a; }
__device__ __forceinline__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float3 xyz(float4 v) { return make_float3(v.x, v.y, v.z); }
__device__ __forceinline__ float3 zero3() { return make_float3(0.f, 0.f, 0.f); }
__device__ __forceinline__ float clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }

__device__ __forceinline__ float3 normalize_or_zero(float3 v) {
    float l2 = dot3(v, v);
    return l2 > 0.f ? v * (1.0f / sqrtf(l2)) : zero3();
}
// flock-ref `forward`: unit direction with a stable fallback.
__device__ __forceinline__ float3 forward_of(float3 v) {
    float l2 = dot3(v, v);
    return l2 > 1e-12f ? v * (1.0f / sqrtf(l2)) : make_float3(1.f, 0.f, 0.f);
}
// flock-ref `side_of`: normalize(f x z) with a fallback.
__device__ __forceinline__ float3 side_of(float3 f) {
    float3 s = cross3(f, make_float3(0.f, 0.f, 1.f));
    float l2 = dot3(s, s);
    return l2 > 1e-8f ? s * (1.0f / sqrtf(l2)) : make_float3(1.f, 0.f, 0.f);
}
__device__ __forceinline__ float3 body_up(float3 f, float beta) {
    float3 side = side_of(f);
    float3 up0 = cross3(side, f);
    return up0 * cosf(beta) + side * sinf(beta);
}

// ---- spatial hash (murmur-ref neighbours.rs) --------------------------------

constexpr float INV_CELL = 1.0f / GRID_CELL;

__device__ __forceinline__ int3 cell_of(float3 p) {
    return make_int3((int)floorf(p.x * INV_CELL), (int)floorf(p.y * INV_CELL), (int)floorf(p.z * INV_CELL));
}
__device__ __forceinline__ uint32_t bucket_of(int3 c, uint32_t mask) {
    uint32_t h = (uint32_t)c.x * 73856093u ^ (uint32_t)c.y * 19349663u ^ (uint32_t)c.z * 83492791u;
    return h & mask;
}
// Skip cells the query sphere cannot touch (as murmur-gpu's sim.wgsl does).
__device__ __forceinline__ bool cell_intersects_sphere(int3 c, float3 p, float r2) {
    const float lo_x = c.x * GRID_CELL, lo_y = c.y * GRID_CELL, lo_z = c.z * GRID_CELL;
    float dx = clampf(p.x, lo_x, lo_x + GRID_CELL) - p.x;
    float dy = clampf(p.y, lo_y, lo_y + GRID_CELL) - p.y;
    float dz = clampf(p.z, lo_z, lo_z + GRID_CELL) - p.z;
    return dx * dx + dy * dy + dz * dz <= r2 + 1e-4f;
}

struct Flock {
    int n;
    uint32_t buckets;
    float4* pos;          // xyz, w unused
    float4* vel;          // xyz, w unused
    float4* steer;        // xyz, w unused
    float* roll;
    float* spin;
    float* radius;
    float* next_decision;
    float* centrality;
    uint32_t* neighbours;
    // grid (the passes of murmur-gpu's grid.wgsl)
    uint32_t* bucket;      // bucket of each bird
    uint32_t* counts;      // histogram, then re-seeded as the scatter cursor
    uint32_t* offsets;     // exclusive scan of counts; buckets + 1 entries
    uint32_t* block_sums;  // one total per scan block
    uint32_t* sorted;      // bird indices ordered by bucket
};

constexpr int SCAN_WG = 256;
constexpr int SCAN_EPT = 8;                   // elements per thread
constexpr int SCAN_BLOCK = SCAN_WG * SCAN_EPT;  // buckets per scan block

// Records each bird's bucket and counts one per bucket (counts zeroed first).
__global__ void grid_histogram(Flock f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f.n) return;
    uint32_t b = bucket_of(cell_of(xyz(f.pos[i])), f.buckets - 1);
    f.bucket[i] = b;
    atomicAdd(&f.counts[b], 1u);
}

// Exclusive scan of 256 per-thread sums in shared memory (Hillis-Steele).
// Returns this thread's exclusive prefix; `total` is the block total.
__device__ __forceinline__ uint32_t block_exclusive_scan(uint32_t v, uint32_t* total) {
    __shared__ uint32_t sh[SCAN_WG];
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int off = 1; off < SCAN_WG; off <<= 1) {
        uint32_t x = t >= off ? sh[t - off] : 0u;
        __syncthreads();
        sh[t] += x;
        __syncthreads();
    }
    *total = sh[SCAN_WG - 1];
    return sh[t] - v;
}

// Level 1: each block scans SCAN_BLOCK buckets and writes its total.
__global__ void grid_scan_blocks(Flock f) {
    const uint32_t base = blockIdx.x * SCAN_BLOCK + threadIdx.x * SCAN_EPT;
    uint32_t c[SCAN_EPT], sum = 0;
    for (int k = 0; k < SCAN_EPT; ++k) { c[k] = f.counts[base + k]; sum += c[k]; }
    uint32_t total;
    uint32_t run = block_exclusive_scan(sum, &total);
    for (int k = 0; k < SCAN_EPT; ++k) { f.offsets[base + k] = run; run += c[k]; }
    if (threadIdx.x == 0) f.block_sums[blockIdx.x] = total;
}

// Level 2: one block scans the block totals in place (at most 2048 of them).
__global__ void grid_scan_totals(Flock f, uint32_t nblocks) {
    const uint32_t base = threadIdx.x * SCAN_EPT;
    uint32_t c[SCAN_EPT], sum = 0;
    for (int k = 0; k < SCAN_EPT; ++k) { c[k] = base + k < nblocks ? f.block_sums[base + k] : 0u; sum += c[k]; }
    uint32_t total;
    uint32_t run = block_exclusive_scan(sum, &total);
    for (int k = 0; k < SCAN_EPT; ++k) { if (base + k < nblocks) f.block_sums[base + k] = run; run += c[k]; }
}

// Level 3: add each block's start offset, re-seed counts as the scatter cursor.
__global__ void grid_scan_add(Flock f) {
    uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= f.buckets) return;
    uint32_t o = f.offsets[b] + f.block_sums[b / SCAN_BLOCK];
    f.offsets[b] = o;
    f.counts[b] = o;
    if (b == 0) f.offsets[f.buckets] = f.n;
}

// Writes each bird's index into its bucket's range.
__global__ void grid_scatter(Flock f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f.n) return;
    f.sorted[atomicAdd(&f.counts[f.bucket[i]], 1u)] = i;
}

// radius_step from flock-ref params.rs.
__device__ __forceinline__ float radius_step(float r, uint32_t n) {
    float nn = (float)(n > 1 ? n : 1);
    float factor = powf(N_C / nn, RADIUS_LERP / 3.0f);
    return clampf(r * factor, R_H, R_MAX);
}

// Mirrors flock-ref model::decide for bird i. Reads positions and velocities
// (written only by `integrate`), writes only bird i's own decision fields.
__global__ void decide(Flock f, float time) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f.n) return;
    if (time < f.next_decision[i]) return;

    const float3 p = xyz(f.pos[i]);
    const float3 v_i = xyz(f.vel[i]);
    const float r = f.radius[i];
    const float3 f_i = forward_of(v_i);
    const uint32_t mask = f.buckets - 1;

    float3 sep = zero3(), coh = zero3(), align = zero3();
    uint32_t n_all = 0, n_star = 0, n_align = 0;

    // Neighbourhood walk (Eq. 3-6): separation, cohesion, alignment.
    {
        const int3 lo = cell_of(make_float3(p.x - r, p.y - r, p.z - r));
        const int3 hi = cell_of(make_float3(p.x + r, p.y + r, p.z + r));
        const float r2 = r * r;
        for (int cz = lo.z; cz <= hi.z; ++cz)
            for (int cy = lo.y; cy <= hi.y; ++cy)
                for (int cx = lo.x; cx <= hi.x; ++cx) {
                    const int3 cell = make_int3(cx, cy, cz);
                    if (!cell_intersects_sphere(cell, p, r2)) continue;
                    const uint32_t b = bucket_of(cell, mask);
                    for (uint32_t s = f.offsets[b], e = f.offsets[b + 1]; s < e; ++s) {
                        const uint32_t j = f.sorted[s];
                        if ((int)j == i) continue;
                        const float4 qj = f.pos[j];
                        const float3 q = xyz(qj);
                        const int3 cq = cell_of(q);
                        if (cq.x != cx || cq.y != cy || cq.z != cz) continue;
                        const float3 d = q - p;
                        const float d2 = dot3(d, d);
                        if (d2 > r2) continue;
                        const float dist = sqrtf(d2);
                        ++n_all;
                        const float3 dhat = d / fmaxf(dist, 1e-6f);
                        const float g = dist <= R_H ? 1.0f : expf(-((dist - R_H) * (dist - R_H)) / (2.0f * SIGMA * SIGMA));
                        sep -= dhat * g;
                        if (dist > R_H) {
                            coh += dhat;
                            ++n_star;
                        }
                        if (dot3(f_i, dhat) >= BLIND_COS) {
                            align += forward_of(xyz(f.vel[j]));
                            ++n_align;
                        }
                    }
                }
    }

    float3 force = zero3();
    if (n_all > 0) force += sep * (W_SEP / (float)n_all);

    // Centrality over 2R_i (Eq. 7).
    float3 csum = zero3();
    uint32_t n_g = 0;
    {
        const float r2 = 4.0f * r * r;
        const float rr = 2.0f * r;
        const int3 lo = cell_of(make_float3(p.x - rr, p.y - rr, p.z - rr));
        const int3 hi = cell_of(make_float3(p.x + rr, p.y + rr, p.z + rr));
        for (int cz = lo.z; cz <= hi.z; ++cz)
            for (int cy = lo.y; cy <= hi.y; ++cy)
                for (int cx = lo.x; cx <= hi.x; ++cx) {
                    const int3 cell = make_int3(cx, cy, cz);
                    if (!cell_intersects_sphere(cell, p, r2)) continue;
                    const uint32_t b = bucket_of(cell, mask);
                    for (uint32_t s = f.offsets[b], e = f.offsets[b + 1]; s < e; ++s) {
                        const uint32_t j = f.sorted[s];
                        if ((int)j == i) continue;
                        const float3 q = xyz(f.pos[j]);
                        const int3 cq = cell_of(q);
                        if (cq.x != cx || cq.y != cy || cq.z != cz) continue;
                        const float3 d = q - p;
                        const float d2 = dot3(d, d);
                        if (d2 > r2) continue;
                        csum += d / fmaxf(sqrtf(d2), 1e-6f);
                        ++n_g;
                    }
                }
    }
    const float centrality = n_g > 0 ? sqrtf(dot3(csum, csum)) / (float)n_g : 0.0f;

    if (n_star > 0) force += normalize_or_zero(coh) * (W_COH * centrality);
    const float3 a = n_align > 0 ? align / (float)n_align - f_i : zero3();
    force += a * W_ALIGN;

    // Vertical roost pull (Eq. 12), capped.
    force += make_float3(0.f, 0.f, W_ROOST_V * clampf((Z0 - p.z) / H_ROOST_V, -1.0f, 1.0f));
    // Speed control (Eq. 1).
    const float speed = sqrtf(dot3(v_i, v_i));
    force += f_i * ((MASS / TAU) * (V0 - speed));

    f.steer[i] = make_float4(force.x, force.y, force.z, 0.f);
    f.radius[i] = radius_step(r, n_all);
    f.neighbours[i] = n_all;
    f.centrality[i] = centrality;
    f.next_decision[i] += DU;
}

// Mirrors flock-ref model::step_bird (base turning model, no forced roll).
__global__ void integrate(Flock f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f.n) return;
    float3 pos = xyz(f.pos[i]);
    float3 vel = xyz(f.vel[i]);
    const float3 steering = xyz(f.steer[i]);
    const float roll = f.roll[i];

    const float3 fwd = forward_of(vel);
    const float3 side = side_of(fwd);
    const float speed = fmaxf(sqrtf(dot3(vel, vel)), 1e-6f);

    const float a_lat = dot3(steering, side) / MASS;
    const float target = clampf(atanf(a_lat / G), -MAX_BANK, MAX_BANK);
    const float rate = clampf((target - roll) / TAU_ROLL, -MAX_ROLL_RATE, MAX_ROLL_RATE);
    const float beta = clampf(roll + rate * DT, -MAX_BANK, MAX_BANK);
    f.spin[i] = rate;
    f.roll[i] = beta;

    const float3 up = body_up(fwd, beta);
    const float cos_beta = fmaxf(cosf(beta), 0.2f);
    const float sr = speed / V0;
    const float lift = sr * sr * L0 / cos_beta;
    const float drag = sr * sr * D0;
    const float3 flight_force = up * lift - fwd * drag + fwd * D0 - make_float3(0.f, 0.f, MASS * G);

    const float3 control = steering - side * dot3(steering, side);
    vel += (control + flight_force) / MASS * DT;
    const float sp = sqrtf(dot3(vel, vel));
    if (sp > V_MAX) vel = vel * (V_MAX / sp);
    else if (sp < V_MIN && sp > 1e-6f) vel = vel * (V_MIN / sp);
    pos += vel * DT;

    f.pos[i] = make_float4(pos.x, pos.y, pos.z, 0.f);
    f.vel[i] = make_float4(vel.x, vel.y, vel.z, 0.f);
}

// ---- host: .flk file --------------------------------------------------------

constexpr int WORDS = 15;

struct HostFlock {
    float time = 0.f;
    std::vector<float> w;  // n * WORDS words, as in the file
    int n() const { return (int)(w.size() / WORDS); }
};

static HostFlock read_flk(const char* path) {
    FILE* fp = std::fopen(path, "rb");
    if (!fp) { std::perror(path); std::exit(1); }
    uint32_t head[4];
    if (std::fread(head, 4, 4, fp) != 4 || std::memcmp(head, "FLK1", 4) != 0 || head[2] != WORDS) {
        std::fprintf(stderr, "%s: not a FLK1 file with %d words per bird\n", path, WORDS);
        std::exit(1);
    }
    HostFlock h;
    std::memcpy(&h.time, &head[3], 4);
    h.w.resize((size_t)head[1] * WORDS);
    if (std::fread(h.w.data(), 4, h.w.size(), fp) != h.w.size()) { std::fprintf(stderr, "%s: truncated\n", path); std::exit(1); }
    std::fclose(fp);
    return h;
}

static void write_flk(const char* path, const HostFlock& h) {
    FILE* fp = std::fopen(path, "wb");
    if (!fp) { std::perror(path); std::exit(1); }
    uint32_t head[4] = {0x314B4C46u /* "FLK1" little endian */, (uint32_t)h.n(), WORDS, 0};
    std::memcpy(&head[3], &h.time, 4);
    std::fwrite(head, 4, 4, fp);
    std::fwrite(h.w.data(), 4, h.w.size(), fp);
    std::fclose(fp);
}

template <class T> static T* dalloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); return p; }

static uint32_t buckets_for(int n) {
    uint64_t want = 1;
    while (want < (uint64_t)n * 2) want <<= 1;
    return (uint32_t)std::min<uint64_t>(std::max<uint64_t>(want, 1u << 16), 1u << 22);
}

int main(int argc, char** argv) {
    const char *in = nullptr, *out = nullptr;
    int steps = 1, warmup = 0;
    for (int k = 1; k < argc; ++k) {
        std::string a = argv[k];
        auto val = [&]() -> const char* { if (k + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++k]; };
        if (a == "--in") in = val();
        else if (a == "--out") out = val();
        else if (a == "--steps") steps = std::atoi(val());
        else if (a == "--warmup") warmup = std::atoi(val());
        else { std::fprintf(stderr, "usage: flock_cuda --in FILE [--steps K] [--warmup W] [--out FILE]\n(the first W steps are not timed)\n"); return 2; }
    }
    if (!in) { std::fprintf(stderr, "missing --in\n"); return 2; }
    warmup = std::min(warmup, steps);

    HostFlock h = read_flk(in);
    const int n = h.n();

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int drv = 0, rt = 0;
    CK(cudaDriverGetVersion(&drv));
    CK(cudaRuntimeGetVersion(&rt));
    std::printf("# flock_cuda: %s (sm_%d%d), CUDA driver %d.%d runtime %d.%d\n", prop.name, prop.major, prop.minor,
                drv / 1000, drv % 1000 / 10, rt / 1000, rt % 1000 / 10);
    std::printf("# %d birds, %d steps (%d warm-up, not timed)\n", n, steps, warmup);

    Flock f{};
    f.n = n;
    f.buckets = buckets_for(n);
    f.pos = dalloc<float4>(n); f.vel = dalloc<float4>(n); f.steer = dalloc<float4>(n);
    f.roll = dalloc<float>(n); f.spin = dalloc<float>(n); f.radius = dalloc<float>(n);
    f.next_decision = dalloc<float>(n); f.centrality = dalloc<float>(n); f.neighbours = dalloc<uint32_t>(n);
    const uint32_t scan_blocks = f.buckets / SCAN_BLOCK;  // buckets is a power of two >= 2^16
    f.bucket = dalloc<uint32_t>(n); f.sorted = dalloc<uint32_t>(n);
    f.counts = dalloc<uint32_t>(f.buckets); f.offsets = dalloc<uint32_t>(f.buckets + 1);
    f.block_sums = dalloc<uint32_t>(scan_blocks);

    {  // upload: file words -> device arrays
        std::vector<float4> pos(n), vel(n), steer(n);
        std::vector<float> roll(n), spin(n), radius(n), next(n), cen(n);
        std::vector<uint32_t> nb(n);
        for (int i = 0; i < n; ++i) {
            const float* w = &h.w[(size_t)i * WORDS];
            pos[i] = make_float4(w[0], w[1], w[2], 0.f);
            vel[i] = make_float4(w[3], w[4], w[5], 0.f);
            roll[i] = w[6]; spin[i] = w[7]; radius[i] = w[8];
            steer[i] = make_float4(w[9], w[10], w[11], 0.f);
            next[i] = w[12];
            std::memcpy(&nb[i], &w[13], 4);
            cen[i] = w[14];
        }
        CK(cudaMemcpy(f.pos, pos.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(f.vel, vel.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(f.steer, steer.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(f.roll, roll.data(), n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(f.spin, spin.data(), n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(f.radius, radius.data(), n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(f.next_decision, next.data(), n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(f.centrality, cen.data(), n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(f.neighbours, nb.data(), n * 4, cudaMemcpyHostToDevice));
    }

    const int TB = 256;
    const int blocks = (n + TB - 1) / TB;
    enum { P_HISTOGRAM, P_SCAN, P_SCATTER, P_DECIDE, P_INTEGRATE, NPASS };
    const char* names[NPASS] = {"grid_histogram", "grid_scan", "grid_scatter", "decide", "integrate"};
    std::vector<cudaEvent_t> ev((size_t)(steps) * (NPASS + 1));
    for (auto& e : ev) CK(cudaEventCreate(&e));

    float time = h.time;
    for (int s = 0; s < steps; ++s) {
        cudaEvent_t* e = &ev[(size_t)s * (NPASS + 1)];
        time += DT;
        CK(cudaEventRecord(e[0]));
        CK(cudaMemsetAsync(f.counts, 0, f.buckets * 4));
        grid_histogram<<<blocks, TB>>>(f);
        CK(cudaEventRecord(e[1]));
        grid_scan_blocks<<<scan_blocks, SCAN_WG>>>(f);
        grid_scan_totals<<<1, SCAN_WG>>>(f, scan_blocks);
        grid_scan_add<<<(f.buckets + TB - 1) / TB, TB>>>(f);
        CK(cudaEventRecord(e[2]));
        grid_scatter<<<blocks, TB>>>(f);
        CK(cudaEventRecord(e[3]));
        decide<<<blocks, TB>>>(f, time);
        CK(cudaEventRecord(e[4]));
        integrate<<<blocks, TB>>>(f);
        CK(cudaEventRecord(e[5]));
    }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    const int timed = steps - warmup;
    if (timed > 0) {
        double total[NPASS] = {0};
        for (int s = warmup; s < steps; ++s) {
            cudaEvent_t* e = &ev[(size_t)s * (NPASS + 1)];
            for (int p = 0; p < NPASS; ++p) {
                float ms;
                CK(cudaEventElapsedTime(&ms, e[p], e[p + 1]));
                total[p] += ms;
            }
        }
        double sum = 0;
        std::printf("# %d timed steps (mean GPU ms per pass, CUDA events)\n", timed);
        for (int p = 0; p < NPASS; ++p) {
            std::printf("  %-16s %9.4f\n", names[p], total[p] / timed);
            sum += total[p] / timed;
        }
        std::printf("  %-16s %9.4f\n  bird-steps/s     %9.0f\n", "step total", sum, n / (sum / 1000.0));
    }

    if (out) {
        std::vector<float4> pos(n), vel(n), steer(n);
        std::vector<float> roll(n), spin(n), radius(n), next(n), cen(n);
        std::vector<uint32_t> nb(n);
        CK(cudaMemcpy(pos.data(), f.pos, n * sizeof(float4), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(vel.data(), f.vel, n * sizeof(float4), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(steer.data(), f.steer, n * sizeof(float4), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(roll.data(), f.roll, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(spin.data(), f.spin, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(radius.data(), f.radius, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(next.data(), f.next_decision, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(cen.data(), f.centrality, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(nb.data(), f.neighbours, n * 4, cudaMemcpyDeviceToHost));
        HostFlock o;
        o.time = time;
        o.w.resize((size_t)n * WORDS);
        for (int i = 0; i < n; ++i) {
            float* w = &o.w[(size_t)i * WORDS];
            w[0] = pos[i].x; w[1] = pos[i].y; w[2] = pos[i].z;
            w[3] = vel[i].x; w[4] = vel[i].y; w[5] = vel[i].z;
            w[6] = roll[i]; w[7] = spin[i]; w[8] = radius[i];
            w[9] = steer[i].x; w[10] = steer[i].y; w[11] = steer[i].z;
            w[12] = next[i];
            std::memcpy(&w[13], &nb[i], 4);
            w[14] = cen[i];
        }
        write_flk(out, o);
    }
    return 0;
}
