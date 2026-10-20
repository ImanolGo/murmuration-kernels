//! Model constants. Values and names are copied from `murmur-ref/src/params.rs`
//! in murmuration-gpu (MIT), restricted to what the core step uses. The CUDA
//! header `cuda/params.cuh` repeats them; `docs/SPEC.md` lists both sides.

/// Integration time step Δt [s].
pub const DT: f32 = 0.005;
/// Reaction time Δu [s]: steering is held this long between decisions.
pub const DU: f32 = 0.05;
/// Cruise speed v0 [m/s].
pub const V0: f32 = 10.0;
/// Body mass m [kg].
pub const MASS: f32 = 0.08;
/// Lift-to-drag ratio C_L/C_D.
pub const CL_CD: f32 = 3.3;
/// Standard gravity [m/s²].
pub const G: f32 = 9.81;
/// Default lift `L0 = m·g` [N].
pub const L0: f32 = MASS * G;
/// Default drag, equal to default thrust [N].
pub const D0: f32 = L0 / CL_CD;
/// Speed relaxation time τ [s].
pub const TAU: f32 = 1.0;
/// Time constant of the roll response [s].
pub const TAU_ROLL: f32 = 0.15;
/// Maximum roll rate [rad/s].
pub const MAX_ROLL_RATE: f32 = 6.0;
/// Bank angle limit [rad].
pub const MAX_BANK: f32 = 80.0 * std::f32::consts::PI / 180.0;
/// Lower speed bound [m/s].
pub const V_MIN: f32 = 0.5 * V0;
/// Upper speed bound [m/s].
pub const V_MAX: f32 = 2.0 * V0;
/// Topological range n_c.
pub const N_C: f32 = 6.5;
/// Max perception radius R_max [m].
pub const R_MAX: f32 = 100.0;
/// Interpolation factor s for the adaptive radius.
pub const RADIUS_LERP: f32 = 0.1;
/// Radius of maximal separation r_h [m].
pub const R_H: f32 = 0.2;
/// Standard deviation σ of the separation Gaussian [m].
pub const SIGMA: f32 = 1.37;
/// Separation weight [N].
pub const W_SEP: f32 = 1.0;
/// Alignment weight [N].
pub const W_ALIGN: f32 = 0.5;
/// Cohesion weight [N].
pub const W_COH: f32 = 0.65;
/// Cosine of the rear blind angle for alignment.
pub const BLIND_COS: f32 = -std::f32::consts::FRAC_1_SQRT_2;
/// Vertical roost weight [N].
pub const W_ROOST_V: f32 = 0.2;
/// Distance over which the vertical roost force ramps to its cap [m].
pub const H_ROOST_V: f32 = 40.0;
/// Preferred flight height [m].
pub const Z0: f32 = 100.0;
/// Uniform grid cell size [m].
pub const GRID_CELL: f32 = 4.0;

/// Advances an adaptive interaction radius toward `n_c` partners.
pub fn radius_step(r: f32, n: u32) -> f32 {
    let n = n.max(1) as f32;
    let factor = (N_C / n).powf(RADIUS_LERP / 3.0);
    (r * factor).clamp(R_H, R_MAX)
}
