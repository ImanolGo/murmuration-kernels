//! The core step: decide (brute force) and integrate.
//!
//! Mirrors `murmur-ref`: `social::decide`, `flight::step_bird` and the order of
//! operations in `World::step`. Omitted on purpose (see `docs/SPEC.md`): the
//! falcon, escape and zigzag, the evening arc, the random force, the horizontal
//! roost and committed return, and the wingbeat. Neighbours are found by brute
//! force: this is the oracle, and the GPU grids must give the same sets.

use glam::Vec3;

use crate::params::*;
use crate::state::{Bird, Flock};

/// Unit forward direction of a velocity, with a stable fallback.
pub fn forward(v: Vec3) -> Vec3 {
    if v.length_squared() > 1e-12 {
        v.normalize()
    } else {
        Vec3::X
    }
}

/// The bird's sideways direction `normalize(f × ẑ)`, with a fallback.
pub fn side_of(f: Vec3) -> Vec3 {
    let s = f.cross(Vec3::Z);
    if s.length_squared() > 1e-8 {
        s.normalize()
    } else {
        Vec3::X
    }
}

/// The bird's body "up" for bank angle `beta`.
pub fn body_up(f: Vec3, beta: f32) -> Vec3 {
    let side = side_of(f);
    let up0 = side.cross(f);
    up0 * beta.cos() + side * beta.sin()
}

/// The outcome of one bird's decision.
pub struct Decision {
    /// Steering force held until the next decision [N].
    pub force: Vec3,
    /// New interaction radius [m].
    pub radius: f32,
    /// Neighbours within the current radius.
    pub neighbours: u32,
    /// Centrality (Eq. 7).
    pub centrality: f32,
}

/// Computes bird `i`'s decision from a snapshot of the flock.
pub fn decide(i: usize, birds: &[Bird], z0: f32) -> Decision {
    let me = &birds[i];
    let p = me.pos;
    let r = me.radius;
    let f_i = forward(me.vel);

    let mut sep = Vec3::ZERO;
    let mut coh = Vec3::ZERO;
    let mut align = Vec3::ZERO;
    let mut n_all = 0u32;
    let mut n_star = 0u32;
    let mut n_align = 0u32;

    let r2 = r * r;
    for (j, other) in birds.iter().enumerate() {
        if j == i {
            continue;
        }
        let d = other.pos - p;
        let d2 = d.length_squared();
        if d2 > r2 {
            continue;
        }
        let dist = d2.sqrt();
        n_all += 1;
        let dhat = d / dist.max(1e-6);
        let g = if dist <= R_H {
            1.0
        } else {
            (-(dist - R_H).powi(2) / (2.0 * SIGMA * SIGMA)).exp()
        };
        sep -= dhat * g;
        if dist > R_H {
            coh += dhat;
            n_star += 1;
        }
        if f_i.dot(dhat) >= BLIND_COS {
            align += forward(other.vel);
            n_align += 1;
        }
    }

    let mut force = Vec3::ZERO;
    if n_all > 0 {
        force += sep * (W_SEP / n_all as f32);
    }

    // Centrality over 2R_i (Eq. 7).
    let c2 = (2.0 * r) * (2.0 * r);
    let mut csum = Vec3::ZERO;
    let mut n_g = 0u32;
    for (j, other) in birds.iter().enumerate() {
        if j == i {
            continue;
        }
        let d = other.pos - p;
        let d2 = d.length_squared();
        if d2 > c2 {
            continue;
        }
        csum += d / d2.sqrt().max(1e-6);
        n_g += 1;
    }
    let centrality = if n_g > 0 {
        csum.length() / n_g as f32
    } else {
        0.0
    };

    if n_star > 0 {
        force += coh.normalize_or_zero() * (W_COH * centrality);
    }
    let a = if n_align > 0 {
        align / n_align as f32 - f_i
    } else {
        Vec3::ZERO
    };
    force += a * W_ALIGN;

    // Vertical roost pull (Eq. 12), capped.
    force += Vec3::Z * (W_ROOST_V * ((z0 - p.z) / H_ROOST_V).clamp(-1.0, 1.0));

    // Speed control (Eq. 1).
    let speed = me.vel.length();
    force += f_i * ((MASS / TAU) * (V0 - speed));

    Decision {
        force,
        radius: radius_step(r, n_all),
        neighbours: n_all,
        centrality,
    }
}

/// Advances one bird by `dt` under its held steering force (base turning
/// model, no forced roll).
pub fn step_bird(b: &mut Bird, dt: f32) {
    let f = forward(b.vel);
    let side = side_of(f);
    let speed = b.vel.length().max(1e-6);

    let a_lat = b.steering.dot(side) / MASS;
    let target = (a_lat / G).atan().clamp(-MAX_BANK, MAX_BANK);
    let rate = ((target - b.roll) / TAU_ROLL).clamp(-MAX_ROLL_RATE, MAX_ROLL_RATE);
    b.spin = rate;
    let beta = (b.roll + rate * dt).clamp(-MAX_BANK, MAX_BANK);
    b.roll = beta;

    let up = body_up(f, beta);
    let cos_beta = beta.cos().max(0.2);
    let lift = (speed / V0).powi(2) * L0 / cos_beta;
    let drag = (speed / V0).powi(2) * D0;
    let flight_force = up * lift - f * drag + f * D0 - Vec3::Z * MASS * G;

    let control = b.steering - side * b.steering.dot(side);
    b.vel += (control + flight_force) / MASS * dt;
    let sp = b.vel.length();
    if sp > V_MAX {
        b.vel *= V_MAX / sp;
    } else if sp < V_MIN && sp > 1e-6 {
        b.vel *= V_MIN / sp;
    }
    b.pos += b.vel * dt;
}

impl Flock {
    /// One integration step, in the order of `World::step`: advance the clock,
    /// decide for every bird whose reaction is due (from a snapshot), then
    /// integrate every bird.
    pub fn step(&mut self) {
        self.time += DT;
        let time = self.time;
        let decisions: Vec<Option<Decision>> = (0..self.birds.len())
            .map(|i| (time >= self.birds[i].next_decision).then(|| decide(i, &self.birds, Z0)))
            .collect();
        for (b, d) in self.birds.iter_mut().zip(decisions) {
            if let Some(d) = d {
                b.steering = d.force;
                b.radius = d.radius;
                b.neighbours = d.neighbours;
                b.centrality = d.centrality;
                b.next_decision += DU;
            }
        }
        for b in &mut self.birds {
            step_bird(b, DT);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::START_RADIUS;

    #[test]
    fn level_cruise_holds_height_and_speed() {
        let mut b = Bird {
            pos: Vec3::new(0.0, 0.0, 100.0),
            vel: Vec3::X * V0,
            ..Bird::default()
        };
        for _ in 0..200 {
            step_bird(&mut b, DT);
        }
        assert!((b.pos.z - 100.0).abs() < 1.0, "z = {}", b.pos.z);
        assert!((b.vel.length() - V0).abs() < 0.5);
    }

    #[test]
    fn a_lone_bird_gets_only_roost_and_speed_forces() {
        let birds = vec![Bird {
            pos: Vec3::new(0.0, 0.0, Z0),
            vel: Vec3::X * V0,
            radius: START_RADIUS,
            ..Bird::default()
        }];
        let d = decide(0, &birds, Z0);
        assert_eq!(d.neighbours, 0);
        assert!(d.force.length() < 1e-6, "{:?}", d.force);
    }

    #[test]
    fn two_close_birds_separate() {
        let birds = vec![
            Bird {
                pos: Vec3::new(0.0, 0.0, Z0),
                vel: Vec3::X * V0,
                radius: 2.0,
                ..Bird::default()
            },
            Bird {
                pos: Vec3::new(0.0, 0.5, Z0),
                vel: Vec3::X * V0,
                radius: 2.0,
                ..Bird::default()
            },
        ];
        let d = decide(0, &birds, Z0);
        assert_eq!(d.neighbours, 1);
        assert!(
            d.force.y < 0.0,
            "pushed away from +y neighbour: {:?}",
            d.force
        );
    }

    #[test]
    fn settled_flock_neighbour_count_is_near_n_c() {
        let mut flock = Flock::scatter(2000, 1);
        for _ in 0..200 {
            flock.step();
        }
        let mean =
            flock.birds.iter().map(|b| b.neighbours as f32).sum::<f32>() / flock.birds.len() as f32;
        assert!((mean - N_C).abs() < 2.0, "mean neighbours {mean}");
    }
}
