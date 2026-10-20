//! Bird state, the benchmark scatter and the `.flk` exchange file.
//!
//! The same file is read and written by the CUDA backend, so every backend
//! starts from identical bytes. Layout (little endian, 4-byte words):
//! `"FLK1"`, `n`, `words_per_bird` (= [`WORDS`]), `time`, then `n` records of
//! [`WORDS`] words: pos xyz, vel xyz, roll, spin, radius, steering xyz,
//! next_decision, neighbour count (u32), centrality.

use std::io::{self, Read, Write};

use glam::Vec3;

use crate::params::*;

/// Words per bird record in the `.flk` file.
pub const WORDS: usize = 15;

/// One bird's dynamic state. Mirrors the fields the core step uses from
/// `murmur-ref`'s `Bird`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bird {
    /// Position [m].
    pub pos: Vec3,
    /// Velocity [m/s].
    pub vel: Vec3,
    /// Bank angle [rad].
    pub roll: f32,
    /// Bank rate [rad/s].
    pub spin: f32,
    /// Adaptive interaction radius [m].
    pub radius: f32,
    /// Held steering force [N].
    pub steering: Vec3,
    /// Simulated time of the next decision [s].
    pub next_decision: f32,
    /// Neighbour count from the last decision.
    pub neighbours: u32,
    /// Centrality from the last decision.
    pub centrality: f32,
}

/// A whole flock plus the world clock.
#[derive(Clone, Debug, Default)]
pub struct Flock {
    /// Simulated time [s].
    pub time: f32,
    /// The birds.
    pub birds: Vec<Bird>,
}

/// Birds per cubic metre in the benchmark scatter (as `murmur-gpu`'s `--bench`).
pub const DENSITY: f32 = 0.5;
/// Starting perception radius in the benchmark scatter [m].
pub const START_RADIUS: f32 = 1.5;

/// SplitMix64: a tiny, portable generator. The scatter is only a starting
/// condition, so it does not need to match `murmur-ref`'s ChaCha stream.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

impl Flock {
    /// A settled-like flock: `n` birds scattered uniformly in a cube of
    /// density [`DENSITY`], flying along +x at cruise speed. Same protocol as
    /// `murmur-gpu`'s `--bench` setup, so timings are comparable in kind.
    pub fn scatter(n: usize, seed: u64) -> Self {
        let side = (n as f32 / DENSITY).cbrt();
        let centre = Vec3::new(0.0, 0.0, Z0);
        let mut rng = SplitMix(seed);
        let birds = (0..n)
            .map(|_| {
                let offset = Vec3::new(
                    (rng.unit() - 0.5) * side,
                    (rng.unit() - 0.5) * side,
                    (rng.unit() - 0.5) * side,
                );
                let jitter = Vec3::new(rng.unit() - 0.5, rng.unit() - 0.5, rng.unit() - 0.5) * 0.05;
                Bird {
                    pos: centre + offset,
                    vel: (Vec3::X + jitter).normalize() * V0,
                    radius: START_RADIUS,
                    next_decision: rng.unit() * DU,
                    ..Bird::default()
                }
            })
            .collect();
        Self { time: 0.0, birds }
    }

    /// Writes the `.flk` file.
    pub fn write(&self, mut w: impl Write) -> io::Result<()> {
        w.write_all(b"FLK1")?;
        w.write_all(&(self.birds.len() as u32).to_le_bytes())?;
        w.write_all(&(WORDS as u32).to_le_bytes())?;
        w.write_all(&self.time.to_le_bytes())?;
        for b in &self.birds {
            let f = [
                b.pos.x,
                b.pos.y,
                b.pos.z,
                b.vel.x,
                b.vel.y,
                b.vel.z,
                b.roll,
                b.spin,
                b.radius,
                b.steering.x,
                b.steering.y,
                b.steering.z,
                b.next_decision,
            ];
            for v in f {
                w.write_all(&v.to_le_bytes())?;
            }
            w.write_all(&b.neighbours.to_le_bytes())?;
            w.write_all(&b.centrality.to_le_bytes())?;
        }
        Ok(())
    }

    /// Reads a `.flk` file.
    pub fn read(mut r: impl Read) -> io::Result<Self> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        let mut head = [0u8; 16];
        r.read_exact(&mut head)?;
        if &head[0..4] != b"FLK1" {
            return Err(bad("not a FLK1 file"));
        }
        let word = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
        let n = word(&head[4..8]) as usize;
        if word(&head[8..12]) as usize != WORDS {
            return Err(bad("unexpected words per bird"));
        }
        let time = f32::from_bits(word(&head[12..16]));
        let mut buf = vec![0u8; n * WORDS * 4];
        r.read_exact(&mut buf)?;
        let birds = buf
            .as_chunks::<{ WORDS * 4 }>()
            .0
            .iter()
            .map(|rec| {
                let f = |k: usize| f32::from_bits(word(&rec[k * 4..k * 4 + 4]));
                Bird {
                    pos: Vec3::new(f(0), f(1), f(2)),
                    vel: Vec3::new(f(3), f(4), f(5)),
                    roll: f(6),
                    spin: f(7),
                    radius: f(8),
                    steering: Vec3::new(f(9), f(10), f(11)),
                    next_decision: f(12),
                    neighbours: word(&rec[13 * 4..14 * 4]),
                    centrality: f(14),
                }
            })
            .collect();
        Ok(Self { time, birds })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_round_trips() {
        let mut flock = Flock::scatter(50, 7);
        flock.time = 1.25;
        flock.birds[3].neighbours = 9;
        let mut bytes = Vec::new();
        flock.write(&mut bytes).unwrap();
        let back = Flock::read(bytes.as_slice()).unwrap();
        assert_eq!(back.time, flock.time);
        assert_eq!(back.birds, flock.birds);
    }

    #[test]
    fn scatter_is_repeatable_and_at_density() {
        let a = Flock::scatter(1000, 1);
        let b = Flock::scatter(1000, 1);
        assert_eq!(a.birds, b.birds);
        let side = (1000.0f32 / DENSITY).cbrt();
        assert!(a.birds.iter().all(|b| (b.pos.x).abs() <= side / 2.0));
        assert!(a.birds.iter().all(|b| (b.vel.length() - V0).abs() < 1e-3));
    }
}
