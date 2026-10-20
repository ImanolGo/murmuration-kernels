# AGENTS.md: rules for working in flock-four-ways

Read `docs/SPEC.md` first. This repo compares GPU programming models on one
fixed workload (a reduced starling-flock step).

- **The oracle is the specification.** `crates/flock-ref` defines behaviour. A
  backend mirrors its functions and order of operations. Never tune a backend
  to look better without the same change in the oracle and a note in SPEC.md.
- **Honest numbers.** Report performance only from GPU events or timestamp
  queries, with GPU name, driver, CUDA version, bird count and the number of
  runs. Say when runs disagree. Do not compare against numbers from another
  repository unless the workload is identical.
- **Do not loosen a tolerance to make a check pass.** If a check fails, explain
  why in `docs/progress.md` and change the criterion in `docs/SPEC.md` only with
  a reason.
- **Optimisations are separate commits**, each with a before and after
  measurement in `docs/progress.md`.
- **Do not claim a backend works on a GPU it has not run on.** cuTile Rust
  needs sm_80 or newer.
- Rust 1.95+, `cargo fmt`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test` before closing a step. CUDA: `scripts/parity.sh` must pass.
- Public repo: nothing private, confidential or taken from an employer.
