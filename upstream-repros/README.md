# Upstream reproductions

Small programs that isolate things found while building the backends in this repository, against NVIDIA/cuda-rust
at `e027ac324f663e8408c13c116d7dddd58d5e8daf` (2026-10-20, still the tip of `main` on 2026-10-21). None needs a GPU;
they only need the CUDA 13.2+ headers (`CUDA_TOOLKIT_PATH`) and, for the oxide ones, `cargo-oxide` and its pinned nightly.
What was posted, and where, is listed in [POSTED.md](POSTED.md).

| Directory | Shows | Upstream status (checked 2026-10-21) |
|---|---|---|
| `cutile-constant-false` | `constant(false, shape)` lowers to `constant <i1: 1>` | Already reported: #1469; fix in open PR #1470 (checked: it fixes this program; a confirmation was posted) |
| `oxide-minmax` | float min/max compile to `setp` + `selp`, never `min.f32`/`max.f32` | Reported: #1594 |
| `oxide-to-int-unchecked` | `f32::to_int_unchecked` is rejected by the backend | Reported: #1595 |
| `cutile-dsl-friction` | generic device function from a generic kernel keeps `{ [B] }`; pointer-tile method chain fails to infer; a reference snippet fails rustc | Generic helper: already #1378 (comment added). Pointer-tile chain: #1596. Doc snippet: #1597 |

Run each with `cargo run` (cuTile ones) or `CUDA_HOME=/opt/cuda cargo oxide inspect|build --arch sm_89` (oxide ones);
each directory's source file explains what to expect. The `cutile-*` programs exit with status 1 when they show the problem.
