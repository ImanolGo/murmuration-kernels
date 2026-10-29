# What was posted upstream, and where

All posted on 2026-10-21 to NVIDIA/cuda-rust, from small programs in this directory that need no GPU. The texts follow the
project's bug-report template and carry an "AI contribution" line, as the project's own reports do.

| Where | What | Reproduction | Status (2026-10-29) |
|---|---|---|---|
| [PR #1470](https://github.com/NVIDIA/cuda-rust/pull/1470) (comment) | Independent check that the fix makes `constant(false, ..)` lower to `i1: 0`; downstream effect | `cutile-constant-false` | open, not merged |
| [Issue #1378](https://github.com/NVIDIA/cuda-rust/issues/1378) (comment) | A simpler variant of the generic-helper bug (generic kernel calling a generic helper) | `cutile-dsl-friction`, `generic_entry` | open |
| [Issue #1594](https://github.com/NVIDIA/cuda-rust/issues/1594) | cuda-oxide: float min/max lower to `setp`+`selp`, not `min.f32`/`max.f32` | `oxide-minmax` | fix in [PR #1668](https://github.com/NVIDIA/cuda-rust/pull/1668) by another contributor; re-profiled and approved by me, awaiting a maintainer |
| [Issue #1595](https://github.com/NVIDIA/cuda-rust/issues/1595) | cuda-oxide: `f32::to_int_unchecked` fails with "invalid input program" | `oxide-to-int-unchecked` | open; a contributor has offered to fix it |
| [Issue #1596](https://github.com/NVIDIA/cuda-rust/issues/1596) | cuTile: a `PointerTile` method chain fails to infer its type | `cutile-dsl-friction`, `ptr_chain` | open; a contributor has offered to fix it |
| [Issue #1597](https://github.com/NVIDIA/cuda-rust/issues/1597) | The `PointerTile` examples in `dsl-api.md` do not type-check | `cutile-dsl-friction`, `--features docs-snippet` | open; fix in PR #1599 |
| [PR #1599](https://github.com/NVIDIA/cuda-rust/pull/1599) | The documentation fix for #1597 (two examples in `dsl-api.md`), from a fork; the corrected form is the `docs_fixed` kernel | `cutile-dsl-friction`, `docs_fixed` | open, awaiting maintainer review and CI vetting |

Already reported before this project found them: `constant(false)` is [#1469](https://github.com/NVIDIA/cuda-rust/issues/1469)
(fix in #1470), and the generic-helper problem is #1378. Two things were checked and ruled out as duplicates of other items:
applying the diff of NVlabs/cutile-rs#226 (tracked in #1424) does not fix #1596.
