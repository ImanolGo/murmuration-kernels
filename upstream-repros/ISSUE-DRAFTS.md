# Issue drafts for NVIDIA/cuda-rust

For review. Nothing here has been posted. CONTRIBUTING.md asks for an issue before a pull request, and the project's own
recent reports carry an "AI contribution" line, so each draft has one: edit it to say what is true when you post.
Every draft names a program in this directory that a maintainer can run without a GPU.

---

## 1. Comment on #1469 / #1470 (confirmation, not a new issue)

> Independent confirmation, plus a check of the fix. On `e027ac3` (2026-10-20, CUDA 13.3, `sm_89` target) a kernel
> containing `constant(false, shape![128])` and `constant(true, shape![128])` emits `constant <i1: 1>` for both.
> I built the same program against the head of #1470 (`d23a842`) and it emits `i1: 0` then `i1: 1`, as expected.
> Compile-only reproduction (no GPU): <link to upstream-repros/cutile-constant-false>.
>
> Downstream effect, in case it helps prioritise: I used `select(a, b, constant(false, ..))` as a bool `and`. It silently
> became `a -> b`, which made a neighbour search count each element as its own neighbour. Every one of 191 elements had
> exactly one extra neighbour, which is how I found it. I now build `false` from an integer comparison.
>
> **AI contribution:** drafted with Claude Code and checked by running the reproduction on both revisions.

---

## 2. cuda-oxide: float min/max lower to `setp` + `selp` instead of `min.f32` / `max.f32`

`f32::max`, `f32::min`, `libm::fmaxf` and `libm::fminf` in a `#[kernel]` all compile to a `setp` + `selp` pair. CUDA C++
compiles `fmaxf`/`fminf` to the PTX `max.f32`/`min.f32`.

Reproduction: `upstream-repros/oxide-minmax` (`cargo oxide inspect --arch sm_89`). For four min/max operations the PTX has
0 `max.f32`/`min.f32`, 6 `setp` and 8 `selp`; `reference.cu` (`nvcc -ptx -arch=sm_89`) has one `max.f32`, one `min.f32` and
no `setp`/`selp`. In a real neighbour-search kernel the cuda-oxide SASS has no `FMNMX` where the CUDA C++ SASS has 20, and
the kernel executes 77 percent more instructions overall (about a fifth of that from bounds checks; I did not attribute the rest).

Expected: the hardware instruction, as in CUDA C++. Environment: cuda-rust `e027ac3`, `nightly-2026-10-11`, CUDA 13.3 and 13.4.
I did not check whether NaN semantics are the reason (`max.f32` returns the non-NaN operand, like `maxnum`).

**AI contribution:** drafted with Claude Code; the PTX counts above were produced by running the reproduction.

---

## 3. cuda-oxide: `f32::to_int_unchecked` fails with "invalid input program"

`x[i].to_int_unchecked::<i32>()` inside a `#[kernel]` fails:
`Device codegen failed: PTX generation failed: Translation failed: ... core/src/convert/num.rs ... Compilation error: invalid input program.`
The checked cast `x[i] as i32` builds. Reproduction: `upstream-repros/oxide-to-int-unchecked`
(`cargo oxide build --arch sm_89` passes, `--features unchecked` fails).

Why it matters: `f32 as i32` saturates and maps NaN to 0, which adds instructions to every conversion (a spatial hash
does several per candidate), and `to_int_unchecked` is the standard way to skip that. cuda-rust `e027ac3`, CUDA 13.3.

**AI contribution:** drafted with Claude Code; the failing build was run as described.

---

## 4. cuTile Rust: a generic device function called from a generic kernel returns an unsubstituted type

```rust
fn keep<const B: i32>(x: Tile<f32, { [B] }>) -> Tile<f32, { [B] }> { /* ... */ }

#[cutile::entry()]
fn generic_entry<const B: i32>(out: &mut Tensor<f32, { [B] }>, a: &Tensor<f32, { [-1] }>) {
    let t: Tile<f32, { [B] }> = a.load_tile(shape![B], [program_id(0)]);
    let k: Tile<f32, { [B] }> = keep::<B>(t);
    out.store(max_tile(k, t));
}
```
With generics `["128"]` this fails at compile time: ``binary `Max` requires operands of the same type, but got `Tile < f32 , { [128] }
>` and `Tile < f32 , { [B] }>` ``. The same call from a non-generic kernel with the literal `128` compiles. Reproduction:
`upstream-repros/cutile-dsl-friction` (`generic_entry`). The result type of the inlined call keeps the callee's `B`, as if
the entry's generic value were not substituted into the return type. Compile-only API; cuda-rust `e027ac3`.

**AI contribution:** drafted with Claude Code; the failure was produced by the reproduction.

---

## 5. cuTile Rust: pointer-tile method chains fail to infer, and a reference snippet does not type-check

(a) `r0.reshape(shape![1]).broadcast(shape![128]).offset_tile(off)` on a rank-0 `PointerTile` fails with "return type is
missing a compiled tile type"; the same steps as separate annotated `let`s compile. (`ptr_chain` vs `ptr_steps`.)

(b) `cutile-book/reference/dsl-api.md` (PointerTile) shows `ptr.broadcast(shape![128]).offset_tile(offsets)` starting
from a scalar `PointerTile<_, { [] }>`. `PointerTile::broadcast` keeps the rank, so rustc rejects it
(``expected `Shape_0`, found `Shape_1<'_, 128>` ``). `cargo build --features docs-snippet` shows it. A `reshape` to rank 1
first is needed (`reshape_ptr(..)` works).

Reproduction: `upstream-repros/cutile-dsl-friction`. cuda-rust `e027ac3`.

**AI contribution:** drafted with Claude Code; both failures were produced by the reproduction.
