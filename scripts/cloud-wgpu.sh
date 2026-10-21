#!/usr/bin/env bash
# The wgpu backend on a rented GPU, with a CUDA C++ run in the same session as the reference.
# Needs scripts/remote-setup.sh first (CUDA toolkit, Rust, and a working Vulkan ICD for the NVIDIA GPU).
#   bash scripts/cloud-wgpu.sh        about 10 minutes
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=${OUT:-$HOME/results-wgpu}; mkdir -p "$OUT"
CUDA_DIR=${CUDA_DIR:-/usr/local/cuda-13.3}
export PATH="$CUDA_DIR/bin:$HOME/.cargo/bin:$PATH" CUDA_TOOLKIT_PATH="$CUDA_DIR" CUDA_HOME="$CUDA_DIR"
. "$HOME/.cargo/env" 2>/dev/null || true
ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
RUNS=${RUNS:-3}
status="$OUT/status.txt"; : > "$status"
step() { local name=$1; shift; local t0=$SECONDS
  if "$@" > "$OUT/$name.log" 2>&1; then echo "ok      $name ($((SECONDS - t0)) s)" | tee -a "$status"
  else echo "FAILED  $name ($((SECONDS - t0)) s)  see $name.log" | tee -a "$status"; fi; }

step vulkan-summary vulkaninfo --summary
step build-ref cargo build --release -p flock-ref
step build-wgpu cargo build --release -p flock-wgpu
step build-cuda make -C cuda ARCH=$ARCH
REF=$ROOT/target/release/flock-ref; WGPU=$ROOT/target/release/flock-wgpu; CUDA=$ROOT/cuda/build/flock_cuda
mkdir -p data
for n in 2000 100000 1000000; do $REF gen --birds $n --seed 1 --out data/s$n.flk; done

for K in 1 10 30 100 1000; do
  $REF step --in data/s2000.flk --steps $K --out data/cpu-$K.flk
  step wgpu-run-$K "$WGPU" --in data/s2000.flk --steps $K --out data/wgpu-$K.flk
  if [ "$K" -le 30 ]; then step wgpu-parity-$K $REF diff data/cpu-$K.flk data/wgpu-$K.flk
  elif [ "$K" -le 100 ]; then step wgpu-parity-$K $REF diff data/cpu-$K.flk data/wgpu-$K.flk --allow-outliers 2 --outlier-max 1e-2
  else step wgpu-parity-$K $REF statdiff data/cpu-$K.flk data/wgpu-$K.flk; fi
done
for n in 100000 1000000; do for r in $(seq "$RUNS"); do
  step bench-cuda-$n-$r "$CUDA" --in data/s$n.flk --steps 40 --warmup 20
  step bench-wgpu-$n-$r "$WGPU" --in data/s$n.flk --steps 40 --warmup 20
done; done
tar -C "$OUT/.." -czf "$OUT.tar.gz" "$(basename "$OUT")" && echo "== bundle: $OUT.tar.gz"
cat "$status"
