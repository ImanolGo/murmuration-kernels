#!/usr/bin/env bash
# CPU oracle vs the CUDA baseline on the same start. Needs a GPU.
#   scripts/parity.sh [birds]      (default 2000; the CPU side is brute force)
#   BACKEND=oxide scripts/parity.sh   uses oxide/target/release/flock-oxide, which
#                                     must be built first (see README, "Run it")
#   BACKEND=wgpu  scripts/parity.sh   uses target/release/flock-wgpu
# Pointwise agreement is required at 1, 10 and 30 steps (tolerance 1e-3 on position
# and velocity, no neighbour-count mismatch). At 100 steps up to 2 of 2000 birds may
# exceed it (an isolated threshold flip, docs/SPEC.md) if none exceeds 1e-2. At 1000
# steps only the aggregate statistics are compared.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-2000}
R=target/release/flock-ref
C=cuda/build/flock_cuda
[ "${BACKEND:-cuda}" = oxide ] && C=oxide/target/release/flock-oxide
[ "${BACKEND:-cuda}" = wgpu ] && C=target/release/flock-wgpu
mkdir -p data
cargo build --release -q
[ "${BACKEND:-cuda}" = cuda ] && make -s -C cuda
$R gen --birds "$N" --seed 1 --out "data/p$N.flk"
status=0
for K in 1 10 30 100 1000; do
  $R step --in "data/p$N.flk" --steps "$K" --out "data/p$N-cpu-$K.flk"
  ${GPU_PREFIX:-prime-run} $C --in "data/p$N.flk" --steps "$K" --out "data/p$N-gpu-$K.flk" > /dev/null
  echo "== $N birds, $K steps"
  if [ "$K" -le 30 ]; then
    $R diff "data/p$N-cpu-$K.flk" "data/p$N-gpu-$K.flk" || status=1
  elif [ "$K" -le 100 ]; then
    $R diff "data/p$N-cpu-$K.flk" "data/p$N-gpu-$K.flk" --allow-outliers 2 --outlier-max 1e-2 || status=1
  else
    $R statdiff "data/p$N-cpu-$K.flk" "data/p$N-gpu-$K.flk" || status=1
  fi
done
exit $status
