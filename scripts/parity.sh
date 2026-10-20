#!/usr/bin/env bash
# CPU oracle vs the CUDA baseline on the same start. Needs a GPU.
#   scripts/parity.sh [birds]      (default 2000; the CPU side is brute force)
# Pointwise agreement is required at 1 and 100 steps (tolerance 1e-3 on position
# and velocity, no neighbour-count mismatch). At 1000 steps only the aggregate
# statistics are compared, because the flock amplifies rounding differences.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-2000}
R=target/release/flock-ref
C=cuda/build/flock_cuda
mkdir -p data
cargo build --release -q
make -s -C cuda
$R gen --birds "$N" --seed 1 --out "data/p$N.flk"
status=0
for K in 1 100 1000; do
  $R step --in "data/p$N.flk" --steps "$K" --out "data/p$N-cpu-$K.flk"
  ${GPU_PREFIX:-prime-run} $C --in "data/p$N.flk" --steps "$K" --out "data/p$N-gpu-$K.flk" > /dev/null
  echo "== $N birds, $K steps"
  if [ "$K" -le 100 ]; then
    $R diff "data/p$N-cpu-$K.flk" "data/p$N-gpu-$K.flk" || status=1
  else
    $R statdiff "data/p$N-cpu-$K.flk" "data/p$N-gpu-$K.flk" || status=1
  fi
done
exit $status
