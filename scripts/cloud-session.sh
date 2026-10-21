#!/usr/bin/env bash
# One scripted session on a rented GPU VM: build every GPU backend, check and compare
# cuTile Rust against the CPU oracle, benchmark CUDA C++ / cuda-oxide / cuTile Rust with
# one protocol, profile each `decide` kernel with Nsight Compute, bundle the results.
#
# Run after scripts/remote-setup.sh, from the repo root copied to the VM:
#   bash scripts/cloud-session.sh            # about 25-40 minutes; every step logs to ~/results
# Steps never abort the session: a failure is recorded and the next step runs.
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=${OUT:-$HOME/results}; mkdir -p "$OUT"
CUDA_DIR=${CUDA_DIR:-/usr/local/cuda-13.3}
export PATH="$CUDA_DIR/bin:$HOME/.cargo/bin:$PATH" CUDA_TOOLKIT_PATH="$CUDA_DIR" CUDA_HOME="$CUDA_DIR"
. "$HOME/.cargo/env" 2>/dev/null || true
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
ARCH=sm_$cc
RUNS=${RUNS:-3}
status="$OUT/status.txt"; : > "$status"

step() {  # step NAME CMD...   -> log in $OUT/NAME.log, status line, never aborts
  local name=$1; shift
  local t0=$SECONDS
  if "$@" > "$OUT/$name.log" 2>&1; then echo "ok      $name ($((SECONDS - t0)) s)" | tee -a "$status"
  else echo "FAILED  $name ($((SECONDS - t0)) s)  see $name.log" | tee -a "$status"; fi
}

echo "== session on $ARCH, results in $OUT"
step gpu-info bash -c 'nvidia-smi -q; nvcc --version; tileiras --version; ncu --version; rustc --version'
step build-ref cargo build --release -p flock-ref
REF=$ROOT/target/release/flock-ref
step build-cuda make -C cuda ARCH=$ARCH
step build-cutile bash -c 'cd cutile && cargo build --release'
step install-cargo-oxide cargo install --git https://github.com/NVIDIA/cuda-rust.git --rev e027ac324f663e8408c13c116d7dddd58d5e8daf cargo-oxide
step build-oxide bash -c "cd oxide && cargo oxide build --arch $ARCH"

CUDA_BIN=$ROOT/cuda/build/flock_cuda
OXIDE_BIN=$ROOT/oxide/target/release/flock-oxide
CUTILE_BIN=$ROOT/cutile/target/release/flock-cutile
mkdir -p data
for n in 2000 100000 1000000; do $REF gen --birds $n --seed 1 --out data/s$n.flk; done

# 1. cuTile correctness: the grid first (names the kernel at fault), then the oracle.
step cutile-grid-check "$CUTILE_BIN" --in data/s2000.flk --steps 3 --check
for K in 1 10 30 100 1000; do
  $REF step --in data/s2000.flk --steps $K --out data/cpu-$K.flk
  step cutile-run-$K "$CUTILE_BIN" --in data/s2000.flk --steps $K --out data/cutile-$K.flk
  if [ "$K" -le 30 ]; then
    step cutile-parity-$K $REF diff data/cpu-$K.flk data/cutile-$K.flk
  elif [ "$K" -le 100 ]; then
    step cutile-parity-$K $REF diff data/cpu-$K.flk data/cutile-$K.flk --allow-outliers 2 --outlier-max 1e-2
  else
    step cutile-parity-$K $REF statdiff data/cpu-$K.flk data/cutile-$K.flk
  fi
done
# the other backends on this GPU, for the same comparison
for K in 1 100; do
  step cuda-parity-$K bash -c "$CUDA_BIN --in data/s2000.flk --steps $K --out data/cuda-$K.flk >/dev/null && $REF diff data/cpu-$K.flk data/cuda-$K.flk"
  step oxide-parity-$K bash -c "$OXIDE_BIN --in data/s2000.flk --steps $K --out data/oxide-$K.flk >/dev/null && $REF diff data/cpu-$K.flk data/oxide-$K.flk"
done

# 2. benchmark: same protocol as everywhere else (20 warm-up + 20 timed steps)
for n in 100000 1000000; do
  for r in $(seq "$RUNS"); do
    step bench-cuda-$n-$r   "$CUDA_BIN"   --in data/s$n.flk --steps 40 --warmup 20
    step bench-oxide-$n-$r  "$OXIDE_BIN"  --in data/s$n.flk --steps 40 --warmup 20
    step bench-cutile-$n-$r "$CUTILE_BIN" --in data/s$n.flk --steps 40 --warmup 20
  done
done

# 3. Nsight Compute: one warmed-up decide launch per backend at 100k birds
for b in cuda oxide cutile; do
  bin_var=${b^^}_BIN; bin=${!bin_var}
  step ncu-$b ncu --kernel-name regex:decide --launch-skip 22 --launch-count 1 --set full -f \
      -o "$OUT/ncu-$b-100k" "$bin" --in data/s100000.flk --steps 24
done
step ncu-summary python3 -I - "$OUT" <<'PY'
import csv, subprocess, sys
out = sys.argv[1]
metrics = ",".join([
    "gpu__time_duration.sum", "launch__block_size", "launch__registers_per_thread",
    "launch__occupancy_limit_registers", "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active", "smsp__inst_executed.sum",
    "smsp__issue_active.avg.pct_of_peak_sustained_active", "smsp__warps_eligible.avg.per_cycle_active",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
])
cols = {}
for b in ("cuda", "oxide", "cutile"):
    import glob  # the extension is .ncu-rep or .ncu-repz depending on the ncu version
    path = (glob.glob(f"{out}/ncu-{b}-100k.ncu-rep*") or [f"{out}/ncu-{b}-100k.ncu-rep"])[0]
    r = subprocess.run(["ncu", "-i", path, "--page", "raw", "--csv", "--metrics", metrics],
                       capture_output=True, text=True)
    rows = [x for x in csv.reader(r.stdout.splitlines()) if len(x) > 20]
    cols[b] = dict(zip(rows[0], rows[2])) if len(rows) >= 3 else {}
print(f"{'metric':<62}" + "".join(f"{b:>16}" for b in cols))
for m in metrics.split(","):
    print(f"{m:<62}" + "".join(f"{cols[b].get(m, 'n/a'):>14}" for b in cols))
PY

tar -C "$OUT/.." -czf "$OUT.tar.gz" "$(basename "$OUT")" && echo "== bundle: $OUT.tar.gz"
cat "$status"
