#!/usr/bin/env bash
# Prepares a rented Ubuntu 22.04 GPU VM (root) for the cuTile Rust backend and Nsight
# Compute profiling. Tested 2026-10-21 on a Vast.ai "Ubuntu 22.04 VM" (RTX 4080 SUPER,
# driver 580.95.05). Takes about 8 minutes, most of it the CUDA toolkit download.
#   scp this script to the VM, then: bash remote-setup.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
CUDA_PKG=${CUDA_PKG:-cuda-toolkit-13-3}     # 13.2 is enough for Ampere/Ada; 13.3 is recommended
CUDA_DIR=${CUDA_DIR:-/usr/local/cuda-13.3}

echo "== GPU"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
[ "$cc" -ge 80 ] || { echo "compute capability < 8.0: cuTile Rust cannot run here"; exit 1; }

echo "== CUDA toolkit ($CUDA_PKG)"
if [ ! -x "$CUDA_DIR/bin/nvcc" ]; then
  cd /root
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
  # NVIDIA's CDN is sometimes mid-sync ("File has unexpected size"); retry instead of aborting
  for attempt in 1 2 3 4 5 6; do apt-get update -qq && break; echo "apt-get update failed (attempt $attempt), retrying"; sleep 20; done
  apt-get install -y "$CUDA_PKG" clang libclang-dev build-essential pkg-config libssl-dev rsync
fi
export PATH="$CUDA_DIR/bin:$PATH" CUDA_TOOLKIT_PATH="$CUDA_DIR" CUDA_HOME="$CUDA_DIR"
nvcc --version | tail -2
which ncu nsys tileiras

echo "== Rust"
command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; }
. /root/.cargo/env

echo "== Nsight Compute smoke test (a kernel, profiled; fails with ERR_NVGPUCTRPERM if counters are blocked)"
cat > /root/smoke.cu <<'CU'
#include <cstdio>
__global__ void axpy(float a, const float* x, float* y, int n){ int i = blockIdx.x*blockDim.x + threadIdx.x; if (i<n) y[i] = a*x[i] + y[i]; }
int main(){ const int n = 1<<22; float *x,*y; cudaMallocManaged(&x,n*4); cudaMallocManaged(&y,n*4);
  for(int i=0;i<n;i++){x[i]=1.f;y[i]=2.f;}
  for(int k=0;k<3;k++) axpy<<<(n+255)/256,256>>>(3.f,x,y,n);
  cudaDeviceSynchronize(); printf("y[0]=%f (expected 11)\n", y[0]); return 0; }
CU
nvcc -arch=native -o /root/smoke /root/smoke.cu && /root/smoke
ncu --kernel-name axpy --launch-skip 1 --launch-count 1 \
    --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active /root/smoke 2>&1 \
  | grep -E "ERR|duration|warps_active"
echo "== Vulkan (only the wgpu backend needs it; failures here are not fatal)"
apt-get install -y libvulkan1 vulkan-tools > /root/vulkan-install.log 2>&1 || echo "   vulkan packages failed, see /root/vulkan-install.log"
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null | sed 's/^/   icd: /'
vulkaninfo --summary 2>&1 | grep -E "deviceName|driverName|driverInfo" | sed 's/^/   /' | head -4 || true

echo "== done. Next: copy the repo (git bundle or rsync), then build the backends."
echo "   export PATH=$CUDA_DIR/bin:\$PATH CUDA_TOOLKIT_PATH=$CUDA_DIR CUDA_HOME=$CUDA_DIR"
