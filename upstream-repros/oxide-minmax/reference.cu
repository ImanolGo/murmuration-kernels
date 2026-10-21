// The same operation in CUDA C++, for comparison: `nvcc -ptx -arch=sm_89 reference.cu`
__global__ void minmax(const float* a, const float* b, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = fmaxf(a[i], b[i]) + fminf(a[i], b[i]);
}
