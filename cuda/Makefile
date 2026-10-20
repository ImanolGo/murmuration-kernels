# CUDA C++ baseline. ARCH defaults to the development GPU (GTX 1650 Max-Q, sm_75).
ARCH ?= sm_75
NVCC ?= nvcc
NVFLAGS ?= -O3 -std=c++20 -arch=$(ARCH) -lineinfo

build/flock_cuda: flock.cu params.cuh
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -o $@ flock.cu

clean:
	rm -rf build

.PHONY: clean
