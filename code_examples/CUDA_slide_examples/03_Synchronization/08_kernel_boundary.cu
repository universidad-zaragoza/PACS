// Slide 21 "The kernel boundary is the global barrier" (03_Synchronization).
// phase2 reads elements that were written by OTHER blocks of phase1. Inside one kernel that would be a race;
// as two kernels in the same stream it is correct, because phase2 starts after phase1 has finished.
// Related with the BUG2 of previous file, the kernel boundary, is the way of providing a memory barrier
// even for threads in different blocks.
#include "common.h"

__global__ void phase1(int *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = 2 * i;
}

__global__ void phase2(const int *a, int *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = a[n - 1 - i];        // written by a different block (usually)
}

int main() {
  const int n = 1 << 20, block = 256, grid = (n + block - 1) / block;
  int *a, *b;
  CUDA_CHECK(cudaMalloc(&a, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&b, n * sizeof(int)));

  phase1<<<grid, block>>>(a, n);          // same (default) stream: phase2 waits for the whole of phase1
  phase2<<<grid, block>>>(a, b, n);
  CUDA_CHECK(cudaGetLastError());

  int *h = (int *)malloc(n * sizeof(int));
  CUDA_CHECK(cudaMemcpy(h, b, n * sizeof(int), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < n; ++i) ok = ok && h[i] == 2 * (n - 1 - i);
  free(h);  cudaFree(a);  cudaFree(b);
  return report("kernel_boundary", ok);
}
