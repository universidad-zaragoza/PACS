// "Combining Barriers and Atomics" and "Walk-through: Counting per Block" (02_MemorySpaces).
// Each block counts its threads in shared memory, then adds its total to the global counter with ONE atomic.
#include "common.h"

__global__ void kernel(int *counter) {
  __shared__ int local;

  if (threadIdx.x == 0)
    local = 0;

  __syncthreads();                       // 1: local = 0 is visible before anyone adds

  atomicAdd(&local, 1);                  // 2: no lost updates inside the block

  __syncthreads();                       // 3: every increment is done before thread 0 reads local

  if (threadIdx.x == 0)
    atomicAdd(counter, local);           // 4: one global atomic per block, not one per thread
}

int main() {
  const int blocks = 100, threads = 256;
  int *d, h = 0;
  CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(int)));
  kernel<<<blocks, threads>>>(d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost));
  printf("counter = %d (expected %d)\n", h, blocks * threads);
  cudaFree(d);
  return report("block_local_count", h == blocks * threads);
}
