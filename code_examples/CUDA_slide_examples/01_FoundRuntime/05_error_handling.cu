// Part 1, slide "Error Handling": CUDA_CHECK, launch errors, errors that arrive late.
#include "common.h"

__global__ void fill(int *x) { x[threadIdx.x] = 1; }

int main() {
  int *d;
  CUDA_CHECK(cudaMalloc(&d, 1024 * sizeof(int)));

  // A correct launch, checked the way the slide shows.
  fill<<<1, 256>>>(d);
  CUDA_CHECK(cudaGetLastError());        // launch errors
  CUDA_CHECK(cudaDeviceSynchronize());   // errors while the kernel ran
  //CUDA_CHECK prompts nothing in case of no errors
  fprintf(stdout, "launch with 256 threads: no errori \n");

  // A wrong launch: 2048 threads exceed the limit of 1024 per block.
  // The launch itself returns nothing; cudaGetLastError reports the problem.
  fill<<<1, 2048>>>(d);
  //WARNING! Without waiting for device synchronization we can only detect launchin errors, not execution errors
  //launch errors: numbero of blocks, number of threads per block, out of shared memory, no kernel image, ... 
  //execution error: illegal memory accesses, unaligned accesses, timeout, ...
  cudaError_t e = cudaGetLastError();
  printf("launch with 2048 threads: %s\n", cudaGetErrorString(e));

  // A runtime call that fails: asking for far more memory than any GPU has.
  void *huge = nullptr;
  // cudaMalloc returns an error status
  e = cudaMalloc(&huge, (size_t)1 << 60);
  printf("cudaMalloc of 1 EiB: %s\n", cudaGetErrorString(e));

  CUDA_CHECK(cudaFree(d));
  return 0;
}
