// Slides 46-49 "Example: Sum Reduction", "Block Summation", "Combining the Blocks" and
// "Single Pass: The Last Block finishes" (03_Synchronization). Exercise "Reduce it" (slide 52).
// The same block reduction is combined three ways:
//   A) a second kernel (the kernel boundary is the global barrier)
//   B) atomicAdd of each block total
//   C) one kernel: __threadfence() + a ticket counter, the last block adds the partial results
#include "common.h"

__device__ int block_reduce(int v, int *s) {            // all threads of the block must call it
  s[threadIdx.x] = v;
  __syncthreads();                                       // all values are stored
  for (int step = blockDim.x / 2; step > 0; step /= 2) {
    if (threadIdx.x < step) s[threadIdx.x] += s[threadIdx.x + step];
    __syncthreads();                                     // this level is complete
  }
  int total = s[0];
  __syncthreads();                                       // nobody still reads s[0] when s is reused
  return total;
}

__device__ int grid_stride_sum(const int *in, int n) {
  int sum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    sum += in[i];
  return sum;
}

// A) one partial result per block; launched twice (second time with one block)
__global__ void reduce_blocks(const int *in, int n, int *out) {
  extern __shared__ int s[];
  int total = block_reduce(grid_stride_sum(in, n), s);
  if (threadIdx.x == 0) out[blockIdx.x] = total;
}

// B) every block adds its total to the result
__global__ void reduce_atomic(const int *in, int n, int *out) {
  extern __shared__ int s[];
  int total = block_reduce(grid_stride_sum(in, n), s);
  if (threadIdx.x == 0) atomicAdd(out, total);
}

// C) single pass: the last block to finish adds up the partial results
__device__ unsigned int tickets = 0;

__global__ void reduce_single_pass(const int *in, int n, volatile int *partial, int *out) {
  extern __shared__ int s[];
  __shared__ bool last;
  int total = block_reduce(grid_stride_sum(in, n), s);

  if (threadIdx.x == 0) {
    partial[blockIdx.x] = total;
    __threadfence();                                     // the partial result is visible before the ticket is taken
    last = (atomicInc(&tickets, gridDim.x - 1) == gridDim.x - 1);
  }
  __syncthreads();                                       // the whole block learns whether it is the last one

  if (last) {
    int v = 0;
    for (int b = threadIdx.x; b < (int)gridDim.x; b += blockDim.x) v += partial[b];
    int all = block_reduce(v, s);
    if (threadIdx.x == 0) *out = all;
  }
}

int main() {
  const int n = (1 << 20) + 77, block = 256, grid = 64;
  const size_t smem = block * sizeof(int);
  int *h = (int *)malloc(n * sizeof(int));
  long long want = 0;
  for (int i = 0; i < n; ++i) { h[i] = i % 10; want += h[i]; }

  int *d_in, *d_part, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_part, grid * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h, n * sizeof(int), cudaMemcpyHostToDevice));
  int got = 0;
  bool ok = true;

  // A) two kernels
  reduce_blocks<<<grid, block, smem>>>(d_in, n, d_part);
  reduce_blocks<<<1, block, smem>>>(d_part, grid, d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof got, cudaMemcpyDeviceToHost));
  printf("A) two kernels   : %d (expected %lld)\n", got, want);  ok = ok && got == want;

  // B) atomicAdd
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
  reduce_atomic<<<grid, block, smem>>>(d_in, n, d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof got, cudaMemcpyDeviceToHost));
  printf("B) atomicAdd     : %d (expected %lld)\n", got, want);  ok = ok && got == want;

  // C) single pass (run twice: the ticket counter resets itself)
  for (int run = 0; run < 2; ++run) {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
    reduce_single_pass<<<grid, block, smem>>>(d_in, n, d_part, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof got, cudaMemcpyDeviceToHost));
    printf("C) single pass #%d : %d (expected %lld)\n", run + 1, got, want);  ok = ok && got == want;
  }
  free(h);  cudaFree(d_in);  cudaFree(d_part);  cudaFree(d_out);
  return report("reduction_three_ways", ok);
}
