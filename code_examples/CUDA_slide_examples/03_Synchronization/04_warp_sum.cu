// Slides 10-11 "Warp Shuffle and Vote" and "Walk-through: Shuffle Reduction" (03_Synchronization).
// warp_sum is the warpSum of the slide: five __shfl_down_sync steps, the total ends up in lane 0.
// block_sum then combines the warp sums through shared memory (not on the slides: the next step up).
#include "common.h"

// After five steps lane 0 holds the sum of the 32 values of the warp.
__device__ float warp_sum(float v) {
  for (int offset = 16; offset > 0; offset /= 2)
    v += __shfl_down_sync(FULL_MASK, v, offset);
  return v;
}

// blockDim.x must be a multiple of 32 and at most 1024. The result is valid in thread 0.
__device__ float block_sum(float v) {
  __shared__ float partial[32];                          // one slot per warp. Note that at most, a block can have 32 warps of 32 lanes each 32x32=1024
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;

  v = warp_sum(v);                                       // 1. every warp reduces its own 32 values
  if (lane == 0) partial[warp] = v;                      // 2. lane 0 publishes the warp sum
  __syncthreads();

  v = (threadIdx.x < blockDim.x / 32) ? partial[lane] : 0.0f;   // 3. the first warp loads the partial sums
  if (warp == 0) v = warp_sum(v);                        // 4. and reduces them
  return v;
}

__global__ void sum_kernel(const float *x, int n, float *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (i < n) ? x[i] : 0.0f;                       // lanes past the end contribute the identity
  v = block_sum(v);
  if (threadIdx.x == 0) 
      atomicAdd(out, v);               // one atomic per block
}

int main() {
  const int n = 1000;                                    // not a multiple of any block size on purpose
  std::vector<float> h(n);
  double expected = 0.0;

  for (int i = 0; i < n; ++i) { 
      h[i] = (float)(i % 7); 
      expected += h[i]; 
  }   // small integers: exact in float

  float *d_x, *d_out;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  bool ok = true;
  const int blocks[3] = {32, 128, 256};
  GpuTimer t; 
  for (int b : blocks) {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
   
    t.start();

    sum_kernel<<<(n + b - 1) / b, b>>>(d_x, n, d_out);
    CUDA_CHECK(cudaGetLastError());
 
    float ms = t.stop();

    float got = 0.0f;
    CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("block size %3d: sum = %.0f (expected %.0f) Elapsed Time: %8.3f ms\n", b, got, expected, ms);
    ok = ok && got == (float)expected;
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_out));
  return report("warp_and_block_sum", ok);
}
