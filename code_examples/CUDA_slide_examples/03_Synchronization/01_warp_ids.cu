// Slide 6 "SIMT Execution and Warps" (03_Synchronization).
// Threads are numbered inside the block with x varying fastest; consecutive groups of 32 form a warp.
// A 16 x 16 block holds 8 warps; a 100-thread block still occupies 4 whole warps.
#include "common.h"

__global__ void ids(int *warp_of, int *lane_of) {
  int tid = threadIdx.x + threadIdx.y * blockDim.x;      // linear index in the block
  warp_of[tid] = tid / warpSize;
  lane_of[tid] = tid % warpSize;
}

int main() {
  const dim3 block(16, 16);                              // 256 threads
  const int n = block.x * block.y;
  int *d_warp, *d_lane;
  CUDA_CHECK(cudaMalloc(&d_warp, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_lane, n * sizeof(int)));

  ids<<<1, block>>>(d_warp, d_lane);
  CUDA_CHECK(cudaGetLastError());

  std::vector<int> warp(n), lane(n);
  CUDA_CHECK(cudaMemcpy(warp.data(), d_warp, n * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(lane.data(), d_lane, n * sizeof(int), cudaMemcpyDeviceToHost));

  // A 16 x 16 block: one warp covers two rows of the block.
  bool ok = true;
  for (int y = 0; y < 16; ++y){
    for (int x = 0; x < 16; ++x) {
      int tid = y * 16 + x;
      ok = ok && warp[tid] == y / 2 && lane[tid] == (y % 2) * 16 + x;
      printf(" (%d %2d)", warp[tid], lane[tid]);
    }
    printf("\n");
  }
  printf("16 x 16 block: %d warps, warp w holds rows 2w and 2w+1\n", n / 32);

  // A 1D block of 100 threads still occupies whole warps.
  const int threads = 100;
  int warps = (threads + 31) / 32;
  int last_active = threads - (warps - 1) * 32;
  printf("%d threads: %d warps, the last one has %d active lanes (%.1f%% of the lanes are used)\n",
         threads, warps, last_active, 100.0 * threads / (warps * 32));
  ok = ok && warps == 4 && last_active == 4;

  CUDA_CHECK(cudaFree(d_warp));
  CUDA_CHECK(cudaFree(d_lane));
  return report("warp_ids", ok);
}
