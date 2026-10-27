// Slides 7-8 "Warp divergence" and "Independent Thread Scheduling" (03_Synchronization).
// Since Volta (compute capability 7.0) every thread has its own program counter: the lanes of a warp are NOT in
// lock-step any more, and divergent paths may interleave. So:
//   - do not rely on lock-step execution,
//   - use __syncwarp() (and the _sync warp functions) to say which lanes must meet.
//
//   unsigned mask = __activemask();   // the lanes that execute THIS instruction right now
//   __syncwarp();                     // all 32 lanes meet here
//   __syncwarp(0x0000ffff);           // only lanes 0 to 15 meet
//
// Part 1: a warp-level reduction of 64 values in shared memory. Every read/write step is separated by __syncwarp()
//         instead of relying on lock-step (the old "warp-synchronous" code is a bug on Volta and newer).
// Part 2: __syncwarp(0x0000ffff): lanes 0..15 exchange data through shared memory while lanes 16..31 do other work.
// Part 3: __activemask() versus a mask that is computed. __activemask() only reports who happens to be converged
//         at that instruction; it does not guarantee that the lanes you WANT are there. Its value inside a
//         divergent branch is printed, not checked. The reliable way is to compute the mask with a vote that
//         ALL lanes execute, and to use that mask in the collective operation.

#include "common.h"

// ---- Part 1 -----------------------------------------------------------------------------------
__global__ void block_sum_64(const float *x, float *out) {
  __shared__ float smem[64];
  int t = threadIdx.x;                                   // 64 threads per block, 2 consecutive warps for a block
  smem[t] = x[blockIdx.x * 64 + t];
  __syncthreads();

  if (t < 32) {                                          // a single warp from here on
    float v = smem[t] + smem[t + 32];
    smem[t] = v;
    __syncwarp();
    for (int offset = 16; offset > 0; offset /= 2) {
      v = smem[t] + smem[t + offset];                    // read ...
      __syncwarp();                                      // ... all lanes have read before anyone writes
      smem[t] = v;                                       // write ...
      __syncwarp();                                      // ... all writes are visible before the next read
    }
    if (t == 0) out[blockIdx.x] = smem[0];
  }
}

// ---- Parts 2 and 3 ---------------------------------------------------------------------------
__global__ void masks(int *half_out, unsigned *active_in_branch, unsigned *computed, int *vote_sum) {
  __shared__ int x[16];
  int lane = threadIdx.x;                                // one warp

  if (lane < 16) {
    x[lane] = lane * 10;
    __syncwarp(0x0000ffff);                              // only lanes 0 to 15 meet: they all execute this call
    half_out[lane] = x[15 - lane];                       // lane i reads what lane 15 - i wrote
  } else {
    half_out[lane] = -1;                                 // lanes 16..31 take no part
  }

  // __activemask() inside a divergent branch: informational.
  if (lane % 2 == 0) active_in_branch[lane] = __activemask();
  else               active_in_branch[lane] = __activemask();

  // The reliable way: every lane votes, everybody gets the same computed mask ...
  unsigned even = __ballot_sync(FULL_MASK, lane % 2 == 0);
  computed[lane] = even;
  // ... and the collective operation inside the branch names exactly those lanes.
  if (lane % 2 == 0) {
    int v = __shfl_sync(even, lane, 30);                 // lane 30 is in `even`: every lane of the mask reads its id
    atomicAdd(vote_sum, v);                              // 16 even lanes add 30
  }
}

int main() {
  bool ok = true;

  // Part 1
  const int blocks = 8, n = blocks * 64;
  std::vector<float> h(n), sums(blocks, 0.0f), expected(blocks, 0.0f);
  for (int i = 0; i < n; ++i) { h[i] = (float)(i % 5 + 1); expected[i / 64] += h[i]; }
  float *d_x, *d_out;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  block_sum_64<<<blocks, 64>>>(d_x, d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(sums.data(), d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  for (int b = 0; b < blocks; ++b) ok = ok && sums[b] == expected[b];
  printf("Part 1: block sums with __syncwarp: %s\n", ok ? "correct" : "WRONG");

  // Parts 2 and 3
  int *d_half, *d_vote;
  unsigned *d_active, *d_comp;
  CUDA_CHECK(cudaMalloc(&d_half, 32 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_active, 32 * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_comp, 32 * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_vote, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_vote, 0, sizeof(int)));
  masks<<<1, 32>>>(d_half, d_active, d_comp, d_vote);
  CUDA_CHECK(cudaGetLastError());
  int half[32], vote = -1;
  unsigned active[32], comp[32];
  CUDA_CHECK(cudaMemcpy(half, d_half, sizeof half, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(active, d_active, sizeof active, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(comp, d_comp, sizeof comp, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&vote, d_vote, sizeof vote, cudaMemcpyDeviceToHost));

  bool ok2 = true;
  for (int l = 0; l < 16; ++l)  ok2 = ok2 && half[l] == (15 - l) * 10;
  for (int l = 16; l < 32; ++l) ok2 = ok2 && half[l] == -1;
  printf("Part 2: lanes 0..15 exchanged data with __syncwarp(0x0000ffff): %s\n", ok2 ? "correct" : "WRONG");

  bool ok3 = (vote == 16 * 30);
  for (int l = 0; l < 32; ++l) ok3 = ok3 && comp[l] == 0x55555555u;
  printf("Part 3: __ballot_sync(FULL, lane %% 2 == 0) = 0x%08x in every lane: %s\n", comp[0], ok3 ? "correct" : "WRONG");
  printf("        __activemask() inside the branch: even lanes saw 0x%08x, odd lanes saw 0x%08x\n", active[0], active[1]);
  printf("        (usually 0x55555555 and 0xaaaaaaaa, but the language does not promise it)\n");

  cudaFree(d_x);  cudaFree(d_out);  cudaFree(d_half);  cudaFree(d_active);  cudaFree(d_comp);  cudaFree(d_vote);
  return report("independent_thread_scheduling", ok && ok2 && ok3);
}
