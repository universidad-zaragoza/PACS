// "Shared Memory Banks (bank conflict)" (02_MemorySpaces): tile[32][32] versus the padded tile[32][33].
// Each lane walks down a column, so without padding all 32 lanes hit the same bank.
#include "common.h"
#include <vector>

template <int PAD>
__global__ void colwalk(float *out, int iters) {
  __shared__ float tile[32][32 + PAD];
  for (int r = 0; r < 32; ++r)
    tile[r][threadIdx.x] = (float)(r + threadIdx.x);    // row write: no conflict
  __syncthreads();

  float s = 0.0f;
  for (int it = 0; it < iters; ++it)
    for (int r = 0; r < 32; ++r)
      s += tile[(threadIdx.x + it) & 31][r];            // column access, possible conflict between threads in the same warp accessing the same bank
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main() {
  const int blocks = 1024, iters = 2000;
  float *d0, *d1;
  CUDA_CHECK(cudaMalloc(&d0, blocks * 32 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d1, blocks * 32 * sizeof(float)));

  colwalk<0><<<blocks, 32>>>(d0, 1);                    // warm-up
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t;
  t.start();
  colwalk<0><<<blocks, 32>>>(d0, iters);                // float tile[32][32]
  float ms_conflict = t.stop();
  t.start();
  colwalk<1><<<blocks, 32>>>(d1, iters);                // float tile[32][33]
  float ms_padded = t.stop();
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> a(blocks * 32), b(blocks * 32);
  CUDA_CHECK(cudaMemcpy(a.data(), d0, a.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b.data(), d1, b.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool same = true;                                      // padding must not change results
  for (size_t i = 0; i < a.size() && same; ++i) same = (a[i] == b[i]);

  printf("tile[32][32]: %.3f ms (32-way bank conflict)\n", ms_conflict);
  printf("tile[32][33]: %.3f ms (conflict-free)\n", ms_padded);

  cudaFree(d0);  cudaFree(d1);
  return report("bank_conflicts", same);
}
