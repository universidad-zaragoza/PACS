// "Three-Point Stencil with Shared Memory" (02_MemorySpaces).
// Each block loads BS elements plus one halo cell on each side into shared memory.
// Exercise "Race hunt" (slide 24): build with -DNO_SYNC to remove the barrier, then run
//     compute-sanitizer --tool racecheck ./03_stencil_shared
// and read the report. The default build keeps __syncthreads() and is correct. The -DNO_SYNC build may print
// PASSED or FAILED depending on the timing of the warps: a race does not have to give a wrong answer every time.
#include "common.h"
#include <vector>

#define BS 256
__global__ void avg3(const float *x, float *y, int n) {
  __shared__ float s[BS + 2];
  int i  = blockIdx.x * BS + threadIdx.x;
  int il = threadIdx.x + 1;
  
  s[il] = (i < n) ? x[i] : 0.0f;  //Comprobar que no nos salimos del dominio

  if (threadIdx.x == 0)
    s[0] = (i > 0) ? x[i - 1] : 0.0f;
  
  if (threadIdx.x == BS - 1)
    s[BS + 1] = (i + 1 < n) ? x[i + 1] : 0.0f;

#ifndef NO_SYNC
  __syncthreads();
#endif
  
  if (i < n)
    y[i] = (s[il-1] + s[il] + s[il+1]) / 3.0f;
}

int main() {
  const int n = (1 << 20) + 37;
  std::vector<float> x(n), y(n), ref(n);
  for (int i = 0; i < n; ++i) x[i] = (float)((i * 7) % 101);
  for (int i = 0; i < n; ++i) {
    float l = (i > 0) ? x[i - 1] : 0.0f, r = (i + 1 < n) ? x[i + 1] : 0.0f;
    ref[i] = (l + x[i] + r) / 3.0f;
  }

  float *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  avg3<<<(n + BS - 1) / BS, BS>>>(d_x, d_y, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n && ok; ++i) ok = close_enough(y[i], ref[i]);

  cudaFree(d_x);  cudaFree(d_y);
  return report("stencil_shared", ok);
}
