// Fondations, slide "SAXPY: From Loop to Kernel".
#include "common.h"
#include <vector>

// Kernel
__global__ void saxpy(int n, float a, const float *x, float *y) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) y[i] += a * x[i];
}


//host side
int main() {
  const int n = (1 << 20) + 123;
  const float a = 2.5f;
  size_t bytes = n * sizeof(float);

  std::vector<float> x(n), y(n), ref(n);
  for (int i = 0; i < n; ++i) {
    x[i] = 0.001f * (i % 1000);
    y[i] = 1.0f + 0.002f * (i % 500);
  }
  for (int i = 0; i < n; ++i) ref[i] = y[i] + a * x[i];   // CPU loop

  float *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n + block - 1) / block;
  saxpy<<<grid, block>>>(n, a, d_x, d_y);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n && ok; ++i) ok = close_enough(y[i], ref[i]);

  cudaFree(d_x);  cudaFree(d_y);
  return report("saxpy", ok);
}
