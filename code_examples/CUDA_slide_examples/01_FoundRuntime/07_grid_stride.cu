// Foundation and Runtime API, slides "Choosing the Grid Size" and "Grid-Stride Loops".
#include "common.h"
#include <vector>

// One thread per element: the grid must cover n.
__global__ void kernel(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= 2.0f;
}

// Grid-stride loop: any n, any grid size.
__global__ void scale(float *x, int n, float a) {
  int stride = gridDim.x * blockDim.x;
  int start  = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = start; i < n; i += stride)
    x[i] *= a;
}

int main() {
  const int n = 5000001;
  size_t bytes = n * sizeof(float);
  std::vector<float> h(n, 1.0f);
  float *d_x;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h.data(), bytes, cudaMemcpyHostToDevice));

  // 1) Grid sized from n.
  int block = 256;
  int grid  = (n + block - 1) / block;
  kernel<<<grid, block>>>(d_x, n);
  CUDA_CHECK(cudaGetLastError());

  // d_x stays in device memory for the following invokation 

  // 2) Grid sized from the GPU, independent of n.
  int dev = 0, numSMs = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
  scale<<<numSMs * 8, 256>>>(d_x, n, 2.0f);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h.data(), d_x, bytes, cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < n && ok; ++i) ok = (h[i] == 4.0f);   // 1 * 2 * 2

  cudaFree(d_x);
  return report("grid_stride", ok);
}
