// Foundations and Runtime API, slides "Complete Example (1/2)" and "(2/2)", in one file.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "common.h"      // provides CUDA_CHECK

//kernel
__global__ void scale(const int *in, int *out, int f, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * f;
}

//host side
int main() {
  const int N = 1 << 20;
  size_t bytes = N * sizeof(int);
  int *h_in  = (int*)malloc(bytes);
  int *h_out = (int*)malloc(bytes);
  for (int i = 0; i < N; ++i) h_in[i] = i;

  int *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in,  bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  int block = 256, grid = (N + block - 1) / block;
  scale<<<grid, block>>>(d_in, d_out, 2, N);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  printf("out[10] = %d\n", h_out[10]);

  bool ok = true;
  for (int i = 0; i < N && ok; ++i) ok = (h_out[i] == 2 * i);

  cudaFree(d_in);  cudaFree(d_out);
  free(h_in);  free(h_out);
  return report("scale_complete", ok);
}
