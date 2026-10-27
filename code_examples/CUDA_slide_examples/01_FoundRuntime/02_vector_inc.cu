// Foundations part, slide "Data Parallelism": out[i] = in[i] + 1 on the GPU.
#include "common.h"


// Kernel function
__global__ void inc(const int *in, int *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] + 1;
}


//Host side
int main() {
  const int N = 1000003;                 // not a multiple of 256 on purpose
  size_t bytes = N * sizeof(int);
  int *h_in  = (int *)malloc(bytes);
  int *h_out = (int *)malloc(bytes);
  for (int i = 0; i < N; ++i) h_in[i] = i;

  int *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in,  bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  // host side
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
  
  inc<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int i = 0; i < N; ++i) bad += (h_out[i] != h_in[i] + 1);

  cudaFree(d_in);  cudaFree(d_out);
  free(h_in);  free(h_out);
  return report("vector_inc", bad == 0);
}
