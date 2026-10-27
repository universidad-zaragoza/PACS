// Foundations and Runtime API  "Task Parallelism with Streams": two independent tasks overlap.
#include "common.h"

constexpr int N = 1 << 20;

__global__ void function1(const float *x, float *u) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) u[i] = 2.0f * x[i];
}
__global__ void function2(const float *x, float *v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) v[i] = x[i] + 1.0f;
}

int main() {
  size_t bytes = N * sizeof(float);
  float *h_x, *h_u, *h_v;                       // pinned memory allows async copies
  CUDA_CHECK(cudaMallocHost(&h_x, bytes));
  CUDA_CHECK(cudaMallocHost(&h_u, bytes));
  CUDA_CHECK(cudaMallocHost(&h_v, bytes));
  for (int i = 0; i < N; ++i) h_x[i] = (float)(i % 1000);

  float *d_x1, *d_x2, *d_u, *d_v;
  CUDA_CHECK(cudaMalloc(&d_x1, bytes));
  CUDA_CHECK(cudaMalloc(&d_x2, bytes));
  CUDA_CHECK(cudaMalloc(&d_u, bytes));
  CUDA_CHECK(cudaMalloc(&d_v, bytes));

  dim3 block(256), grid((N + 255) / 256);

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);  cudaStreamCreate(&s2);
  cudaMemcpyAsync(d_x1, h_x, bytes, cudaMemcpyHostToDevice, s1);
  cudaMemcpyAsync(d_x2, h_x, bytes, cudaMemcpyHostToDevice, s2);
  function1<<<grid, block, 0, s1>>>(d_x1, d_u);
  function2<<<grid, block, 0, s2>>>(d_x2, d_v);
  cudaMemcpyAsync(h_u, d_u, bytes, cudaMemcpyDeviceToHost, s1);
  cudaMemcpyAsync(h_v, d_v, bytes, cudaMemcpyDeviceToHost, s2);

  CUDA_CHECK(cudaStreamSynchronize(s1));
  CUDA_CHECK(cudaStreamSynchronize(s2));
  CUDA_CHECK(cudaGetLastError());

  bool ok = true;
  for (int i = 0; i < N && ok; ++i)
    ok = (h_u[i] == 2.0f * h_x[i]) && (h_v[i] == h_x[i] + 1.0f);

  cudaStreamDestroy(s1);  cudaStreamDestroy(s2);
  cudaFree(d_x1); cudaFree(d_x2); cudaFree(d_u); cudaFree(d_v);
  cudaFreeHost(h_x); cudaFreeHost(h_u); cudaFreeHost(h_v);
  return report("streams_task_parallel", ok);
}
