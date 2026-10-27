// "Kinds of Host Memory" and "Pinned Memory in Practice" (02_MemorySpaces):
// the same copy -> kernel -> copy sequence, once with pageable and once with pinned host memory.
#include "common.h"

__global__ void scale(float *x, int n, float a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) 
      x[i] *= a;
}

// Copy in, kernel, copy out, all queued in stream s. Returns the time of the sequence in ms.
static float run(float *h, float *d, size_t n, cudaStream_t s) {
  size_t bytes = n * sizeof(float);
  GpuTimer t;
  
  t.start(s);
  
  CUDA_CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s));
  
  scale<<<(int)((n + 255) / 256), 256, 0, s>>>(d, (int)n, 2.0f);
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s));
  
  float ms = t.stop(s);                       // records the stop event and waits for it
  CUDA_CHECK(cudaStreamSynchronize(s));       // h holds the result only after this point
  
  return ms;
}

int main() {
  const size_t n = (size_t)1 << 24;           // 64 MiB of floats
  const size_t bytes = n * sizeof(float);
  float *d;
  
  CUDA_CHECK(cudaMalloc(&d, bytes));
  
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  float *pageable = (float *)malloc(bytes);
  float *pinned;
  CUDA_CHECK(cudaMallocHost(&pinned, bytes)); // pinned: free it with cudaFreeHost, not cudaFree

  auto fill = [&](float *p) { for (size_t i = 0; i < n; ++i) p[i] = 1.0f; };
  auto all_two = [&](const float *p) {
    for (size_t i = 0; i < n; ++i) if (p[i] != 2.0f) return false;
    return true;
  };

  fill(pageable);  
  run(pageable, d, n, s);                 // warm-up: context and module loading
  
  fill(pageable);  
  float t_pageable = run(pageable, d, n, s);
  
  fill(pinned);    
  float t_pinned   = run(pinned, d, n, s);

  printf("pageable: %7.2f ms   %6.2f GB/s moved\n", t_pageable, 2.0 * bytes / (t_pageable * 1e6));
  printf("pinned:   %7.2f ms   %6.2f GB/s moved\n", t_pinned, 2.0 * bytes / (t_pinned * 1e6));
  bool ok = all_two(pageable) && all_two(pinned);

  free(pageable);
  CUDA_CHECK(cudaFreeHost(pinned));
  CUDA_CHECK(cudaFree(d));
  CUDA_CHECK(cudaStreamDestroy(s));
  return report("pinned_memory", ok);
}
