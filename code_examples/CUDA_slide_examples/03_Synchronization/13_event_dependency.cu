// Slide 30 "Events: Dependencies between streams" (03_Synchronization): stream 2 waits for a point in stream 1,
// the CPU does not. Exercise "Two streams and an event" (slide 52).
#include "common.h"

__global__ void kernel1(int *a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] = i; }
__global__ void kernel2(const int *a, int *b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) b[i] = 3 * a[i]; }

int main() {
  const int n = 1 << 18, block = 256, grid = (n + block - 1) / block;
  int *a, *b;

  CUDA_CHECK(cudaMalloc(&a, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&b, n * sizeof(int)));

  cudaStream_t stream1, stream2;
  CUDA_CHECK(cudaStreamCreate(&stream1));
  CUDA_CHECK(cudaStreamCreate(&stream2));
  
  cudaEvent_t event;
  CUDA_CHECK(cudaEventCreate(&event));

  kernel1<<<grid, block, 0, stream1>>>(a, n);
  CUDA_CHECK(cudaEventRecord(event, stream1));             // "everything issued so far in stream1 is done"
  CUDA_CHECK(cudaStreamWaitEvent(stream2, event, 0));      // stream2 must not go past this point before that
  kernel2<<<grid, block, 0, stream2>>>(a, b, n);           // reads what kernel1 wrote
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream2));              // the CPU waits only at the end

  int *h = (int *)malloc(n * sizeof(int));
  CUDA_CHECK(cudaMemcpy(h, b, n * sizeof(int), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < n; ++i) ok = ok && h[i] == 3 * i;
  
  free(h);
  cudaEventDestroy(event);  
  cudaStreamDestroy(stream1);  
  cudaStreamDestroy(stream2);
  cudaFree(a);  
  cudaFree(b);
  return report("event_dependency", ok);
}
