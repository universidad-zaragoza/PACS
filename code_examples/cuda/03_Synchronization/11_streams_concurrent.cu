// Slides 24, 29 and 32 "cudaStreamSynchronize(): CPU waits for a stream", "Streams may run concurrently" and
// "cudaDeviceSynchronize(): CPU waits for the Device" (03_Synchronization).
// Waiting for stream 1 says nothing about stream 2. The result of cudaStreamQuery depends on timing and on
// the GPU, so it is reported, not asserted.
#include "common.h"

__global__ void spin(long long cycles) {                 // keeps one thread busy for about `cycles` clock ticks
  long long t0 = clock64();
  while (clock64() - t0 < cycles) { }
}

__global__ void quick(int *x) { *x = 42; }

int main() {
  cudaStream_t s1, s2;

  CUDA_CHECK(cudaStreamCreate(&s1));
  CUDA_CHECK(cudaStreamCreate(&s2));
  int *d;  CUDA_CHECK(cudaMalloc(&d, sizeof(int)));

  spin<<<1, 1, 0, s2>>>(1000000000LL);                    // long kernel in stream 2 (about 0.3 s or more)
  quick<<<1, 1, 0, s1>>>(d);                              // short kernel in stream 1

  CUDA_CHECK(cudaStreamSynchronize(s1));                  // the CPU waits for stream 1 ...
  cudaError_t r = cudaStreamQuery(s2);                    // ... but stream 2 may still be running
  printf("after cudaStreamSynchronize(s1), cudaStreamQuery(s2) = %s\n",
         r == cudaSuccess ? "cudaSuccess (stream 2 finished or the streams did not overlap)"
                          : "cudaErrorNotReady (stream 2 still running: the streams overlap)");

  CUDA_CHECK(cudaDeviceSynchronize());                    // now wait for everything on the device
  bool ok = cudaStreamQuery(s2) == cudaSuccess;           // after the device barrier both streams are idle

  int h = 0;
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost));
  ok = ok && h == 42;
  
  cudaStreamDestroy(s1); 
  cudaStreamDestroy(s2);  
  cudaFree(d);
  return report("streams_concurrent", ok);
}
