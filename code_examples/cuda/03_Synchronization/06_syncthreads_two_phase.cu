// Slides 17-18 "__syncthreads(): A Barrier for the Block" and "Walk-through: Two Phases" (03_Synchronization).
// Phase 1: every thread writes one element to shared memory. Phase 2: every thread reads its neighbour's element.
// Exercise "Remove the barrier" (slide 52): see 23_sanitizer_bugs.cu, or delete the barrier here and run racecheck.
#include "common.h"

__global__ void example(const float *input, float *output) {
  __shared__ float data[256];
  int tid = threadIdx.x;

  data[tid] = input[tid];                  // phase 1: write
  __syncthreads();                         // the whole block has written before anyone reads
  output[tid] = data[(tid + 1) % 256];     // phase 2: read what another thread wrote
}

int main() {
  const int n = 256;
  float h_in[n], h_out[n];
  for (int i = 0; i < n; ++i) 
      h_in[i] = (float)i;

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

  example<<<1, n>>>(d_in, d_out);                       // one block of 256 threads
  CUDA_CHECK(cudaGetLastError());
  
  CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i) 
      ok = ok && h_out[i] == h_in[(i + 1) % n];
  
  printf("output[255] = %g (thread 255 read data[0], written by thread 0, try removing line 11(__syncthreads) and see what happens)\n", h_out[255]);
  
  cudaFree(d_in);  cudaFree(d_out);
  return report("syncthreads_two_phase", ok);
}
