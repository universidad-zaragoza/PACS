// Slide 9 "__syncwarp(): Synchronizing a Warp" (03_Synchronization): reverse 32 values inside one warp.
// The launch uses exactly one warp (1 block of 32 threads), so lane i can read x[31 - i].
#include "common.h"

__global__ void reverse_in_warp(float *out) {
  __shared__ float x[32];
  int lane = threadIdx.x;               // 0..31, one warp

  x[lane] = lane * 2;                   // every lane writes its element
  __syncwarp();                         // the lanes of the warp have written
  out[lane] = x[31 - lane];             // read the element written by another lane
}

int main() {
  float *d_out, h_out[32];
  
  CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(float)));
  
  reverse_in_warp<<<1, 32>>>(d_out);
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost));
  
  bool ok = true;
  for (int i = 0; i < 32; ++i) ok = ok && h_out[i] == 2.0f * (31 - i);
  cudaFree(d_out);
  
  return report("syncwarp_reverse", ok);
}
