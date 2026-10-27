// Slide 9 "Constant and Read-Only Memory" (02_MemorySpaces): a 16-tap filter.
// coeff[] lives in constant memory (every lane reads the same coeff[j]); `in` is const __restrict__.
// The input holds n + 15 values because each of the n outputs reads 16 consecutive inputs
// (the buffer below has n + 16, one more than needed).
#include "common.h"
#include <vector>

__constant__ float coeff[16];

__global__ void fir(const float *__restrict__ in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    float acc = 0.0f;
    
    for (int j = 0; j < 16; ++j)
      acc += coeff[j] * in[i + j];   // coeff[j]: same address in every lane
    
    out[i] = acc;                    // in[i]: read-only, may use the read-only cache
  }
}

int main() {
  const int n = 1 << 20;
  float h_coeff[16];
  
  for (int j = 0; j < 16; ++j) 
      h_coeff[j] = 1.0f / (1 + j);

  std::vector<float> in(n + 16), out(n), ref(n);
  for (int i = 0; i < n + 16; ++i) 
      in[i] = (float)(i % 13);
  
  for (int i = 0; i < n; ++i) {
    double acc = 0.0;
    for (int j = 0; j < 16; ++j) 
        acc += (double)h_coeff[j] * in[i + j];
    
    ref[i] = (float)acc;
  }

  CUDA_CHECK(cudaMemcpyToSymbol(coeff, h_coeff, sizeof(h_coeff)));

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, (n + 16) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, in.data(), (n + 16) * sizeof(float), cudaMemcpyHostToDevice));

  fir<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n && ok; ++i) 
      ok = close_enough(out[i], ref[i], 1e-5, 1e-4);

  cudaFree(d_in);  
  cudaFree(d_out);
  return report("constant_memory", ok);
}
