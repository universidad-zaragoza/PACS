// Foundations and Runtime API, slides "Built-in Vector Types" and "Alignment and Endianness".
#include "common.h"
#include <vector>

// 16-byte aligned structure: one 128-bit load or store per object.
struct __align__(16) Particle {
  float x, y, z, w;
};
static_assert(sizeof(Particle) == 16 && alignof(Particle) == 16,
              "Particle must be 16 bytes, 16-byte aligned");


// n4 is the number of float4 elements (n / 4).
__global__ void add1(const float *x, float *y, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    // one 128-bit load per thread
    float4 v = reinterpret_cast<const float4*>(x)[i];
    v.x += 1.f;  v.y += 1.f;  v.z += 1.f;  v.w += 1.f;
    reinterpret_cast<float4*>(y)[i] = v;
  }
}

__host__ __device__ inline float4 operator+(float4 a, float4 b) {
     return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__host__ __device__ inline int operator==(float4 a, float4 b) {
  if ((a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w)){
    return true;
  }else{
    return false;
  }
} 

// n4 is the number of float4 elements (n / 4).
__global__ void add11(const float4 *x, float4 *y, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 uno = {1.f, 1.f, 1.f, 1.f};

  if (i < n4) {
    // one 128-bit load per thread
    float4 v = x[i];
    v = v + uno; 
    y[i] = v;
  }
}

int main() {
  float4 a = make_float4(1.f, 2.f, 3.f, 4.f);
  printf("float4 a = (%g, %g, %g, %g)\n", a.x, a.y, a.z, a.w);

  const int n = 1 << 20;                       // multiple of 4
  int n4 = n / 4;
  size_t bytes = n4 * sizeof(float4);
  std::vector<float4> h(n);
  for (int i = 0; i < n4; ++i) h[i] = make_float4(4*i, 4*i+1, 4*i+2, 4*i+3);

  float4 *d_x, *d_y;                            // cudaMalloc returns 256-byte aligned memory
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h.data(), bytes, cudaMemcpyHostToDevice));

  add11<<<(n4 + 255) / 256, 256>>>(d_x, d_y, n4);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n4 && ok; ++i){
      ok = (h[i] == make_float4(4*i+1, 4*i+2, 4*i+3, 4*i+4));
      if (ok != true){
          fprintf(stderr, "Error en h[%d] valores %f, %f, %f, %f\n" , i, h[i].x, h[i].y, h[i].z, h[i].w);
      }
  }
  cudaFree(d_x);  cudaFree(d_y);
  return report("vector_types", ok);
}
