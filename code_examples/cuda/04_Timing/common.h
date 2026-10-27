// common.h - helpers shared by the timing examples.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

inline bool close_enough(double got, double want, double rtol = 1e-4, double atol = 1e-5) {
  return std::fabs(got - want) <= atol + rtol * std::fabs(want);
}

// Host wall clock in milliseconds. steady_clock never goes backwards.
inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// A kernel whose run time we can scale with `iters`: every thread iterates
// v = 0.999*v + 1 on its own element. The iteration converges to 1000, so the
// result does not depend on rounding details (fused multiply-add or not).
__global__ void work(const float *in, float *out, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = in[i];
    for (int k = 0; k < iters; ++k) v = 0.999f * v + 1.0f;
    out[i] = v;
  }
}

// The same computation on the CPU, for checking.
inline float work_ref(float v, int iters) {
  for (int k = 0; k < iters; ++k) v = 0.999f * v + 1.0f;
  return v;
}

// Prints the verdict and returns the process exit code.
inline int report(const char *name, bool ok) {
  printf("%s: %s\n", name, ok ? "PASSED" : "FAILED");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
