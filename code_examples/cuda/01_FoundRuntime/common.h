// common.h - small helpers shared by all examples.
// CUDA_CHECK is the macro shown on the "Error Handling" slide (Part 1).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", \
      __FILE__, __LINE__, \
      cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

// Floating-point results of the GPU and the CPU can differ in the last bits
// (fused multiply-add, different summation order), so compare with a
// tolerance instead of ==.
inline bool close_enough(double got, double want,
                         double rtol = 1e-5, double atol = 1e-6) {
  return std::fabs(got - want) <= atol + rtol * std::fabs(want);
}

// Minimal GPU timer built on CUDA events (see Part 2, "CUDA Events for Timing").
struct GpuTimer {
  cudaEvent_t a, b;
  GpuTimer()  { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
  ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
  void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(a, s)); }
  float stop(cudaStream_t s = 0) {      // milliseconds
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
  }
};

// Prints the verdict and returns the process exit code.
inline int report(const char *name, bool ok) {
  printf("%s: %s\n", name, ok ? "PASSED" : "FAILED");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
