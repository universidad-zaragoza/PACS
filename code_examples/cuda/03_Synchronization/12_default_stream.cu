// Slides 25-28 "Streams and the Default Stream" ... "The Per-Thread Default Stream" (03_Synchronization).
// Three kernels busy-wait for 20 ms each and record when they ran:
//   k1 in stream a, k2 in the default stream, k3 in stream b.
// The program prints the three time intervals for four configurations.
// Only the first one is guaranteed by the rules of the legacy default stream, so only it is checked.
// Case 4 uses plain stream 0, so it changes with the compile flag:
//     nvcc --default-stream per-thread 12_default_stream.cu   (stream 0 then means cudaStreamPerThread)
// Cases 1 to 3 name their stream with an explicit handle, so the flag does not change them.
#include "common.h"
#include <algorithm>

__device__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));   // GPU-wide timer, in nanoseconds (the tick is coarser)
  return t;
}

__global__ void spin(unsigned long long ns, unsigned long long *stamp) {
  unsigned long long t0 = now_ns();
  while (now_ns() - t0 < ns) { }
  stamp[0] = t0;                                          // when this kernel started
  stamp[1] = now_ns();                                    // when it finished
}

static bool overlaps(const unsigned long long *t, int p, int q) {
  return t[2 * p] < t[2 * q + 1] && t[2 * q] < t[2 * p + 1];
}

// Launch k1 in a, k2 in `mid`, k3 in b. Returns true when no two kernels overlapped.
static bool run(const char *title, unsigned stream_flags, cudaStream_t mid) {
  const unsigned long long ns = 20000000ULL;              // 20 ms per kernel
  
  cudaStream_t a, b;
  CUDA_CHECK(cudaStreamCreateWithFlags(&a, stream_flags));
  CUDA_CHECK(cudaStreamCreateWithFlags(&b, stream_flags));
  
  unsigned long long *d_stamp;
  CUDA_CHECK(cudaMalloc(&d_stamp, 6 * sizeof(unsigned long long)));

  spin<<<1, 1, 0, a>>>(ns, d_stamp + 0);                  // k1
  spin<<<1, 1, 0, mid>>>(ns, d_stamp + 2);                // k2
  spin<<<1, 1, 0, b>>>(ns, d_stamp + 4);                  // k3
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize()); //Force waiting for everything pending at the device

  unsigned long long t[6];
  CUDA_CHECK(cudaMemcpy(t, d_stamp, sizeof(t), cudaMemcpyDeviceToHost));
  unsigned long long t0 = std::min(t[0], std::min(t[2], t[4]));
  printf("%s\n", title);
  for (int k = 0; k < 3; ++k)
    printf("  k%d: %6.1f ms -> %6.1f ms\n", k + 1, (t[2 * k] - t0) / 1e6, (t[2 * k + 1] - t0) / 1e6);
  bool none = !overlaps(t, 0, 1) && !overlaps(t, 0, 2) && !overlaps(t, 1, 2);
  printf("  %s\n\n", none ? "serialized: no two kernels overlap" : "overlap: some kernels ran at the same time");

  // With a legacy-stream barrier the order is guaranteed: k1, then k2, then k3.
  bool in_order = t[2] >= t[1] && t[4] >= t[3];
  CUDA_CHECK(cudaFree(d_stamp));
  CUDA_CHECK(cudaStreamDestroy(a));
  CUDA_CHECK(cudaStreamDestroy(b));
  return none && in_order;
}

int main() {
  // The explicit handles cudaStreamLegacy and cudaStreamPerThread select the behavior
  // whatever the "--default-stream" compile flag says.
  bool serialized = run("1. Blocking streams a and b, legacy default stream: a barrier",
                        cudaStreamDefault, cudaStreamLegacy);
  run("2. Non-blocking streams a and b, legacy default stream: no implicit barrier",
      cudaStreamNonBlocking, cudaStreamLegacy);
  run("3. Blocking streams a and b, per-thread default stream: an ordinary stream",
      cudaStreamDefault, cudaStreamPerThread);
  // Case 4 uses plain stream 0: its meaning is chosen at COMPILE time (slide 28).
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  run("4. Blocking streams a and b, plain stream 0 (compiled with --default-stream per-thread)",
      cudaStreamDefault, 0);
#else
  run("4. Blocking streams a and b, plain stream 0 (legacy default stream; compile with --default-stream per-thread to change it)",
      cudaStreamDefault, 0);
#endif
  printf("Cases 2 and 3 usually overlap, but the hardware is free not to run kernels concurrently.\n");
  return report("default_stream", serialized);
}
