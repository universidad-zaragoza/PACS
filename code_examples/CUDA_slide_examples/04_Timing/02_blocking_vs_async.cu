// Slide 5 "Blocking and Asynchronous Operations" (04_Timing).
// How long does the HOST spend inside each call while the GPU is busy?
//
// For every case the program queues a long kernel (about 40 ms) in the default stream, then makes the call to
// measure and reads the host clock right after the call returns. If the call is asynchronous it returns long before
// the kernel ends; if it has to wait for the stream (or for the device) the host time is close to the kernel time.
//
//   operation                                   slide says
//   kernel launch                               asynchronous: returns immediately
//   cudaMemcpy, device to host                  synchronous: host waits for the copy
//   cudaMemcpy, host to device                  pageable: returns once staged. Pinned: waits for the copy
//   cudaMemcpyAsync, pageable host memory       may be staged: overlap not guaranteed
//   cudaMemcpyAsync, pinned host memory         asynchronous: overlaps with work in other streams
//   cudaMalloc, cudaFree                        may serialize the device
//
// The numbers depend on the GPU, the driver and the size of the copy (small pageable copies are often staged and
// return at once), so the program REPORTS the classification and asserts only that the data are correct.
#include "common.h"

__device__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));      // GPU-wide timer, in nanoseconds
  return t;
}

__global__ void busy(unsigned long long ns) {                 // keeps the GPU busy for about `ns`
  unsigned long long t0 = now_ns();
  while (now_ns() - t0 < ns) { }
}

__global__ void incr(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += 1.0f;
}

constexpr double KERNEL_MS = 740.0;

// Queue the busy kernel, run `call`, return the host time spent in `call` (ms), then drain the GPU.
template <class F>
static double host_time_of(F call) {
  
  CUDA_CHECK(cudaDeviceSynchronize());                        // start from an idle device
  
  busy<<<1, 1>>>((unsigned long long)(KERNEL_MS * 1e6));
  CUDA_CHECK(cudaGetLastError());
  
  double t0 = now_ms();
  call();
  double t1 = now_ms();
  CUDA_CHECK(cudaDeviceSynchronize());
  return t1 - t0;
}

static void row(const char *what, double ms) {
  const char *verdict = ms < 0.25 * KERNEL_MS ? "returned at once (asynchronous)" : "waited for the GPU (blocking)";
  printf("  %-44s %8.3f ms  %s\n", what, ms, verdict);
}

int main() {
  const int n = 1 << 20;
  const size_t bytes = n * sizeof(float);
  float *pageable = (float *)malloc(bytes), *pinned, *d;
  CUDA_CHECK(cudaMallocHost(&pinned, bytes));
  CUDA_CHECK(cudaMalloc(&d, bytes));
  
  for (int i = 0; i < n; ++i) { 
      pageable[i] = 1.0f; 
      pinned[i] = 1.0f; 
  }
  CUDA_CHECK(cudaMemcpy(d, pinned, bytes, cudaMemcpyHostToDevice));

  printf("A kernel of %.0f ms is queued before each call; host time spent in the call:\n", KERNEL_MS);
  row("kernel launch", host_time_of([&] { incr<<<(n + 255) / 256, 256>>>(d, n); }));
  row("cudaMemcpy device to host (pageable)", host_time_of([&] { CUDA_CHECK(cudaMemcpy(pageable, d, bytes, cudaMemcpyDeviceToHost)); }));
  row("cudaMemcpy device to host (pinned)", host_time_of([&] { CUDA_CHECK(cudaMemcpy(pinned, d, bytes, cudaMemcpyDeviceToHost)); }));
  row("cudaMemcpy host to device (pageable)", host_time_of([&] { CUDA_CHECK(cudaMemcpy(d, pageable, bytes, cudaMemcpyHostToDevice)); }));
  row("cudaMemcpy host to device (pinned)", host_time_of([&] { CUDA_CHECK(cudaMemcpy(d, pinned, bytes, cudaMemcpyHostToDevice)); }));
  row("cudaMemcpyAsync host to device (pageable)", host_time_of([&] { CUDA_CHECK(cudaMemcpyAsync(d, pageable, bytes, cudaMemcpyHostToDevice, 0)); }));
  row("cudaMemcpyAsync host to device (pinned)", host_time_of([&] { CUDA_CHECK(cudaMemcpyAsync(d, pinned, bytes, cudaMemcpyHostToDevice, 0)); }));
  row("cudaMemcpyAsync device to host (pinned)", host_time_of([&] { CUDA_CHECK(cudaMemcpyAsync(pinned, d, bytes, cudaMemcpyDeviceToHost, 0)); }));

  float *tmp = nullptr;
  row("cudaMalloc", host_time_of([&] { CUDA_CHECK(cudaMalloc(&tmp, bytes)); }));
  row("cudaFree", host_time_of([&] { CUDA_CHECK(cudaFree(tmp)); }));

  // Data check, after all the measured calls
  bool ok = true;
  for (int i = 0; i < n && ok; ++i) ok = (pageable[i] == 2.0f);

  free(pageable);
  CUDA_CHECK(cudaFreeHost(pinned));
  CUDA_CHECK(cudaFree(d));
  return report("blocking_vs_async", ok);
}
