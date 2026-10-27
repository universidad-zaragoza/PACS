// Slides 16-17 "Timing inside a Kernel" and "Walk-through: A timeline of Blocks" (04_Timing).
// Part 1: every block records when it started and finished, and on which SM, using the global timer.
// Part 2: one thread counts SM clock cycles around a section of code with clock64().
#include "common.h"

// Global timer, in nanoseconds. It is shared by the whole GPU, but its tick is coarser than the SM clock.
__device__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Number of the SM the calling thread runs on.
__device__ unsigned smid() {
  unsigned id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

struct Stamp { unsigned long long start, end; unsigned sm; };

__global__ void timed_work(const float *in, float *out, int n, int iters, Stamp *stamp) {
  if (threadIdx.x == 0) {                                // one thread per block takes the stamps
    stamp[blockIdx.x].start = now_ns();
    stamp[blockIdx.x].sm = smid();
  }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = in[i];
    for (int k = 0; k < iters; ++k) v = 0.999f * v + 1.0f;
    out[i] = v;
  }
  __syncthreads();                                       // every thread of the block has finished
  if (threadIdx.x == 0) stamp[blockIdx.x].end = now_ns();
}

// Cycles spent by thread 0 on a chain of dependent additions. Only differences of clock64()
// taken by the same thread on the same SM mean anything.
__global__ void cycles_of_loop(int iters, long long *cycles, float *sink) {
  if (threadIdx.x == 0) {
    float v = 1.0f;
    long long c0 = clock64();
    for (int k = 0; k < iters; ++k) v += 0.5f;
    long long c1 = clock64();
    cycles[0] = c1 - c0;
    sink[0] = v;                                         // keep the loop alive
  }
}

int main() {
  const int n = 1 << 18, iters = 2000, block = 256, grid = (n + block - 1) / block;
  const size_t bytes = n * sizeof(float);

  float *h_in = (float *)malloc(bytes), *h_out = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 8);
  float *d_in, *d_out;
  Stamp *d_stamp;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMalloc(&d_stamp, grid * sizeof(Stamp)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));
  timed_work<<<grid, block>>>(d_in, d_out, n, iters, d_stamp);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ev_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ev_ms, e0, e1));

  std::vector<Stamp> st(grid);
  CUDA_CHECK(cudaMemcpy(st.data(), d_stamp, grid * sizeof(Stamp), cudaMemcpyDeviceToHost));

  unsigned long long first = st[0].start, last = st[0].end;
  bool stamps_ok = true;
  for (const Stamp &s : st) {
    first = std::min(first, s.start);
    last = std::max(last, s.end);
    stamps_ok = stamps_ok && s.end >= s.start;
  }
  printf("block   SM   start (us)   duration (us)\n");
  for (int b = 0; b < std::min(grid, 8); ++b)
    printf("%5d %4u %12.1f %15.1f\n", b, st[b].sm, (st[b].start - first) / 1e3, (st[b].end - st[b].start) / 1e3);
  printf("first start to last end : %.3f ms (events: %.3f ms)\n", (last - first) / 1e6, ev_ms);

  // ---- Part 2: cycles with clock64(), converted with the SM clock rate
  long long *d_cycles;
  float *d_sink;
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
  cycles_of_loop<<<1, 32>>>(10000, d_cycles, d_sink);
  CUDA_CHECK(cudaGetLastError());
  long long cycles = 0;
  CUDA_CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(cycles), cudaMemcpyDeviceToHost));
  int dev = 0, khz = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev));
  printf("10000 dependent additions: %lld cycles = about %.2f us at %.0f MHz\n", cycles, cycles * 1e3 / khz, khz / 1e3);

  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  bool ok = stamps_ok && cycles > 0 && ev_ms >= 0.0f;
  for (int i = 0; i < 8; ++i) ok = ok && close_enough(h_out[i], work_ref((float)i, iters));

  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_stamp));
  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_sink));
  free(h_in);
  free(h_out);
  return report("kernel_clocks", ok);
}
