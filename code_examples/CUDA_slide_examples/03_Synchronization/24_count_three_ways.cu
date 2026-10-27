// Exercise "Count three ways" (slide 52, 03_Synchronization): plain increment, global atomicAdd, and a shared
// counter with one global atomic. Compare results and time. The result could be the same but not the performance 
//
//   A) count_plain   : (*counter)++ from every thread             -> lost updates (slides 34-35)
//   B) count_global  : atomicAdd(counter, 1) from every thread    -> correct, but every thread hits ONE global word
//   C) count_shared  : atomicAdd on a __shared__ counter, then ONE global atomicAdd per block
//                      (slides 38-39: two barriers around the shared atomics)
//
// Expected: A is wrong (unless the GPU is lucky), B and C are exact, and C is usually the fastest because the
// contention happens in fast on-chip memory and only `blocks` atomics reach global memory
//
// Observe the timing results for option B and C. Compare this results with other grid sizes , f.i. blocks=1024
// and threads = 256. Any difference? why?

#include "common.h"

__global__ void count_plain(int *counter)  { (*counter)++; }
__global__ void count_global(int *counter) { atomicAdd(counter, 1); }

__global__ void count_shared(int *counter) {
  __shared__ int local;
  if (threadIdx.x == 0)
    local = 0;
  __syncthreads();                           // local = 0 is visible before any thread adds to it
  atomicAdd(&local, 1);
  __syncthreads();                           // every increment is done before thread 0 reads local
  if (threadIdx.x == 0)
    atomicAdd(counter, local);
}

// Zero the counter, time the launch with events, read the counter back.
template <class Launch>
static float run(Launch launch, int *d, int *result) {
  CUDA_CHECK(cudaMemset(d, 0, sizeof(int)));
  GpuTimer t;
  t.start();
  launch();
  float ms = t.stop();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(result, d, sizeof(int), cudaMemcpyDeviceToHost));
  return ms;
}

int main() {
  const int blocks = 1024, threads = 256, total = blocks * threads;
  int *d;
  CUDA_CHECK(cudaMalloc(&d, sizeof(int)));

  auto plain  = [&] { count_plain<<<blocks, threads>>>(d); };
  auto global = [&] { count_global<<<blocks, threads>>>(d); };
  auto shared = [&] { count_shared<<<blocks, threads>>>(d); };

  int r = 0;
  run(global, d, &r);                                          // warm-up
  float ms_a = run(plain, d, &r);   int a = r;
  float ms_b = run(global, d, &r);  int b = r;
  float ms_c = run(shared, d, &r);  int c = r;

  printf("expected                       : %d\n", total);
  printf("A) plain increment             : %8d  %8.3f ms  (lost updates; not asserted)\n", a, ms_a);
  printf("B) global atomicAdd            : %8d  %8.3f ms\n", b, ms_b);
  printf("C) shared counter + 1 global   : %8d  %8.3f ms\n", c, ms_c);

  cudaFree(d);
  return report("count_three_ways", b == total && c == total);
}
