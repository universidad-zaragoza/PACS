// Slide 20 "Atomicity is not Synchronization" (02_MemorySpaces; slide 37 of 03_Synchronization; same code in both folders).
//
//   atomicAdd(&counter, 1);
//   do_something();   // threads can be anywhere
//   atomicAdd(&counter, 1);
//
// Every thread adds 1 to a counter, does something, and adds 1 again. The counter always ends at 2N: no update is
// lost, because each atomicAdd is indivisible. But the atomics did not make the threads wait for each other:
//
//   second = value of the counter just before the SECOND add of a thread.
//   If the first adds acted as a barrier, every second would be >= N (all N first adds are done).
//
// Kernel A has only atomics: some threads do their second add before the last thread has done its first one.
//         The number of such "early" threads depends on the timing, so it is reported, not asserted.
// Kernel B puts a __syncthreads() between the two adds: now NO thread can be early. That is the barrier.
//         (It works only inside one block. Nothing like it exists between blocks inside a kernel: the kernel
//          boundary is the barrier there, see 03_Synchronization/08_kernel_boundary.cu.)
//
// Atomics also order nothing else: other memory accesses need a barrier or a fence.
#include "common.h"
#include <curand_kernel.h>
__device__ unsigned do_something(int *tid) {
  // stands for any work between the two adds; threads may be at completely different points
  curandStatePhilox4_32_10_t st;          // each thread owns its generator state
  curand_init(17, *tid, 0, &st);         // seed, subsequence (use tid), offset

  unsigned local = 0;
  for (int i = 0; i < *tid; ++i) {
     float x = curand_uniform(&st);        // in (0, 1]
     float y = curand_uniform(&st);
     local += (x * x + y * y <= 1.0f);
  }

  return local;
}

__global__ void atomics_only(int *counter, int *early, int N, int *out) {
  int tid = threadIdx.x;
  atomicAdd(counter, 1);                         // first add
  unsigned tmp=0;
  //tmp = do_something(&tid);
  int second = atomicAdd(counter, 1);            // second add: returns the value before it
  if (second < N) 
      atomicAdd(early, 1);           // the first adds of other threads were not all done yet
      
  *out = tmp;
}

__global__ void with_barrier(int *counter, int *early, int N) {   // one block of N threads
  atomicAdd(counter, 1);
  __syncthreads();                               // every first add is done before any second add starts
  int second = atomicAdd(counter, 1);
  if (second < N) atomicAdd(early, 1);
}

int main() {
  const int N = 256;
  int *d_counter, *d_early, *d_out;
  CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_early, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
  int counter = 0, early = 0, out;

  CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_early, 0, sizeof(int)));
  
  atomics_only<<<4, N / 4>>>(d_counter, d_early, N, d_out);          // 4 blocks of 64 threads
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&counter, d_counter, sizeof counter, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&early, d_early, sizeof early, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&out, d_out, sizeof out, cudaMemcpyDeviceToHost));
  printf("A) atomics only : counter = %d (expected %d); threads whose second add came before all first adds: %d\n",
         counter, 2 * N, early);
  if (out == 0)
      printf("This problem can go unnoticed. Uncomment line 41\n");
  bool ok = (counter == 2 * N);

  CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_early, 0, sizeof(int)));
  
  with_barrier<<<1, N>>>(d_counter, d_early, N);              // one block of 256 threads
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&counter, d_counter, sizeof counter, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&early, d_early, sizeof early, cudaMemcpyDeviceToHost));
  printf("B) with barrier : counter = %d (expected %d); early threads: %d (must be 0)\n", counter, 2 * N, early);
  ok = ok && counter == 2 * N && early == 0;

  cudaFree(d_counter);  cudaFree(d_early);
  return report("atomic_not_barrier", ok);
}
