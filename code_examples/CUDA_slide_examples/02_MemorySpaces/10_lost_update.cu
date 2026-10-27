// Slides 17-18 "The Lost Update: race conditions" and "atomicAdd(): an indivisible update"
// (02_MemorySpaces slides 17-18, 03_Synchronization slides 34-35; the code is the same in both folders).
//
//   (*counter)++    is  read, added, written: three separate steps, so two threads can read the same old value
//   atomicAdd       is one indivisible step, and it returns the OLD value
//
// 1. count_racy: many updates are lost. The printed value is usually below the expected one on a real GPU;
//    it is reported, not asserted, because a race may happen to give the right answer.
// 2. count_atomic: always exactly the number of threads.
// 3. tickets: atomicAdd returns the old value, so every thread gets a different number 0 .. N-1.
#include "common.h"

__global__ void count_racy(int *counter)   { (*counter)++; }          // read, add, write: race conditions are possible
__global__ void count_atomic(int *counter) { atomicAdd(counter, 1); } // one indivisible step

__global__ void tickets(int *counter, int *taken) {
  int old = atomicAdd(counter, 1);        // the old value: a different number for every thread
  atomicAdd(&taken[old], 1);              // count how many threads got each number
}

int main() {
  int *d;
  CUDA_CHECK(cudaMalloc(&d, sizeof(int)));

  // The slide: count<<<100, 10>>>(counter) with 1000 threads.
  int racy;
  CUDA_CHECK(cudaMemset(d, 0, sizeof(int)));
  count_racy<<<100,10>>>(d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&racy, d, sizeof racy, cudaMemcpyDeviceToHost));
  
  int atom;
  CUDA_CHECK(cudaMemset(d, 0, sizeof(int)));
  count_atomic<<<100,10>>>(d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&atom, d, sizeof atom, cudaMemcpyDeviceToHost));
  
  printf("100 x 10 threads : plain increment = %4d (expected 1000, usually less on a real GPU)\n", racy);
  printf("                   atomicAdd       = %4d (expected 1000)\n", atom);

  // With more threads more updates are lost.
  int racy2 ;
  CUDA_CHECK(cudaMemset(d, 0, sizeof(int)));
  count_racy<<<100,256>>>(d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&racy2, d, sizeof racy2, cudaMemcpyDeviceToHost));
  
  int atom2;
  CUDA_CHECK(cudaMemset(d, 0, sizeof(int)));
  count_atomic<<<100,256>>>(d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&atom2, d, sizeof atom2, cudaMemcpyDeviceToHost));
  
  printf("100 x 256 threads: plain increment = %5d (expected 25600)\n", racy2);
  printf("                   atomicAdd       = %5d (expected 25600)\n", atom2);

  // Tickets: every thread receives a different old value.
  const int blocks = 100, threads = 10, total = blocks * threads;
  int *d_taken;
  CUDA_CHECK(cudaMalloc(&d_taken, total * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_taken, 0, total * sizeof(int)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(int)));
  tickets<<<blocks, threads>>>(d, d_taken);
  CUDA_CHECK(cudaGetLastError());
  std::vector<int> taken(total);
  CUDA_CHECK(cudaMemcpy(taken.data(), d_taken, total * sizeof(int), cudaMemcpyDeviceToHost));
  bool distinct = true;
  for (int i = 0; i < total; ++i) distinct = distinct && taken[i] == 1;
  printf("tickets          : every number 0..%d was received by exactly one thread: %s\n",
         total - 1, distinct ? "yes" : "NO");

  cudaFree(d);  cudaFree(d_taken);
  return report("lost_update", atom == 1000 && atom2 == 25600 && distinct);
}
