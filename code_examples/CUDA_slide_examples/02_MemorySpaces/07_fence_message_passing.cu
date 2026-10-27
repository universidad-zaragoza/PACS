// "__threadfence(): Ordering, Not waiting" (02_MemorySpaces); the same slide appears in
// 
//
//   writer thread                 reader thread
//   data = 42;                    int f = flag;
//   __threadfence();              __threadfence();
//   flag = 1;                     int d = data;     // if f == 1 then d == 42
//
// A fence ORDERS the accesses of ONE thread. It does not make anybody wait. So the reader below does not
// spin: it samples (flag, data) a bounded number of times and checks the invariant "flag == 1 implies
// data == 42" every time it sees flag == 1. Spinning on a flag written by another block could deadlock
// the kernel if that block is not resident.
//
// Part 1: one kernel, one writer thread (block 0), many reader threads (the other blocks).
//         Count how many samples saw flag == 1 and how many of those saw a wrong data value (must be 0).
//         A litmus test like this cannot PROVE the fences are right (a missing fence may still give 0
//         violations on a given GPU), but it shows the pattern. Build with -DNO_FENCE to remove both fences.
// Part 2: the safe way to wait: the consumer is a LATER kernel in the same stream, so the kernel boundary
//         supplies the waiting and a plain read is enough.
//
// The other fences: __threadfence_block() orders for the threads of the same block;
// __threadfence_system() also orders for host threads and for other devices (peer-to-peer or managed memory).
//
// Note volatile: the loads and stores of data and flag must really happen, in this order, so the compiler
// may not keep them in registers or reorder them. Fences order the hardware; volatile controls the compiler.
#include "common.h"

__global__ void message_passing(volatile int *data, volatile int *flag,
                                int *seen, int *violations, int rounds) {
  if (blockIdx.x == 480) {
    if (threadIdx.x == 33) {                    // writer
      *data = 42;
#ifndef NO_FENCE
      __threadfence();                         // data before flag
#endif
      *flag = 1;
    }
  } else {                                     // readers: nobody waits, nobody spins
    for (int r = 0; r < rounds; ++r) {
      int f = *flag;
#ifndef NO_FENCE
      __threadfence();                         // flag before data
#endif
      int d = *data;
      if (f == 1) {
        atomicAdd(seen, 1); //if I saw flag equal 1, then data must be 42 following sequential consistency memory ordering (or realaxed with fences)
        if (d != 42) 
            atomicAdd(violations, 1);
      }
    }
  }
}

__global__ void producer(int *data, int *flag) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *data = 42;
    __threadfence();                           // the data write is observed before the flag write
    *flag = 1;
  }
}

__global__ void consumer(const int *data, const int *flag, int *out) {   // starts after producer has finished
  if (blockIdx.x == 0 && threadIdx.x == 0) *out = (*flag == 1) ? *data : -1;
}

int main() {
  int *d_data, *d_flag, *d_seen, *d_viol, *d_out;
  
  CUDA_CHECK(cudaMalloc(&d_data, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_seen, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_viol, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));

  // ---- Part 1: the litmus test -----------------------------------------------------------------
  CUDA_CHECK(cudaMemset(d_data, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_seen, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_viol, 0, sizeof(int)));
  
  message_passing<<<512, 64>>>(d_data, d_flag, d_seen, d_viol, 1000);
  
  CUDA_CHECK(cudaGetLastError());
  int seen = 0, viol = 0;
  CUDA_CHECK(cudaMemcpy(&seen, d_seen, sizeof seen, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&viol, d_viol, sizeof viol, cudaMemcpyDeviceToHost));
  printf("litmus: %d samples saw flag == 1, %d of them saw a wrong data value\n", seen, viol);
  printf("        (how many samples see flag == 1 depends on the timing of the blocks)\n");

  // ---- Part 2: the consumer is a later kernel --------------------------------------------------
  CUDA_CHECK(cudaMemset(d_data, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));
  producer<<<1, 32>>>(d_data, d_flag);
  consumer<<<1, 32>>>(d_data, d_flag, d_out);
  CUDA_CHECK(cudaGetLastError());
  int got = 0;
  CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof got, cudaMemcpyDeviceToHost));
  printf("consumer kernel read %d\n", got);

  cudaFree(d_data);  cudaFree(d_flag);  cudaFree(d_seen);  cudaFree(d_viol);  cudaFree(d_out);
#ifdef NO_FENCE
  printf("NO_FENCE build: the invariant is not guaranteed any more; violations are not an error here.\n");
  return report("fence_message_passing (no fences)", got == 42);
#else
  return report("fence_message_passing", viol == 0 && got == 42);
#endif
}
