// Slides 8 and 11 "The Event API" and "Event Flags and Return Codes" (04_Timing).
//
// Part 1, the API of slide 8, in the order of use: cudaEventCreate, cudaEventRecord, cudaEventQuery,
//         cudaEventSynchronize, cudaEventElapsedTime, cudaEventDestroy.
// Part 2, the flags of slide 11 (cudaEventCreateWithFlags):
//         cudaEventDefault        timing enabled
//         cudaEventBlockingSync   cudaEventSynchronize() sleeps instead of spinning
//         cudaEventDisableTiming  no timestamp: cheaper, but not usable for elapsed time (just dependencies)
//         cudaEventInterprocess   shareable between processes, requires cudaEventDisableTiming
// Part 3, the return codes of cudaEventElapsedTime:
//         cudaSuccess                      ms holds the time
//         cudaErrorNotReady                recorded, not yet complete: call cudaEventSynchronize first
//         cudaErrorInvalidResourceHandle   never recorded, or created with cudaEventDisableTiming
//
// The three error codes are EXPECTED here, so they are not wrapped in CUDA_CHECK; cudaGetLastError() clears them.
#include "common.h"

__device__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void busy(unsigned long long ns) {                  // keeps the GPU busy for about `ns`
  unsigned long long t0 = now_ns();
  while (now_ns() - t0 < ns) { }
}

int main() {
  bool ok = true;
  const unsigned long long ten_ms = 10000000ULL; //number of nanoseconds in ten miliseconda 

  // ---- Part 1: the API --------------------------------------------------------------------------
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  CUDA_CHECK(cudaEventRecord(e0, 0));                           // marker in the stream (e0)
  busy<<<1, 1>>>(ten_ms);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(e1, 0));                           // second marker (e1)

  int polls = 0;
  cudaError_t q;

  while ((q = cudaEventQuery(e1)) == cudaErrorNotReady) //cudaErrorNotReady is a CUDA error code (34) indicating that an 
      //asynchronous operation being queried has not yet completed 
      ++polls;   // non-blocking: the CPU could work here
  
  if (q != cudaSuccess) { fprintf(stderr, "cudaEventQuery: %s\n", cudaGetErrorString(q)); return EXIT_FAILURE; }
  CUDA_CHECK(cudaEventSynchronize(e1));                         // already complete here: returns at once
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  printf("Part 1: elapsed = %.3f ms (kernel asked for 10 ms); cudaEventQuery said NotReady %d times\n", ms, polls);
  ok = ok && ms >= 9.0f;                                        // the kernel spins for 10 ms
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));

  // ---- Part 2: the flags -------------------------------------------------------------------------
  cudaEvent_t t0, t1, blocking0, blocking1, dep;
  CUDA_CHECK(cudaEventCreateWithFlags(&t0, cudaEventDefault));
  CUDA_CHECK(cudaEventCreateWithFlags(&t1, cudaEventDefault));
  CUDA_CHECK(cudaEventCreateWithFlags(&blocking0, cudaEventBlockingSync));
  CUDA_CHECK(cudaEventCreateWithFlags(&blocking1, cudaEventBlockingSync));
  CUDA_CHECK(cudaEventCreateWithFlags(&dep, cudaEventDisableTiming));

  CUDA_CHECK(cudaEventRecord(blocking0, 0));  // synchronizing event marker for busy
  busy<<<1, 1>>>(ten_ms);
  CUDA_CHECK(cudaEventRecord(blocking1, 0));  // second synchrnizing event marker for busp
  
  double h0 = now_ms();
  CUDA_CHECK(cudaEventSynchronize(blocking1));                  // the CPU thread sleeps while it waits
  double h1 = now_ms();

  CUDA_CHECK(cudaEventElapsedTime(&ms, blocking0, blocking1));  // blocking-sync events still have timestamps, processor can be 
                                                                // yield to other processes or threads
                                                                // In opposition to default CUDA events that forces an spin lock 
                                                                // spin-lock has better response time in exchange of resource consumption
  printf("Part 2: cudaEventBlockingSync event: waited %.3f ms on the host, GPU time %.3f ms\n", h1 - h0, ms);

  cudaEvent_t ipc;                                              // Interprocess requires DisableTiming
  cudaError_t e = cudaEventCreateWithFlags(&ipc, cudaEventInterprocess);
  printf("        cudaEventInterprocess alone: %s (the slide says it requires cudaEventDisableTiming)\n",
         e == cudaSuccess ? "accepted" : cudaGetErrorString(e));
  cudaGetLastError();                                           // clear if it failed
  if (e == cudaSuccess) cudaEventDestroy(ipc);

  // A DisableTiming event is still a perfectly good dependency: see 03_Synchronization/13_event_dependency.cu.
  cudaStream_t s2;
  CUDA_CHECK(cudaStreamCreate(&s2));
  busy<<<1, 1>>>(ten_ms);
  CUDA_CHECK(cudaEventRecord(dep, 0));                          // No-timing events are suposedly faster since they do not pay timestamp overhead
  CUDA_CHECK(cudaStreamWaitEvent(s2, dep, 0));                  // s2 waits on the GPU; the CPU does not
  CUDA_CHECK(cudaStreamSynchronize(s2));
  CUDA_CHECK(cudaStreamDestroy(s2));

  // ---- Part 3: return codes of cudaEventElapsedTime ----------------------------------------------
  // (a) NotReady: ask for the time before the second event has completed.
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  
  CUDA_CHECK(cudaEventRecord(a, 0));
  busy<<<1, 1>>>(ten_ms);
  CUDA_CHECK(cudaEventRecord(b, 0));
  
  cudaError_t r_notready = cudaEventElapsedTime(&ms, a, b);     // the GPU is still in the kernel
  cudaGetLastError();
  printf("Part 3: (a) before cudaEventSynchronize: %s\n", cudaGetErrorString(r_notready));
  ok = ok && (r_notready == cudaErrorNotReady || r_notready == cudaSuccess);   // Success only if the GPU was very fast
  CUDA_CHECK(cudaEventSynchronize(b));
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));                  // now it works
  printf("        (a) after  cudaEventSynchronize: cudaSuccess, %.3f ms\n", ms);

  // (b) InvalidResourceHandle: an event that was never recorded.
  cudaEvent_t never;
  CUDA_CHECK(cudaEventCreate(&never));
  cudaError_t r_never = cudaEventElapsedTime(&ms, a, never);
  cudaGetLastError();
  printf("        (b) one event never recorded : %s\n", cudaGetErrorString(r_never));
  ok = ok && r_never == cudaErrorInvalidResourceHandle;

  // (c) InvalidResourceHandle: an event created with cudaEventDisableTiming.
  cudaEvent_t nt0, nt1;
  CUDA_CHECK(cudaEventCreateWithFlags(&nt0, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&nt1, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(nt0, 0));
  CUDA_CHECK(cudaEventRecord(nt1, 0));
  CUDA_CHECK(cudaEventSynchronize(nt1));
  cudaError_t r_nt = cudaEventElapsedTime(&ms, nt0, nt1);
  cudaGetLastError();
  printf("        (c) created with DisableTiming: %s\n", cudaGetErrorString(r_nt));
  ok = ok && r_nt == cudaErrorInvalidResourceHandle;

  cudaEventDestroy(t0);  cudaEventDestroy(t1);  cudaEventDestroy(blocking0);  cudaEventDestroy(blocking1);
  cudaEventDestroy(dep); cudaEventDestroy(a);   cudaEventDestroy(b);          cudaEventDestroy(never);
  cudaEventDestroy(nt0); cudaEventDestroy(nt1);
  return report("event_api_flags", ok);
}
