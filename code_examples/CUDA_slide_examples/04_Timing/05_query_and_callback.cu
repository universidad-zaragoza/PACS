// Slide 13 "Waiting, Polling and Host Callbacks" (04_Timing).
// Part 1: cudaEventQuery lets the CPU do other work while the GPU runs, and
//         cudaLaunchHostFunc stamps the host clock at a chosen point of the stream.
// Part 2: the CPU time spent in cudaEventSynchronize with and without cudaEventBlockingSync.
#include "common.h"
#include <ctime>

// Runs on a driver thread when the stream reaches this point. CUDA API calls are not allowed here.
static void CUDART_CB on_done(void *user) {
  *(double *)user = now_ms();
}

int main() {
  const int n = 1 << 20, iters = 100000;
  const size_t bytes = n * sizeof(float);
  float *h_in = (float *)malloc(bytes), *h_out = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 8);
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const int block = 256, grid = (n + block - 1) / block;

  // ---- Part 1: poll with cudaEventQuery, stamp the host clock with a host function
  cudaEvent_t done;
  CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));   // a pure synchronization event

  double t_cb = 0.0;
  double t_launch = now_ms();
  work<<<grid, block, 0, s>>>(d_in, d_out, n, iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaLaunchHostFunc(s, on_done, &t_cb));
  CUDA_CHECK(cudaEventRecord(done, s));

  long polls = 0;
  double other_work = 0.0;
  cudaError_t q;
  while ((q = cudaEventQuery(done)) == cudaErrorNotReady) {   // not ready is not an error
    other_work += std::sqrt((double)++polls);                 // stand-in for useful CPU work
  }
  CUDA_CHECK(q);                                              // anything else is a real error
  double t_seen = now_ms();
  CUDA_CHECK(cudaStreamSynchronize(s));

  printf("polls while waiting       : %ld\n", polls);
  printf("host function ran at      : %8.3f ms after the launch\n", t_cb - t_launch);
  printf("query reported completion : %8.3f ms after the launch\n", t_seen - t_launch);

  // ---- Part 2: spinning versus blocking while waiting
  const unsigned modes[2] = {cudaEventDefault, cudaEventBlockingSync};
  const char *names[2] = {"default      ", "blocking sync"};
  bool waits_ok = true;
  for (int m = 0; m < 2; ++m) {
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreateWithFlags(&e, modes[m] | cudaEventDisableTiming));
    
    work<<<grid, block, 0, s>>>(d_in, d_out, n, iters);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaEventRecord(e, s));
    
    std::clock_t c0 = std::clock();
    double w0 = now_ms();
    
    CUDA_CHECK(cudaEventSynchronize(e));
    
    double wall = now_ms() - w0;
    double cpu = 1000.0 * (double)(std::clock() - c0) / CLOCKS_PER_SEC;
    
    printf("%s: waited %8.3f ms, CPU time used %8.3f ms\n", names[m], wall, cpu);
    waits_ok = waits_ok && wall >= 0.0 && cpu >= 0.0;
    CUDA_CHECK(cudaEventDestroy(e));
  }

  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  bool ok = waits_ok && t_cb >= t_launch && t_cb <= t_seen && other_work >= 0.0;
  for (int i = 0; i < 8; ++i) 
      ok = ok && close_enough(h_out[i], work_ref((float)i, iters));

  CUDA_CHECK(cudaEventDestroy(done));
  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  free(h_in);
  free(h_out);
  return report("query_and_callback", ok);
}
