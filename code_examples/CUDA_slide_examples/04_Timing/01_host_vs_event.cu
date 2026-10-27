// Slides 4, 7, 9 and 10 "Timing can be hazardous", "Host Timers and Asynchronous Launches",
// "Timing a Kernel with Events" and "Host and CUDA event timing diagram" (04_Timing).
// One kernel launch, measured three ways:
//   (a) host clock around the launch only      -> the cost of *issuing* the work
//   (b) host clock around launch + synchronize -> the work plus launch and wake-up overhead
//   (c) CUDA events recorded in the stream     -> the time between two points of the GPU timeline
// ATTENTION: work kernel is definedd in common.h

// The aim of this examples is that an appropiate timing method is required for every feature of 
// the application you are focussing on 

#include "common.h"

int main() {
  const int n = 1 << 20, iters = 100000;
  const size_t bytes = n * sizeof(float);

  float *h_in = (float *)malloc(bytes), *h_out = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 8);
  
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  const int block = 256, grid = (n + block - 1) / block;

  work<<<grid, block>>>(d_in, d_out, n, iters);          // warm-up: not measured
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Creating timing Cuda Events. Regular cuda events serve as stamps in the stream that can be timed (timestamp) 
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  double t0 = now_ms();
  CUDA_CHECK(cudaEventRecord(e0));
  work<<<grid, block>>>(d_in, d_out, n, iters);
  CUDA_CHECK(cudaGetLastError());
  double t1 = now_ms();                                  // the launch call has returned
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));                  // wait until the kernel is done. To be sure event e1 has completed
  double t2 = now_ms();

  float ev_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ev_ms, e0, e1));      // calculating the time difference between events e1-e0

  // Three different time measurements. Only the last one measures kernel computation time 
  printf("(a) host clock, launch only      : %9.3f ms\n", t1 - t0);
  printf("(b) host clock, launch + wait    : %9.3f ms\n", t2 - t0);
  printf("(c) CUDA events                  : %9.3f ms\n", ev_ms);

  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < 8; ++i) ok = ok && close_enough(h_out[i], work_ref((float)i, iters));
  ok = ok && (t1 - t0) <= (t2 - t0) && ev_ms > 0.0f;

  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  free(h_in);
  free(h_out);
  return report("host_vs_event", ok);
}
