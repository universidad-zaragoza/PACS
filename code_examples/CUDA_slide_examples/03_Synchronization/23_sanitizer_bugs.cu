// Slides 43 and 50 "Common Synchronization Bugs" and "Compute-sanitizer" (03_Synchronization), and the exercise
// "Remove the barrier" (slide 52).
//
// Four kernels, each one the home of one bug that one compute-sanitizer tool finds:
//
//   tool        build with      the bug                                          kernel
//   racecheck   -DBUG_RACE      a shared-memory race: a missing barrier          two_phase
//   synccheck   -DBUG_SYNC      a barrier in divergent code                      branch_barrier
//   memcheck    -DBUG_OOB       a write one element past the end of an array     fill
//   initcheck   -DBUG_UNINIT    a read of global memory nobody has written       sum_buffer
//
// The default build has none of the bugs: it runs clean and prints PASSED. To see the tools work:
//
//   nvcc -lineinfo -DBUG_RACE -o bug_race 23_sanitizer_bugs.cu
//   compute-sanitizer --tool racecheck ./bug_race
//
// (the same with the other macros and tools; compile with -lineinfo to get source lines in the reports).
// The results of a bug build are undefined, so a bug build prints a note and does not judge them.
#include "common.h"

constexpr int N = 256;

// --- racecheck: write, barrier, read ---------------------------------------------------------------
__global__ void two_phase(const int *in, int *out) {
  __shared__ int data[N];
  int tid = threadIdx.x;
  data[tid] = in[tid];                       // phase 1
#ifndef BUG_RACE
  __syncthreads();
#endif
  out[tid] = data[(tid + 1) % N];            // phase 2: reads what ANOTHER thread wrote
}

// --- synccheck: a barrier that every thread of the block must reach ----------------------------------
__global__ void branch_barrier(int *out) {
  __shared__ int s[N];
  int tid = threadIdx.x;
  s[tid] = tid;
#ifdef BUG_SYNC
  if (tid < N / 2)                           // BUG: half of the block never reaches the barrier
#endif
    __syncthreads();
  out[tid] = s[N - 1 - tid];
}

// --- memcheck: out-of-bounds access ------------------------------------------------------------------
__global__ void fill(int *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef BUG_OOB
  if (i <= n)                                // BUG: i == n writes p[n], one element past the end
#else
  if (i < n)
#endif
    p[i] = i;
}

// --- initcheck: read of uninitialized global memory ------------------------------------------------
__global__ void sum_buffer(const int *p, int n, long long *out) {
  long long s = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += p[i];
  atomicAdd((unsigned long long *)out, (unsigned long long)s);
}

int main() {
  bool ok = true;

  // two_phase
  std::vector<int> h_in(N), h_out(N);
  for (int i = 0; i < N; ++i) h_in[i] = 3 * i;
  int *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  two_phase<<<1, N>>>(d_in, d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) ok = ok && h_out[i] == h_in[(i + 1) % N];

  // branch_barrier
  branch_barrier<<<1, N>>>(d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) ok = ok && h_out[i] == N - 1 - i;

  // fill: the buffer has exactly n elements
  const int n = 1000;
  int *d_buf;
  CUDA_CHECK(cudaMalloc(&d_buf, n * sizeof(int)));
  fill<<<(n + 255) / 256 + 1, 256>>>(d_buf, n);                  // the grid covers more than n threads
  CUDA_CHECK(cudaGetLastError());
  std::vector<int> h_buf(n);
  CUDA_CHECK(cudaMemcpy(h_buf.data(), d_buf, n * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) ok = ok && h_buf[i] == i;

  // sum_buffer: d_buf is initialized by fill() above, so it is clean. The BUG_UNINIT build reads a new buffer.
  int *d_src = d_buf;
#ifdef BUG_UNINIT
  CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(int)));               // never written
#endif
  long long *d_sum;
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(long long)));
  CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(long long)));
  sum_buffer<<<1, 128>>>(d_src, n, d_sum);
  CUDA_CHECK(cudaGetLastError());
  long long sum = 0;
  CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof sum, cudaMemcpyDeviceToHost));
#ifndef BUG_UNINIT
  ok = ok && sum == (long long)n * (n - 1) / 2;
#endif

  cudaFree(d_in);  cudaFree(d_out);  cudaFree(d_buf);  cudaFree(d_sum);
#if defined(BUG_RACE) || defined(BUG_SYNC) || defined(BUG_OOB) || defined(BUG_UNINIT)
  printf("Bug build: the results are undefined and are not judged. Run it under compute-sanitizer.\n");
  return EXIT_SUCCESS;
#else
  return report("sanitizer_bugs (clean build)", ok);
#endif
}
