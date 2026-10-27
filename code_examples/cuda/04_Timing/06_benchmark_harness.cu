// Slide 14 "Reliable Measurements" (04_Timing).
// A small benchmark helper: warm-up launches, then `reps` timed runs, each between two events.
// Prints the cold first launch next to the statistics of the warm runs.
// `launch(stream)` must enqueue the work to measure. Returns one time in ms per repetition.
#include "common.h"

// `launch(stream)` must enqueue the work to measure. Returns one time in ms per repetition.
template <class F>
std::vector<float> bench(F launch, int warmup, int reps, cudaStream_t s = 0) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  for (int i = 0; i < warmup; ++i) launch(s);            // not measured
  CUDA_CHECK(cudaStreamSynchronize(s));                  // start from an idle stream

  std::vector<float> ms(reps);
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaEventRecord(a, s));
    launch(s);
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ms[r], a, b));
  }
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms;
}

struct Stats { float min, median, mean, stddev; };

Stats summarize(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  double sum = 0, sq = 0;
  for (float x : v) { sum += x; sq += (double)x * x; }
  double mean = sum / v.size();
  double var = sq / v.size() - mean * mean;
  return {v.front(), v[v.size() / 2], (float)mean, (float)std::sqrt(var > 0 ? var : 0)};
}

int main() {
  const int n = 1 << 20, iters = 20000;
  const size_t bytes = n * sizeof(float);
  float *h_in = (float *)malloc(bytes), *h_out = (float *)malloc(bytes);
  
  for (int i = 0; i < n; ++i) 
      h_in[i] = (float)(i % 8);
  
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  const int block = 256, grid = (n + block - 1) / block;
  auto launch = [&](cudaStream_t s) {
    work<<<grid, block, 0, s>>>(d_in, d_out, n, iters);
    CUDA_CHECK(cudaGetLastError());
  };

  // Cold: the very first launch also pays for one-time work (for example loading the kernel code).
  std::vector<float> cold = bench(launch, /*warmup=*/0, /*reps=*/1);

  std::vector<float> warm = bench(launch, /*warmup=*/3, /*reps=*/20);
  Stats st = summarize(warm);

  printf("first launch : %8.3f ms\n", cold[0]);
  printf("warm runs    : min %.3f  median %.3f  mean %.3f  stddev %.3f ms (%zu runs)\n",
         st.min, st.median, st.mean, st.stddev, warm.size());

  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  bool ok = warm.size() == 20 && st.min <= st.median && st.min <= st.mean && st.stddev >= 0.0f;
  for (int i = 0; i < 8; ++i) 
      ok = ok && close_enough(h_out[i], work_ref((float)i, iters));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  free(h_in);
  free(h_out);
  return report("benchmark_harness", ok);
}
