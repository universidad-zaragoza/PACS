// Slides 18-19 "NVTX Ranges" and "Command-Line Profiling" (04_Timing).
// Names the phases of a program so that they appear as labelled ranges on the Nsight Systems timeline,
// and limits the profiled region with cudaProfilerStart / cudaProfilerStop.
//
//   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi -o report ./09_nvtx_ranges
//   nsys stats report.nsys-rep
//   ncu --profile-from-start off --metrics gpu__time_duration.sum ./09_nvtx_ranges
//
// Two ways to write the same thing, both used below:
//   - the C API of the slide:   nvtxRangePushA("upload"); ... nvtxRangePop();   (also nvtxMarkA for an instant)
//   - the C++ API, RAII:        nvtx3::scoped_range r{"compute"};               (the range ends with the scope,
//                               even on an early return)
// Both come from the NVTX3 headers shipped with recent toolkits: header-only, no extra library to link.
// The exact flag names of nsys and ncu are written from memory: check  nsys profile --help  and  ncu --help.
#include "common.h"
#include </opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64/nvtx/include/nvtx3/nvToolsExt.h>
#include </opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64/nvtx/include/nvtx3/nvtx3.hpp>
#include <cuda_profiler_api.h>

int main() {
  const int n = 1 << 20, iters = 20000, block = 256, grid = (n + block - 1) / block;
  const size_t bytes = n * sizeof(float);

  float *h_in, *h_out, *d_in, *d_out;
  CUDA_CHECK(cudaMallocHost(&h_in, bytes));
  CUDA_CHECK(cudaMallocHost(&h_out, bytes));
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 8);

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  { nvtx3::scoped_range r{"warm-up"};                    // outside the profiled window
    work<<<grid, block, 0, s>>>(d_in, d_out, n, 1);
    CUDA_CHECK(cudaStreamSynchronize(s)); }

  CUDA_CHECK(cudaProfilerStart());                       // the profiler window opens here
  for (int step = 0; step < 3; ++step) {
    nvtxMarkA("step begins");                            // an instant, drawn as a flag on the timeline
    nvtxRangePushA("upload");                            // C API: push ... pop
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, s));
    nvtxRangePop();
    { nvtx3::scoped_range r{"compute"};                  // C++ API: the destructor pops
      work<<<grid, block, 0, s>>>(d_in, d_out, n, iters);
      CUDA_CHECK(cudaGetLastError()); }
    nvtxRangePushA("download");
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, s));
    nvtxRangePop();
    { nvtx3::scoped_range r{"wait"};
      CUDA_CHECK(cudaStreamSynchronize(s)); }
  }
  CUDA_CHECK(cudaProfilerStop());                        // ... and closes here

  bool ok = true;
  for (int i = 0; i < 8; ++i) ok = ok && close_enough(h_out[i], work_ref((float)i, iters));

  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  return report("nvtx_ranges", ok);
}
