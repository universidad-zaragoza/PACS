// Slide 21 "The kernel boundary is the global barrier", speaker notes (03_Synchronization):
// "Cooperative launch is the exception: grid.sync() gives a grid barrier, but only if the whole grid fits on the
// GPU at once." (this has not been covered in deatail in the course)
//
// Phase 1: every thread doubles its element.
// grid.sync()
// Phase 2: every thread reads an element written by a thread of ANOTHER block (the next block's) and adds 1.
//          That read is correct only because every block has finished phase 1.
//
// Rules:
//   - launch with cudaLaunchCooperativeKernel (an ordinary launch with grid.sync() is an error),
//   - the grid must be resident: at most (blocks per SM) x (SMs) blocks. Size it with the occupancy API,
//   - the device must support cooperative launch (attribute cudaDevAttrCooperativeLaunch).
// Older toolkits needed  nvcc -rdc=true  for grid.sync(); the Makefile of this folder passes it for this file.
#include <cooperative_groups.h>
#include "common.h"
namespace cg = cooperative_groups;

__global__ void two_phases(const float *in, float *data, float *out, int N) {
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Phase 1: process data
  if (tid < N) data[tid] = in[tid] * 2.0f;

  // Grid-wide barrier
  grid.sync();

  // Phase 2: all grid threads may now rely on Phase 1 having completed.
  // The element read here was written by a thread of the next block.
  if (tid < N) out[tid] = data[(tid + blockDim.x) % N] + 1.0f;
}

int main() {
  int dev = 0, supported = 0, numSMs = 0, blocksPerSM = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, dev));
  if (!supported) {
    printf("This device does not support cooperative launch: nothing to run.\n");
    printf("two_phases: SKIPPED\n");
    return EXIT_SUCCESS;
  }
  CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));

  const int block = 256;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, two_phases, block, 0));
  const int grid = numSMs * blocksPerSM;                  // the largest grid that is resident all at once
  const int N = grid * block;
  printf("%d SMs x %d blocks per SM = %d blocks of %d threads (%d elements)\n", numSMs, blocksPerSM, grid, block, N);

  std::vector<float> h_in(N), h_out(N);
  for (int i = 0; i < N; ++i) h_in[i] = (float)(i % 1000);
  float *d_in, *d_data, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  int n = N;
  void *args[] = {&d_in, &d_data, &d_out, &n};
  CUDA_CHECK(cudaLaunchCooperativeKernel((void *)two_phases, dim3(grid), dim3(block), args, 0, 0));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < N && ok; ++i) ok = (h_out[i] == h_in[(i + block) % N] * 2.0f + 1.0f);

  cudaFree(d_in);  cudaFree(d_data);  cudaFree(d_out);
  return report("grid_sync_cooperative", ok);
}
