// Part 1, slide "Discovering Devices". Also prints the CUDA versions in use.
#include "common.h"

int main() {
  int rt = 0, drv = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  CUDA_CHECK(cudaDriverGetVersion(&drv));
  printf("CUDA runtime %d.%d, driver supports up to %d.%d\n", rt / 1000, (rt % 1000) / 10, drv / 1000, (drv % 1000) / 10);

  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  if (n == 0) { fprintf(stderr, "no CUDA device found\n"); return 1; }

  for (int d = 0; d < n; ++d) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, d));
    printf("%d: %s, CC %d.%d, %d SMs, %.1f GB\n", d, p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem / 1e9);

    // Other useful fields.
    printf("   maxThreadsPerBlock %d, sharedMemPerBlock %zu B, warpSize %d\n", p.maxThreadsPerBlock, p.sharedMemPerBlock, p.warpSize);

    // The clock rate is no longer a cudaDeviceProp member in CUDA 13.
    // Ask for it as an attribute instead; this works in CUDA 12 and 13.
    int khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, d));
    printf("   clock %.2f GHz\n", khz / 1e6);
  }
  CUDA_CHECK(cudaSetDevice(0));
  return 0;
}
