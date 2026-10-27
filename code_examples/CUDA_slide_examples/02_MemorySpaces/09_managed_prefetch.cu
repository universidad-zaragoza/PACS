// Slides 14-15 "Unified Memory (managed)" and "Managed: Where Do the Pages Live?" (02_MemorySpaces), and the
// exercise "Unified memory" (slide 25): add a prefetch to the managed example and compare.
//
//   step  code                              the pages are on
//   1     cudaMallocManaged(&x, bytes)      nowhere yet: placed on first touch
//   2     host loop: x[i] = 1.0f            the host
//   3     scale<<<...>>>(x, n)              moved to the GPU on demand, as the kernel touches them
//   4     cudaDeviceSynchronize()           the GPU
//   5     host reads x[0]                   moved back to the host
//
// Variant A runs exactly these steps. Variant B prefetches the pages to the GPU before the kernel, and back to
// the host afterwards, so the kernel does not stall on page faults ("hopefully": a prefetch is a hint).
// Compare the two with:   nsys profile --trace=cuda --cuda-um-gpu-page-faults=true ./09_managed_prefetch
// (the flag name is written from memory: check  nsys profile --help).
//
// The signature of cudaMemPrefetchAsync changed in CUDA 13: it takes a cudaMemLocation and flags. The
// code below selects the right call with CUDART_VERSION, so it builds with CUDA 12 and CUDA 13.
#include "common.h"

__global__ void scale(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= 2.0f;
}

static void prefetch_to_gpu(const void *x, size_t bytes, int dev, cudaStream_t s) {
#if CUDART_VERSION >= 13000
  cudaMemLocation gpu = {cudaMemLocationTypeDevice, dev};
  CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, gpu, 0, s));            // CUDA 13: location, flags, stream
#else
  CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, dev, s));               // CUDA 12: device, stream
#endif
}

static void prefetch_to_host(const void *x, size_t bytes, cudaStream_t s) {
#if CUDART_VERSION >= 13000
  cudaMemLocation cpu = {cudaMemLocationTypeHost, 0};               // the id is ignored for the host
  CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, cpu, 0, s));
#else
  CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, cudaCpuDeviceId, s));
#endif
}

// Steps 1 to 5 of the table. Returns the time of steps 3-4 (the kernel and its page faults) in ms.
static float run(bool prefetch, int dev, int n, bool &ok) {
  size_t bytes = (size_t)n * sizeof(float);
  int block = 256, grid = (n + block - 1) / block;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  float *x;
  CUDA_CHECK(cudaMallocManaged(&x, bytes));          // 1: nowhere yet
  for (int i = 0; i < n; ++i) 
      x[i] = 1.0f;           // 2: the pages are on the host

  GpuTimer t;
  t.start(stream);
  
  if (prefetch) 
      prefetch_to_gpu(x, bytes, dev, stream);   // before the kernel
  
  scale<<<grid, block, 0, stream>>>(x, n);                // 3: touches the pages
  CUDA_CHECK(cudaGetLastError());
  float ms = t.stop(stream);                              // 4: wait for the GPU

  if (prefetch) {                                         // optional: bring the pages back before the host loop
    prefetch_to_host(x, bytes, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  printf("  x[0] = %g\n", x[0]);                          // 5: the host reads
  for (int i = 0; i < n && ok; ++i) 
      ok = (x[i] == 2.0f);

  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return ms;
}

int main() {
  int dev = 0, concurrent = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  // 1 on Pascal or newer GPUs with a Linux driver: pages move on demand and prefetching is supported.
  CUDA_CHECK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev));
  printf("cudaDevAttrConcurrentManagedAccess = %d\n", concurrent);

  scale<<<1, 1>>>(nullptr, 0);                              // warm-up: loads the kernel, touches no memory
  CUDA_CHECK(cudaDeviceSynchronize());

  const int n = 1 << 22;                                    // 16 MiB
  bool ok = true;
  printf("A) on-demand migration only\n");
  
  float ms_a = run(false, dev, n, ok);
  float ms_b = -1.0f;
  
  if (concurrent) {
    printf("B) prefetch to the GPU before the kernel, back to the host after it\n");
    ms_b = run(true, dev, n, ok);
  } else {
    printf("B) skipped: this device or platform does not support concurrent managed access\n");
  }
  
  printf("kernel time with page faults : %.3f ms\n", ms_a);
  if (ms_b >= 0) printf("kernel time after prefetch   : %.3f ms\n", ms_b);
  return report("managed_prefetch", ok);
}
