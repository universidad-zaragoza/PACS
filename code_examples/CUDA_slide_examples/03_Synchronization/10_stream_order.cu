// Slide 23 "Streams: Ordered Queues of Operations" (03_Synchronization): kernel1 -> kernel2 -> memcpy -> kernel3 in ONE stream.
// The copy sees the values written by kernel2 and not yet changed by kernel3, because the stream keeps the order.
#include "common.h"

__global__ void kernel1(int *x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = i; }
__global__ void kernel2(int *x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] *= 2; }
__global__ void kernel3(int *x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += 1; }

int main() {
  const int n = 1 << 16, block = 256, grid = (n + block - 1) / block;
  int *d, *h;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&h, n * sizeof(int)));       // pinned: the copy can be truly asynchronous

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  kernel1<<<grid, block, 0, stream>>>(d, n);              // 1
  kernel2<<<grid, block, 0, stream>>>(d, n);              // 2
  CUDA_CHECK(cudaMemcpyAsync(h, d, n * sizeof(int), cudaMemcpyDeviceToHost, stream));   // 3
  kernel3<<<grid, block, 0, stream>>>(d, n);              // 4
  // All four calls returned immediately. Now the CPU waits for this stream only.
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());

  bool ok = true;
  for (int i = 0; i < n; ++i) ok = ok && h[i] == 2 * i;   // copy made after kernel2, before kernel3

  int *h2 = (int *)malloc(n * sizeof(int));
  CUDA_CHECK(cudaMemcpy(h2, d, n * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) ok = ok && h2[i] == 2 * i + 1;   // kernel3 ran last
  free(h2);

  CUDA_CHECK(cudaStreamDestroy(stream));
  cudaFreeHost(h);  cudaFree(d);
  return report("stream_order", ok);
}
