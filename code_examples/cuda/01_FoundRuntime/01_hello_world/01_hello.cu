// Part 1, slide "Your First Kernel".
#include <cstdio>

__global__ void hello() {
  printf("block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
  hello<<<2, 4>>>();         // 2 blocks of 4 threads
  cudaDeviceSynchronize();   // wait, and flush printf
  return 0;
}
