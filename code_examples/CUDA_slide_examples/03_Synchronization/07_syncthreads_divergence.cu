// Slide 19 "What __syncthreads() does not do" (03_Synchronization).
//
// We have an array, `out`, initialized with the identifiers of all threads.
// The first part of the code increments the elements at even positions by two
// and decrements the elements at odd positions by two.
//
// For demonstration purposes, lines 20-30 perform these operations using only
// the first warp of each block, with each half of the warp handling the
// corresponding half of the array. Meanwhile, lines 35-39 perform the same
// operations with all even-numbered threads incrementing their corresponding
// elements and all odd-numbered threads decrementing theirs.
//
// This part of the code illustrates a potential issue at line 30 if we force
// some threads in the block to synchronize. You can compile the code with
// -DBUG:
//   nvcc -DBUG file.cu -o executable
// Depending on the implementation and execution conditions, you may observe
// unexpected behavior or a hang.
//
// The second part of the code swaps the values in the upper half of the array
// with those in the lower half. The operation is performed twice to restore
// the array to its original state.
//
// Although the code may appear to work correctly, __syncthreads() only
// synchronizes threads within the same block; it does not synchronize threads
// across different blocks. Therefore, when an artificial delay is introduced,
// the lack of inter-block synchronization can cause incorrect results or
// race conditions. You can expose this problem compiling with -DBUG2.
#include "common.h"

__device__ int tmp[256];

__global__ void exchange(int *out) {
  int t = threadIdx.x;
  int global_t = t + 256 * blockIdx.x;

  out[global_t] = global_t;
  
  __syncthreads();
  
  if (threadIdx.x < 16){  
     if (threadIdx.x % 2 == 0){
        for (int i = 0; i < 16; ++i)  out[global_t + 16*i] += 1;
     }else{
        for (int i = 0; i < 16; ++i)  out[global_t + 16*i] -= 1;
     }  
     #ifdef BUG
     __syncthreads();
     #endif
  }

  __syncthreads();

  if (threadIdx.x % 2 == 0){
     out[global_t ] += 1;
  }else{
     out[global_t] -= 1;
  }


  __syncthreads();
  
  long long t0 = clock64(); while (clock64() - t0 < 1000000) {}
 
// end of the first part, beginning of the second one  

  if (blockIdx.x == 1){
#ifdef BUG2
    t0 = clock64(); while (clock64() - t0 < 1000000000) {}
#endif
    tmp[threadIdx.x] = out[global_t];
    out[global_t] = out[global_t - 256];
  }

  __syncthreads();
  
  if (blockIdx.x == 0){
     out[global_t] = tmp[threadIdx.x];
  }
  
  __syncthreads();
  
  if (blockIdx.x == 0){
     tmp[threadIdx.x] = out[global_t];
     out[global_t] = out[global_t + 256];
  }

  __syncthreads();
  
  if (blockIdx.x == 1){
     out[global_t] = tmp[threadIdx.x];
  }
}

int main() {
  int *d;
  
  CUDA_CHECK(cudaMalloc(&d, 2 * 256 * sizeof(int)));
  
  exchange<<<2, 256>>>(d);
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> h(2 * 256);
  CUDA_CHECK(cudaMemcpy(h.data(), d, 2 * 256 * sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d);

  bool ok = true;
  for (int t = 0; t < 512; ++t) {
    printf("%d-%d ",t,h[t]);
    ok = ok && h[t] == ((t % 2 == 0) ? (t+2) : t-2); 
  }
  if (ok == false){ 
     printf("BUG build: the result is undefined; run it under \n$> compute-sanitizer --tool synccheck cuda_bynary.\n");
  } 
  return report("syncthreads_divergence", ok);
  return EXIT_SUCCESS;
}
