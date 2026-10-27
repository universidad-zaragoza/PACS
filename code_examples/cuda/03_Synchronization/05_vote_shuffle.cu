// Slides 12-14 "__all_sync, __any_sync, __ballot_sync", "Warp-level primitives" and
// "Example: Voting and Broadcasting" (03_Synchronization).
#include "common.h"

__global__ void warp_demo(unsigned *ballot, int *any7, int *all_lt32, int *all_lt31, int *bcast, int *up) {
  const unsigned FULL = 0xffffffffu;      // all 32 lanes take part
  int lane = threadIdx.x;                 // one warp
  int v = lane * 10;

  unsigned even = __ballot_sync(FULL, lane % 2 == 0);   // bit i is set when lane i has an even index
  int a = __any_sync(FULL, lane == 7);                   // true: lane 7 satisfies it
  int b = __all_sync(FULL, lane < 32);                   // true: every lane satisfies it
  int c = __all_sync(FULL, lane < 31);                   // false: lane 31 does not
  int from5 = __shfl_sync(FULL, v, 5);                   // every lane receives lane 5's v
  int left = __shfl_up_sync(FULL, v, 1);                 // lane i receives the v of lane i-1 (lane 0 keeps its own)

  bcast[lane] = from5;
  up[lane] = left;
  if (lane == 0) { 
      *ballot = even; 
      *any7 = a; 
      *all_lt32 = b; 
      *all_lt31 = c; 
  }
}

int main() {
  unsigned *d_ballot;  
  int *d_a, *d_b, *d_c, *d_bc, *d_up;
  
  CUDA_CHECK(cudaMalloc(&d_ballot, sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_a, sizeof(int)));  CUDA_CHECK(cudaMalloc(&d_b, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_c, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_bc, 32 * sizeof(int)));  CUDA_CHECK(cudaMalloc(&d_up, 32 * sizeof(int)));

  warp_demo<<<1, 32>>>(d_ballot, d_a, d_b, d_c, d_bc, d_up);
  CUDA_CHECK(cudaGetLastError());

  unsigned ballot;  int a, b, c, bc[32], up[32];
  CUDA_CHECK(cudaMemcpy(&ballot, d_ballot, sizeof ballot, cudaMemcpyDeviceToHost));//ballot
  CUDA_CHECK(cudaMemcpy(&a, d_a, sizeof a, cudaMemcpyDeviceToHost));  //any7
  CUDA_CHECK(cudaMemcpy(&b, d_b, sizeof b, cudaMemcpyDeviceToHost));  //all_lt32
  CUDA_CHECK(cudaMemcpy(&c, d_c, sizeof c, cudaMemcpyDeviceToHost));  //all_lt31
  CUDA_CHECK(cudaMemcpy(bc, d_bc, sizeof bc, cudaMemcpyDeviceToHost));//bcast
  CUDA_CHECK(cudaMemcpy(up, d_up, sizeof up, cudaMemcpyDeviceToHost));//up

  printf("ballot = 0x%08x, any = %d, all(<32) = %d, all(<31) = %d (0:false and 1:true)\n", ballot, a, b, c);
  bool ok = ballot == 0x55555555u && a != 0 && b != 0 && c == 0;
  for (int i = 0; i < 32; ++i) {
    printf("B_cast('v in lane5 to lane%d) == %d. Lane %d received %d from lane %d\n", i, bc[i], i, up[i], (i ==0 ? 0 : (i-1))); 
    ok = ok && bc[i] == 50;
    ok = ok && up[i] == (i == 0 ? 0 : (i - 1) * 10);
  }
  return report("warp_vote_shuffle", ok);
}
