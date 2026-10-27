// Slide 21 "atomicCAS(): Compare and Swap" (02_MemorySpaces) and the exercise "Custom atomic" (slide 25):
// write atomicMinFloat with atomicCAS and test it against a CPU result.
// (03_Synchronization has its own CAS examples: 18_cas_election.cu, 19_cas_retry_loop.cu, 21_cas_lock.cu.)
//
// Part 1: the meaning of atomicCAS, written as ordinary host code. On the GPU the whole body is ONE atomic step.
//         The slide calls the function atomicCAS; here it is cas_model so that it does not clash with the built-in.
// Part 2: a new atomic operation built from atomicCAS: atomicMinFloat (atomicMin has no float version).
//         Read, compute, try to swap; if somebody changed the value in between (assumed != old), retry.
// Part 3: electing one thread: only the first CAS sees -1.
#include "common.h"

// ---- Part 1 -----------------------------------------------------------------------------------
static int cas_model(int *address, int compare, int val) {
  int old = *address;
  if (old == compare)
    *address = val;
  return old;
}

// ---- Part 2 -----------------------------------------------------------------------------------
__device__ float atomicMinFloat(float *addr, float val) {
  int *a = (int *)addr;                                    // atomicCAS compares the bit patterns of ints
  int old = *a, assumed;
  do {
    assumed = old;
    if (__int_as_float(assumed) <= val) break;             // the stored value is already small enough
    old = atomicCAS(a, assumed, __float_as_int(val));      // try to replace what we saw
  } while (assumed != old);                                // somebody was faster: loop with the new value
  return __int_as_float(old);
}

__global__ void min_kernel(const float *x, int n, float *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicMinFloat(result, x[i]);
}

static bool check_min(const char *name, const std::vector<float> &h) {
  const int n = (int)h.size();
  float *d_x, *d_r;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  float init = INFINITY;                                   // larger than every input
  CUDA_CHECK(cudaMemcpy(d_r, &init, sizeof(float), cudaMemcpyHostToDevice));
  min_kernel<<<(n + 255) / 256, 256>>>(d_x, n, d_r);
  CUDA_CHECK(cudaGetLastError());
  float got = 0.0f;
  CUDA_CHECK(cudaMemcpy(&got, d_r, sizeof(float), cudaMemcpyDeviceToHost));
  float want = *std::min_element(h.begin(), h.end());
  printf("  %s: min = %g (expected %g)\n", name, got, want);
  cudaFree(d_x);  cudaFree(d_r);
  return got == want;                                      // a minimum returns one of the inputs: exact
}

// ---- Part 3 -----------------------------------------------------------------------------------
__global__ void elect(int *winner, int *winners_seen) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (atomicCAS(winner, -1, id) == -1)                     // old == compare: this thread did the swap
    atomicAdd(winners_seen, 1);
}

int main() {
  bool ok = true;

  printf("Part 1: the meaning of compare-and-swap\n");
  int v = 5;
  int r1 = cas_model(&v, 5, 9);        // equal: swaps, returns the old value 5
  int r2 = cas_model(&v, 5, 7);        // v is 9 now, not 5: nothing changes, returns 9
  printf("  cas(5 -> 9) returned %d, then cas(5 -> 7) returned %d; v = %d\n", r1, r2, v);
  ok = ok && r1 == 5 && r2 == 9 && v == 9;

  printf("Part 2: atomicMinFloat built from atomicCAS\n");
  const int n = (1 << 18) + 3;
  std::vector<float> h(n);
  unsigned s = 987654u;
  for (int i = 0; i < n; ++i) {                            // pseudo-random values in [-1000, 1000)
    s = s * 1664525u + 1013904223u;
    h[i] = ((s >> 8) % 2000000u) / 1000.0f - 1000.0f;
  }
  ok = check_min("random values    ", h) && ok;
  for (int i = 0; i < n; ++i) h[i] = (float)(n - i);       // descending: every thread tries to be the new minimum
  ok = check_min("descending values", h) && ok;

  printf("Part 3: electing one thread\n");
  int *d_winner, *d_seen;
  CUDA_CHECK(cudaMalloc(&d_winner, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_seen, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_winner, 0xFF, sizeof(int)));     // every byte 0xFF = the int -1
  CUDA_CHECK(cudaMemset(d_seen, 0, sizeof(int)));
  elect<<<4, 256>>>(d_winner, d_seen);
  CUDA_CHECK(cudaGetLastError());
  int winner = -2, seen = -2;
  CUDA_CHECK(cudaMemcpy(&winner, d_winner, sizeof winner, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&seen, d_seen, sizeof seen, cudaMemcpyDeviceToHost));
  printf("  winner = %d, threads that saw -1: %d (must be 1)\n", winner, seen);
  ok = ok && seen == 1 && winner >= 0 && winner < 1024;
  cudaFree(d_winner);  cudaFree(d_seen);

  return report("atomic_cas", ok);
}
