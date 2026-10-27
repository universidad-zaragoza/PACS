// compilation hint: nvcc -arch=sm_86 or -arch=native 11_atomic_functions.cu -o exe (by default compiles with CC sm_52

// Slide 19 "Atomic Functions" (02_MemorySpaces; slide 36 of 03_Synchronization; same code in both folders).
// One kernel applies every function of the table to a variable in global memory. All 256 threads
// (4 blocks of 64) take part in each operation, so the final values do not depend on the order in which
// the threads arrive (except atomicExch and atomicCAS, which are checked with an invariant).
// Each function returns the OLD value; the examples that use it are atomicExch and atomicCAS.
//
//   function              operation       types used here
//   atomicAdd             old + val       int, unsigned, unsigned long long, float, double
//   atomicSub             old - val       int, unsigned
//   atomicExch            swap            int (float and unsigned long long exist too)
//   atomicMin, atomicMax  min, max        int, unsigned, unsigned long long
//   atomicAnd, Or, Xor    bitwise         int, unsigned, unsigned long long
//   atomicCAS             compare & swap  int (unsigned and unsigned long long exist too; see 13_atomic_cas.cu)
//   atomicInc, atomicDec  wrapping count  unsigned
//
// atomicAdd on double needs compute capability 6.0 or newer (the default of recent toolkits).
// They work on global and on shared memory; this example uses global memory (14_block_local_count.cu uses shared).
#include "common.h"

struct Results {
  int                add_i;
  unsigned           add_u;
  unsigned long long add_ull;
  float              add_f;
  double             add_d;
  int                sub_i;
  unsigned           sub_u;
  int                exch_i;      
  unsigned long long exch_old_sum;   // sum of the old values returned by atomicExch
  int                min_i, max_i;
  unsigned           min_u, max_u;
  unsigned long long min_ull, max_ull;
  unsigned           and_u, or_u, xor_u;
  unsigned long long or_ull;
  int                cas_winner;
  unsigned           inc_u, dec_u;
};

__global__ void apply_all(Results *r) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;               // 0 .. 255

  atomicAdd(&r->add_i, tid);                                     // 0 + 1 + ... + 255
  atomicAdd(&r->add_u, 1u);
  atomicAdd(&r->add_ull, 1ull << 32);  //forzamos utilizar la parte alta del tipoh
  atomicAdd(&r->add_f, 0.5f);
  atomicAdd(&r->add_d, 0.25);

  atomicSub(&r->sub_i, tid);
  atomicSub(&r->sub_u, 1u);

  int old = atomicExch(&r->exch_i, tid + 1);                     // everybody swaps in its own value
  atomicAdd(&r->exch_old_sum, (unsigned long long)old);

  atomicMin(&r->min_i, tid + 5);
  atomicMax(&r->max_i, tid + 5);
  atomicMin(&r->min_u, (unsigned)(tid + 5));
  atomicMax(&r->max_u, (unsigned)(tid + 5));
  atomicMin(&r->min_ull, (unsigned long long)(tid + 5));
  atomicMax(&r->max_ull, (unsigned long long)(tid + 5));

  atomicAnd(&r->and_u, ~(1u << (tid % 32)));                     // clears bit tid % 32
  atomicOr(&r->or_u, 1u << (tid % 32));                          // sets bit tid % 32
  atomicOr(&r->or_ull, 1ull << (tid % 64));
  atomicXor(&r->xor_u, (unsigned)(tid + 1));                     // xor of 1 .. 256

  atomicCAS(&r->cas_winner, -1, tid);                            // only the first CAS sees -1

  atomicInc(&r->inc_u, 9);                                       // 0 1 2 ... 9 0 1 ...: wraps after 9
  atomicDec(&r->dec_u, 9);                                       // 0 -> 9, then 8, 7 ... 0, 9, 8 ...
}

int main() {
  const int blocks = 4, threads = 64, N = blocks * threads;     // 256 threads

  Results h = {};
  h.sub_u = 1000000u;
  h.exch_i = 0;
  h.min_i = 1 << 30;           h.max_i = -(1 << 30);
  h.min_u = 0xffffffffu;       h.max_u = 0u;
  h.min_ull = ~0ull;           h.max_ull = 0ull;
  h.and_u = 0xffffffffu;       h.or_u = 0u;   h.or_ull = 0ull;   h.xor_u = 0u;
  h.cas_winner = -1;
  h.inc_u = 0u;                h.dec_u = 0u;

  Results *d;
  CUDA_CHECK(cudaMalloc(&d, sizeof(Results)));
  CUDA_CHECK(cudaMemcpy(d, &h, sizeof h, cudaMemcpyHostToDevice));
  apply_all<<<blocks, threads>>>(d);
  CUDA_CHECK(cudaGetLastError());
  Results g;
  CUDA_CHECK(cudaMemcpy(&g, d, sizeof g, cudaMemcpyDeviceToHost));

  bool ok = true;
  auto check = [&](const char *name, bool good) {
    printf("  %-34s %s\n", name, good ? "ok" : "WRONG");
    ok = ok && good;
  };
  check("atomicAdd int (sum of ids)",        g.add_i == N * (N - 1) / 2);
  check("atomicAdd unsigned",                g.add_u == (unsigned)N);
  check("atomicAdd unsigned long long",      g.add_ull == ((unsigned long long)N << 32));
  check("atomicAdd float",                   g.add_f == 0.5f * N);
  check("atomicAdd double",                  g.add_d == 0.25 * N);
  check("atomicSub int",                     g.sub_i == -(N * (N - 1) / 2));
  check("atomicSub unsigned",                g.sub_u == 1000000u - N);
  // atomicExch: the values that went in (initial 0 and 1..N) are the values that came out
  // (N old values and the final one). Which thread got which is not defined.
  check("atomicExch (nothing lost)",         g.exch_old_sum + g.exch_i == (unsigned long long)N * (N + 1) / 2);
  check("atomicMin / atomicMax int",         g.min_i == 5 && g.max_i == N + 4);
  check("atomicMin / atomicMax unsigned",    g.min_u == 5u && g.max_u == (unsigned)(N + 4));
  check("atomicMin / atomicMax 64-bit",      g.min_ull == 5ull && g.max_ull == (unsigned long long)(N + 4));
  check("atomicAnd (all bits cleared)",      g.and_u == 0u);
  check("atomicOr  (all bits set)",          g.or_u == 0xffffffffu);
  check("atomicOr  64-bit (all bits set)",   g.or_ull == ~0ull);
  check("atomicXor (xor of 1..256 = 256)",   g.xor_u == 256u);
  check("atomicCAS (one winner)",            g.cas_winner >= 0 && g.cas_winner < N);
  check("atomicInc (256 mod 10 = 6)",        g.inc_u == 6u);
  check("atomicDec (10 - 256 mod 10 = 4)",   g.dec_u == 4u);

  cudaFree(d);
  return report("atomic_functions", ok);
}
