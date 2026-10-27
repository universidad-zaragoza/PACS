// compile with: nvcc -arch=sm_86 01_memory_spaces.cu -o exe

// (02_MemorySpaces): one declaration of every kind of variable, grouped by the
// memory space where it lives. Each space writes one row of `out`; the host checks every row.
//
//   space     declared with                       lives in               seen by            lifetime
//   register  plain local variable                register file          one thread         thread
//   local     local array, runtime index          device memory (L1/L2)  one thread         thread
//   shared    __shared__, extern __shared__       on-chip, per SM        threads of a block block
//   global    __device__, cudaMalloc, malloc()    device memory          all threads, host  program / free()
//   constant  __constant__                        device memory + cache  all threads (read) program
//   param     kernel parameters                   constant bank          all threads (read) launch
//   managed   __managed__, cudaMallocManaged      migrates on demand     host and device    program / free
#include "common.h"
#include <vector>

constexpr int TILE = 8;               // constexpr scalar: an immediate in the code, no memory at all
constexpr int N    = TILE * TILE;     // threads in the one block we launch

struct Pair { int count; float scale; };   // the same struct type can be used in every space

enum Row { R_REG, R_LOCAL, R_HEAP, R_SHARED, R_DYNAMIC, R_GLOBAL, R_CONST, R_PARAM, R_MANAGED,
           R_POINTERS, ROWS };

// ---- file scope: variables that live as long as the program ------------------------------
__device__   int   g_counter = 0;                           // global memory: scalar
__device__   float g_table[8] = {1, 2, 3, 4, 5, 6, 7, 8};   // global memory: array with initializer
__device__   Pair  g_pair = {2, 0.25f};                     // global memory: struct

__constant__ float c_coeff[4];                // constant memory: array, filled by cudaMemcpyToSymbol
__constant__ int   c_n = 2;                   // constant memory: scalar with a constant initializer
__constant__ Pair  c_pair = {3, 0.5f};        // constant memory: struct

__managed__  float m_gain  = 1.5f;            // managed memory: host and device use it directly
__managed__  int   m_total = 0;

__global__ void spaces(const float *__restrict__ in,   // parameter that points to global memory
                                                       // __restrict_ no aliases expected
                                                       // const qualifier denotes that array elements can be cached at L1 (read-only cache) 
                       const float *um,                // parameter that points to managed memory
                       const float *zc,                // parameter that points to mapped host memory
                       float *out,                     // parameter that points to global memory
                       int n,                          // parameter: a plain value
                       const __grid_constant__ Pair cfg) {  // parameter whose address may be taken
  // Registers: automatic scalars and vector types. Each thread has its own copy (private to threads)
  int tx = threadIdx.x, ty = threadIdx.y;
  int i = ty * TILE + tx;
  float4 v = make_float4(2.0f * i, 1.0f, 2.0f, 3.0f);        // a vector type takes four registers
  out[R_REG * n + i] = v.x + v.y + v.z + v.w;                // 2i + 6

  // Local memory: a per-thread array (private) indexed with a runtime value cannot live in registers.
  // (A small array whose indices are all known at compile time may be kept in registers.)
  float scratch[8];
  for (int j = 0; j < 8; ++j) 
      scratch[j] = (float)(i + j);
  out[R_LOCAL * n + i] = scratch[i % 8];                     // i + i % 8

  // Device heap: malloc() in a kernel returns global memory. Free it in the kernel.
  float heap_sum = 0.0f;
  int *heap = (int *)malloc(4 * sizeof(int));
  if (heap) {
    for (int j = 0; j < 4; ++j) heap[j] = i + j;
    for (int j = 0; j < 4; ++j) heap_sum += heap[j];
    free(heap);
  }
  out[R_HEAP * n + i] = heap_sum;                            // 4i + 6

  // Shared memory: static declarations have a size known at compile time.
  __shared__ float tile[TILE][TILE];                         // static 2D array
  __shared__ Pair  s_pair;                                   // static struct
  __shared__ int   s_flag;                                   // static scalar
  // Dynamic shared memory: the size is the third launch parameter at kernel invocation (dyn_bytes). There is one array;
  // carve typed views from it.
  extern __shared__ float dyn[];
  float *dyn_f = dyn;                                        // n floats
  int   *dyn_i = (int *)(dyn + n);                           // then n ints

  if (i == 0) { s_pair = Pair{3, 0.5f}; s_flag = 10; }       // shared variables cannot have initializers
  tile[ty][tx] = (float)i;
  dyn_f[i] = 0.5f * i;
  dyn_i[i] = i;
  __syncthreads();
  out[R_SHARED * n + i]  = tile[tx][ty] + s_pair.count * s_pair.scale + s_flag;   // 8tx + ty + 11.5
  out[R_DYNAMIC * n + i] = dyn_f[(i + 1) % n] + dyn_i[n - 1 - i];                 // 0.5((i+1)%64) + 63 - i

  // Global memory: __device__ variables are shared by all threads of all blocks.
  atomicAdd(&g_counter, 1);                                  // every thread counts itself
  out[R_GLOBAL * n + i] = g_table[i % 8] + g_pair.count * g_pair.scale;           // i % 8 + 1.5

  // Constant memory: read-only in the kernel; all lanes of a warp read the same address in one go.
  out[R_CONST * n + i] = c_coeff[i % 4] * c_n + c_pair.count * c_pair.scale;      // 2(i % 4 + 1) + 1.5

  // Kernel parameters: read-only, also served by the constant bank.
  out[R_PARAM * n + i] = n + cfg.count * cfg.scale;                               // 64 + 4 * 0.5

  // Managed memory: the same variable is written by the host and by the device.
  atomicAdd(&m_total, 1);
  out[R_MANAGED * n + i] = m_gain * i;                                            // 3i (host set 3)

  // Pointers: the kernel cannot tell where the memory behind a pointer lives.
  // in = cudaMalloc, um = cudaMallocManaged, zc = mapped pinned host memory (read across the bus).
  // __ldg() reads through the read-only data cache.
  out[R_POINTERS * n + i] = __ldg(&in[i]) + um[i] + zc[i];                        // i + 1 + 10
}

// What each row of `out` must hold for thread i.
static float expected(int row, int i) {
  int tx = i % TILE, ty = i / TILE;
  switch (row) {
    case R_REG:      return 2.0f * i + 6.0f;
    case R_LOCAL:    return (float)(i + i % 8);
    case R_HEAP:     return 4.0f * i + 6.0f;
    case R_SHARED:   return (float)(TILE * tx + ty) + 1.5f + 10.0f;
    case R_DYNAMIC:  return 0.5f * ((i + 1) % N) + (float)(N - 1 - i);
    case R_GLOBAL:   return (float)(i % 8 + 1) + 0.5f;
    case R_CONST:    return 2.0f * (i % 4 + 1) + 1.5f;
    case R_PARAM:    return (float)N + 2.0f;
    case R_MANAGED:  return 3.0f * i;
    default:         return (float)i + 11.0f;          // R_POINTERS
  }
}

int main() {
  const size_t bytes = N * sizeof(float);
  static const char *names[ROWS] = {
    "registers (int, float4)",       "local memory (runtime-indexed array)",
    "device heap (malloc in kernel)", "shared, static (scalar, 2D, struct)",
    "shared, dynamic (extern)",       "global (__device__ scalar, array, struct)",
    "constant (__constant__)",        "kernel parameters",
    "managed (__managed__)",          "pointers (device, managed, mapped)"};

  // ---- host side: where the data lives before it reaches the GPU ---------------------------
  std::vector<float> h_out(ROWS * N);                        // pageable host memory
  float h_coeff[4] = {1, 2, 3, 4};                           // pageable host memory (the stack)
  float *h_pinned;                                           // pinned (page-locked) host memory
  CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
  float *h_zc, *d_zc;                                        // pinned host memory mapped into the
  CUDA_CHECK(cudaHostAlloc((void **)&h_zc, bytes, cudaHostAllocMapped));   // device address space
  CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_zc, h_zc, 0));
  float *um;                                                 // managed memory
  CUDA_CHECK(cudaMallocManaged(&um, bytes));
  float *d_in, *d_out;                                       // device global memory
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, ROWS * bytes));

  for (int i = 0; i < N; ++i) { h_pinned[i] = (float)i; h_zc[i] = 10.0f; um[i] = 1.0f; }
  CUDA_CHECK(cudaMemcpy(d_in, h_pinned, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(c_coeff, h_coeff, sizeof(h_coeff)));   // fill the constant array
  m_gain = 3.0f;                                             // managed: a plain assignment

  dim3 block(TILE, TILE);
  size_t dyn_bytes = N * sizeof(float) + N * sizeof(int);    // the extern __shared__ array
  Pair cfg = {4, 0.5f};
  spaces<<<1, block, dyn_bytes>>>(d_in, um, d_zc, d_out, N, cfg);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());       // the host may touch managed memory only after this

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, ROWS * bytes, cudaMemcpyDeviceToHost));
  int counter = 0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&counter, g_counter, sizeof(counter)));   // read a __device__ variable

  bool ok = true;
  for (int row = 0; row < ROWS; ++row) {
    bool row_ok = true;
    for (int i = 0; i < N && row_ok; ++i)
      row_ok = close_enough(h_out[row * N + i], expected(row, i));
    printf("%-44s %s\n", names[row], row_ok ? "OK" : "WRONG");
    ok = ok && row_ok;
  }
  printf("g_counter = %d, m_total = %d (both must be %d)\n", counter, m_total, N);
  ok = ok && counter == N && m_total == N;

  cudaFree(d_in);  cudaFree(d_out);  cudaFree(um);
  cudaFreeHost(h_pinned);  cudaFreeHost(h_zc);
  return report("memory_spaces", ok);
}

// Not shown: texture and surface objects (cudaTextureObject_t, cudaSurfaceObject_t), CUDA arrays
// (cudaMallocArray), pitched allocations (cudaMallocPitch), stream-ordered pools (cudaMallocAsync)
// and, on Hopper, distributed shared memory across a thread block cluster.
