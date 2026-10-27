// Foundations Runtime API, slide "Worked Example: 2D Launch": 8 x 6 elements, blocks of 4 x 3.
#include "common.h"

//kernel
__global__ void index2d(int *out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) out[y * w + x] = y * w + x;   // row-major index
}

//host side
int main() {
  const int W = 8, H = 6;
  int h_out[W * H];
  int *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_out)));
  CUDA_CHECK(cudaMemset(d_out, 0xff, sizeof(h_out)));   // fill with -1

  dim3 block(4, 3);
  dim3 grid((W + block.x - 1) / block.x,
            (H + block.y - 1) / block.y);               // 2 x 2 blocks
  index2d<<<grid, block>>>(d_out, W, H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      printf("%3d", h_out[y * W + x]);
      ok = ok && (h_out[y * W + x] == y * W + x);
    }
    printf("\n");
  }

  // The highlighted thread of the slide.
  int tx = 2, ty = 1, bx = 0, by = 1;
  printf("threadIdx = (%d, %d), blockIdx = (%d, %d) -> x = %d, y = %d, h_out[x,y]=%d (x columns and y rows)\n", tx, ty, bx, by, bx * 4 + tx, by * 3 + ty,h_out[(by*3+ty)*W+(bx*4+tx)]);

  cudaFree(d_out);
  return report("launch_2d", ok);
}
