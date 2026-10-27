#include <stdio.h>
#include <cuda_runtime.h>

// Kernel: every thread prints its own message
__global__ void hello_kernel(void)
{
    int id_global = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello world form thread %d of block %d (global id %d)\n",
           threadIdx.x, blockIdx.x, id_global);
}

int main(void)
{
    const int num_blocks = 2;
    const int threads_per_block = 4;

    // kernel launch: 2 blocks x 4 threads = 8 threads in total
    hello_kernel<<<num_blocks, threads_per_block>>>();

    // icheck errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error al lanzar el kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Execution error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Hello world from the host: kernel finished\n");
    return 0;
}

