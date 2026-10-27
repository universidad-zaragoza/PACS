#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Abort with a readable message if a CUDA call fails
#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err__));        \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// SAXPY (Single-precision A*X Plus Y): y = alpha * x + y
// Data parallelism: every thread updates ONE element of y
__global__ void saxpy(int n, float alpha, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global thread index

    if (i < n)                                       // the last block may have extra threads
        y[i] = alpha * x[i] + y[i];
}

int main(void)
{
    const int n = 1 << 20;                           // 1,048,576 elements
    const size_t bytes = n * sizeof(float);
    const float alpha = 2.0f;

    // Host memory: x, original y, and the result coming back from the GPU
    float *h_x   = (float *)malloc(bytes);
    float *h_y   = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    if (!h_x || !h_y || !h_out) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < n; i++) {
        h_x[i] = (float)(i % 100);
        h_y[i] = (float)(i % 10);
    }

    // Device memory
    float *d_x, *d_y;
    CHECK(cudaMalloc(&d_x, bytes));
    CHECK(cudaMalloc(&d_y, bytes));

    // Host -> device
    CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // Launch configuration: enough blocks to cover all n elements
    const int threads_per_block = 256;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    saxpy<<<num_blocks, threads_per_block>>>(n, alpha, d_x, d_y);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Device -> host
    CHECK(cudaMemcpy(h_out, d_y, bytes, cudaMemcpyDeviceToHost));

    // Verify against the CPU result (small tolerance for floating point)
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = alpha * h_x[i] + h_y[i];
        if (fabsf(h_out[i] - expected) > 1e-5f * fabsf(expected) + 1e-6f)
            errors++;
    }

    printf("Launched %d blocks x %d threads for %d elements\n",
           num_blocks, threads_per_block, n);
    for (int i = 0; i < 5; i++)
        printf("y[%d] = %.1f * %.1f + %.1f = %.1f\n",
               i, alpha, h_x[i], h_y[i], h_out[i]);
    printf(errors == 0 ? "Result correct\n" : "Result WRONG: %d mismatches\n", errors);

    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    free(h_x);
    free(h_y);
    free(h_out);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

