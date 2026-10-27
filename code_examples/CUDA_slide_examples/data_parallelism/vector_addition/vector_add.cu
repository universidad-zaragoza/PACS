#include <stdio.h>
#include <stdlib.h>
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

// Data parallelism: every thread computes ONE element of the output vector
__global__ void vector_add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global thread index

    if (i < n)                                       // the last block may have extra threads
        c[i] = a[i] + b[i];
}

int main(void)
{
    const int n = 1 << 20;                           // 1,048,576 elements
    const size_t bytes = n * sizeof(float);

    // Host memory
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = 2.0f * i;
    }

    // Device memory
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    // Host -> device
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch configuration: enough blocks to cover all n elements
    const int threads_per_block = 256;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    vector_add<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Device -> host
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify against the CPU result
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i])
            errors++;
    }

    printf("Launched %d blocks x %d threads for %d elements\n",
           num_blocks, threads_per_block, n);
    for (int i = 0; i < 5; i++)
        printf("c[%d] = %.1f + %.1f = %.1f\n", i, h_a[i], h_b[i], h_c[i]);
    printf(errors == 0 ? "Result correct\n" : "Result WRONG: %d mismatches\n", errors);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

