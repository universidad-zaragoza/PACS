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

// Data parallelism: every thread adds 1 to ONE element of the array (in place)
__global__ void add_one(float *a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global thread index

    if (i < n)                                       // the last block may have extra threads
        a[i] += 1.0f;
}

int main(void)
{
    const int n = 1 << 20;                           // 1,048,576 elements
    const size_t bytes = n * sizeof(float);

    // Host memory
    float *h_a = (float *)malloc(bytes);
    if (!h_a) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < n; i++)
        h_a[i] = (float)i;

    // Device memory
    float *d_a;
    CHECK(cudaMalloc(&d_a, bytes));

    // Host -> device
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // Launch configuration: enough blocks to cover all n elements
    const int threads_per_block = 256;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    add_one<<<num_blocks, threads_per_block>>>(d_a, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Device -> host (overwrites h_a with the updated values)
    CHECK(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));

    // Verify: element i must now be i + 1
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_a[i] != (float)i + 1.0f)
            errors++;
    }

    printf("Launched %d blocks x %d threads for %d elements\n",
           num_blocks, threads_per_block, n);
    for (int i = 0; i < 5; i++)
        printf("a[%d] = %.1f\n", i, h_a[i]);
    printf(errors == 0 ? "Result correct\n" : "Result WRONG: %d mismatches\n", errors);

    CHECK(cudaFree(d_a));
    free(h_a);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
