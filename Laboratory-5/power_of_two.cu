#include <cuda_runtime.h>
#include <iostream>

__global__ void power_of_two_kernel(const float* input, float* output, int n) {

}

int main() {
    const int n = 1024;
    float h_input[n];
    float h_output[n];

    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i + 1);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaError_t err = cudaMalloc( /**/ );
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_input failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMalloc( /**/ );
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_output failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMemcpy( /**/ );
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy host->device failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    const int threads_per_block = ???;
    const int blocks = ???;

    power_of_two_kernel<<< ??? >>>( /**/ );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMemcpy( /**/ );
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy device->host failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Input[42] = " << h_input[42] << " -> Output[42] = " << h_output[42] << " -> " << (h_input[42] * h_input[42] == h_output[42] ? "PASSED" : "FAILED") << std::endl;
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
