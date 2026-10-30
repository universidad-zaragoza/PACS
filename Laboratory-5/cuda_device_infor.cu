// device_query_exercise.cu
// Lab exercise: query the properties of every CUDA-capable GPU in the system.
// Build: nvcc -o device_query device_query_exercise.cu
//
// Your task: complete every line marked TODO with the right CUDA runtime call.
// The file compiles as it is, but prints zeros until the calls are filled in.
//
// Useful reference: CUDA Runtime API documentation, "Device Management" and
// "Version Management" sections.

// https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/apis.html

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// Checks the result of a CUDA call and exits with a clear message on failure.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " (" #call "): " << cudaGetErrorString(err_) << '\n'; \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// Converts the CUDA version format (e.g. 12060) to "12.6".
static void printVersion(const char* label, int v) {
    std::cout << label << v / 1000 << "." << (v % 1000) / 10 << '\n';
}

int main() {
    // --- Driver and runtime versions -----------------------------------
    int driverVersion = 0, runtimeVersion = 0;

    // TODO 1: get the version of the installed CUDA driver into driverVersion.
    //         Wrap the call in CUDA_CHECK( ... ).

    // TODO 2: get the version of the CUDA runtime into runtimeVersion.
    //         Wrap the call in CUDA_CHECK( ... ).
    // https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART____VERSION.html


    printVersion("CUDA driver version:  ", driverVersion);
    printVersion("CUDA runtime version: ", runtimeVersion);

    // A driver older than the runtime is a common cause of kernels failing to launch.
    if (driverVersion < runtimeVersion) {
        std::cerr << "Warning: the driver is older than the runtime; "
                     "kernels may fail to run.\n";
    }

    // --- Number of devices ---------------------------------------------
    int deviceCount = 0;

    // TODO 3: get the number of CUDA devices into deviceCount and store the
    //         returned error code in err.
    // https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html#group__cudart__device_1gaf639fa9dfe6f65432869c52b5797f8ce
    cudaError_t err = cudaSuccess;  // <- replace cudaSuccess with the call

    if (err == cudaErrorNoDevice || (err == cudaSuccess && deviceCount == 0)) {
        std::cout << "\nNo CUDA-capable GPU found.\n";
        return 0;
    }
    if (err != cudaSuccess) {
        // TODO 4: print a readable description of err instead of "?".
        // https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__ERROR.html#group__cudart__error_1gad793fa381e73475eeea89f9afbf89f2e
        std::cerr << "Device count query failed: " << "?" << '\n';
        return 1;
    }

    std::cout << "\nNumber of CUDA devices: " << deviceCount << '\n';

    const double KiB = 1024.0;
    const double MiB = 1024.0 * 1024.0;

    for (int i = 0; i < deviceCount; ++i) {
        // --- Device properties -----------------------------------------
        cudaDeviceProp prop{};

        // TODO 5: fill prop with the properties of device i and store the
        //         returned error code in err.
        // https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html#group__cudart__device_1ga1bf9d625a931d657e08db2b4391170f0

        // You can have an overview of available cudaDeviceProp fields at 
        // https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaDeviceProp.html#_CPPv414cudaDeviceProp
        err = cudaSuccess;  // <- replace cudaSuccess with the call

        if (err != cudaSuccess) {
            std::cerr << "Property query failed for device " << i
                      << ": " << cudaGetErrorString(err) << '\n';
            continue;  // Skip this device but keep querying the others.
        }

        // Clock rates are not read from properties: clockRate and memoryClockRate
        // are deprecated in cudaDeviceProp since CUDA 12 and removed in CUDA 13.
        // Read them as device attributes instead. Both are returned in kHz.
        // https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html#group__cudart__device_1ga3559b8cc346511d423b5a6def5fbd3e0
        // list of available attributes:
        // https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html#_CPPv414cudaDeviceAttr

        int clockKHz = 0, memClockKHz = 0;

        // TODO 6: read the GPU clock rate of device i into clockKHz
        //         (attribute cudaDevAttrClockRate).

        // TODO 7: read the memory clock rate of device i into memClockKHz
        //         (attribute cudaDevAttrMemoryClockRate).

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "\n=== Device " << i << ": " << prop.name << " ===\n";
       
        // You can have an overview of available cudaDeviceProp fields at 
        // https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaDeviceProp.html#_CPPv414cudaDeviceProp
        // General information
        std::cout << "General\n";
        std::cout << "  Compute capability:            " << prop.major << "." << prop.minor << '\n';
        std::cout << "  Multiprocessors (SMs):         " << prop.multiProcessorCount << '\n';
        std::cout << "  GPU clock rate:                " << clockKHz / 1000 << " MHz\n";
        std::cout << "  Warp size:                     " << prop.warpSize << '\n';

        // Execution limits: relevant when choosing grid and block sizes
        std::cout << "Execution limits\n";
        std::cout << "  Max threads per block:         " << prop.maxThreadsPerBlock << '\n';
        std::cout << "  Max block dimensions:          (" << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
        std::cout << "  Max grid dimensions:           (" << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
        std::cout << "  Max threads per SM:            " << prop.maxThreadsPerMultiProcessor << '\n';
        std::cout << "  Registers per block:           " << prop.regsPerBlock << '\n';
        std::cout << "  Registers per SM:              " << prop.regsPerMultiprocessor << '\n';

        // Memory hierarchy
        std::cout << "Memory\n";
        std::cout << "  Global memory:                 " << prop.totalGlobalMem / MiB << " MiB\n";
        std::cout << "  Constant memory:               " << prop.totalConstMem / KiB << " KiB\n";
        // Default per-block limit; kernels can opt in to more (see next line).
        std::cout << "  Shared mem per block (default):" << prop.sharedMemPerBlock / KiB << " KiB\n";
        std::cout << "  Shared mem per block (opt-in): " << prop.sharedMemPerBlockOptin / KiB << " KiB\n";
        std::cout << "  Shared mem per SM:             " << prop.sharedMemPerMultiprocessor / KiB << " KiB\n";
        std::cout << "  L2 cache:                      " << prop.l2CacheSize / KiB << " KiB\n";
        std::cout << "  Memory bus width:              " << prop.memoryBusWidth << " bits\n";
        std::cout << "  Memory clock rate:             " << memClockKHz / 1000 << " MHz\n";

        // Theoretical peak bandwidth: 2 (double data rate) * clock * bus width in bytes.
        double bandwidthGBs = 2.0 * memClockKHz * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
        std::cout << "  Theoretical bandwidth:         " << bandwidthGBs << " GB/s\n";

        // Concurrency and memory model features
        std::cout << "Other\n";
        std::cout << "  Async copy engines:            " << prop.asyncEngineCount << '\n';
        std::cout << "  Concurrent kernels:            " << (prop.concurrentKernels ? "yes" : "no") << '\n';
        std::cout << "  Managed (unified) memory:      " << (prop.managedMemory ? "yes" : "no") << '\n';
    }

    return 0;
}
