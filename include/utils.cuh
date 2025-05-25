#ifndef CUDA_UTILS
#define CUDA_UTILS

#include <cuda_runtime.h>
#include <iostream>

namespace utils {

template<typename T>
inline void checkCudaError(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " '" << func << "': "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

} 
#define cudaCheck(val) utils::checkCudaError((val), #val, __FILE__, __LINE__)

template<typename Kernel, typename... Args>
inline void launchKernel(Kernel kernel,
                         const dim3& gridDim,
                         const dim3& blockDim,
                         size_t sharedMemBytes,
                         cudaStream_t stream,
                         Args... args) {
    kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

#endif
