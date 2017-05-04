#ifndef __CUDAUTILS_H_
#define __CUDAUTILS_H_

#include <iostream>
#include <cstdlib>
#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


static inline void cuda_assert(
        cudaError_t code, const char* file, int line,
        bool abort=true
        ) {

    if (code != cudaSuccess) {
        std::cerr << "cuda_assert(): "
            << cudaGetErrorString(code) << " (File '"
            << file << "', Line "
            << line << ")"
            << std::endl;

        if (abort) {
            exit(code);
        }
    }
}


#define CUDA_CHECK_ERROR(ans) do {\
    cuda_assert((ans), __FILE__, __LINE__);\
} while (0);


#define CUDA_CHECK_ERROR_DEVICE(code) do {\
    if ((code) != cudaSuccess) {\
        printf(\
            "CUDA Device Error: %s (File '%s', Line %d)\n",\
            cudaGetErrorString(code), __FILE__, __LINE__\
        );\
        return;\
    }\
} while (0);


// If we don't do this, Visual Studio will keep flipping its $#!& because it 
// apparently can't parse kernel launch syntax correctly.
#define KERNEL_LAUNCH(kernel, gridDim, blockDim, ...) do {\
    kernel<<<gridDim, blockDim>>>(__VA_ARGS__);\
    CUDA_CHECK_ERROR(cudaPeekAtLastError());\
} while (0);


#define KERNEL_LAUNCH_SHARED(kernel, gridDim, blockDim, shared, ...) do {\
    kernel<<<gridDim, blockDim, shared>>>(__VA_ARGS__);\
    CUDA_CHECK_ERROR(cudaPeekAtLastError());\
} while (0);


#define KERNEL_LAUNCH_DEVICE(kernel, gridDim, blockDim, ...) do {\
    kernel<<<gridDim, blockDim>>>(__VA_ARGS__);\
    CUDA_CHECK_ERROR_DEVICE(cudaPeekAtLastError());\
} while (0);


#define KERNEL_LAUNCH_SHARED_DEVICE(kernel, gridDim, blockDim, shared, ...) \
do {\
    kernel<<<gridDim, blockDim, shared>>>(__VA_ARGS__);\
    CUDA_CHECK_ERROR_DEVICE(cudaPeekAtLastError());\
} while (0);


static __host__ __device__ inline size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}


static __host__ inline void flush_wddm_queue(void) {
    cudaEvent_t event;
    CUDA_CHECK_ERROR(cudaEventCreate(&event));
    CUDA_CHECK_ERROR(cudaEventRecord(event));
    cudaEventQuery(event);
    cudaGetLastError();
}

#endif
