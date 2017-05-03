#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudamatrix.h"

#define KERNEL_LAUNCH(kernel, gridDim, blockDim, ...) \
    kernel<<<gridDim, blockDim>>>(__VA_ARGS__)


static __host__ __device__ inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void multiply_kernel(void) {

}


namespace CUDAMatrix {

}
