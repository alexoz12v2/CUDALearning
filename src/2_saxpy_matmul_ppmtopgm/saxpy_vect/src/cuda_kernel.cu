// Include associated header file.
#include "../include/cuda_kernel.cuh"

// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

inline uint32_t constexpr numConstants = 2;

union SaxpyScalarConstants_Type {
    float f;
    uint32_t n;
};

__constant__ SaxpyScalarConstants_Type saxpyConstants[numConstants];

/**
 * Sample CUDA device function which adds an element from array A and array B.
 */
__global__ void saxpyKernel(float const *A, float const *B, float *C) {
    uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < saxpyConstants[1].n) {
        C[tid] = saxpyConstants[0].f * A[tid] + B[tid];
    }
}

/**
 * Wrapper function for the CUDA kernel function.
 */
void kernel(float const *A, float const *B, float scalar, float *C, uint32_t N) {
    // Launch CUDA kernel.
    float *d_A, *d_B, *d_C;
    SaxpyScalarConstants_Type const constants[numConstants] { { .f = scalar }, { .n = N } };

    cudaMalloc((void**) &d_A, N*sizeof(float));
    cudaMalloc((void**) &d_B, N*sizeof(float));
    cudaMalloc((void**) &d_C, N*sizeof(float));
    
    cudaMemcpyToSymbol(saxpyConstants, constants, numConstants * sizeof(SaxpyScalarConstants_Type));

    cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 const blockSize(512, 1, 1);
    dim3 const gridSize((N >> 9) + 1, 1, 1);

    saxpyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    
    cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
}











