// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// Include associated header file.
#include "../include/cuda_kernel.cuh"

__host__ static void checkCudaCall(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("[%s:%d]%s\n", file, line, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(1);
    } 
}
#if 1
#define CUDA_CALL(stmt) checkCudaCall((stmt), __FILE__, __LINE__)
#else
#define CUDA_CALL(stmt) stmt
#endif

__device__ static inline float4 operator+(float4 a, float4 b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

// TODO: Define the kernel function right here
__global__ void VectorAdd(float4 const *a, float4 const *b, float4 *c) {
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    ////c[i] = sin(tanh(a[i] / b[i])) * cosh(tanh(a[i])) / cosh(b[i])+ expm1(b[i]);
    //c[i] = a[i] + b[i];
    // By utilizing float4 instead of float, the generated assembly will use 
    // a LDG.E.128 (load 128 bits) and STG.E.128 (store 128 bits), but still 4 FADD instructions
    // That is because there is no need for Explicit SIMD, CUDA will figure it out itself
    // But, it is nice to have an overloaded operator such that we have an easier time writing code
    c[i] = a[i] + b[i];
}

/**
 * Iteration 1: assume matrices are m x k, k x n, and the result will be m x n
 * There will be a thread for each element in the first input matrix.
 * Therefore each thread is associated to 
 *   - An element in the left matrix
 *   - A row in the right matrix, whose row index equal to the column index of the left matrix
 *   - will produce a part of the result of the row having the same row index of the left matrix
 */
__global__ void matmul(float const *a, float const *b, float *c, uint32_t m, uint32_t k, uint32_t n) {
    // TODO: Assuming 1 block for now
    uint32_t rowIndex = threadIdx.x;
    uint32_t colIndex = threadIdx.y;

    float leftElem = a[rowIndex * n + colIndex];
    for (uint32_t i = 0; i != n; ++i) {
        float rightElem = b[colIndex * n + i];
        atomicAdd(&(c[rowIndex * n + i]), leftElem * rightElem);
    }
}

/**
 * Wrapper function for the CUDA kernel function.
 * we expect N to be a power of two and to be an integer multiple of blkSize
 * A call to a __global__ function is asynchronous, meaning it returns before the device has completed its execution.
 */
Timings kernel(float const *A_cpu, float const *B_cpu, float *C_cpu, uint32_t N, uint32_t blkSize) {
    // TODO: create the device pointers
    auto start = std::chrono::high_resolution_clock::now();
    float *A_gpu, *B_gpu, *C_gpu;
    uint32_t arrayBytes = N * sizeof(float);

    // TODO: allocate device memory. Hint: used cudaMalloc
    CUDA_CALL(cudaMalloc((void**)&A_gpu, arrayBytes));
    CUDA_CALL(cudaMalloc((void**)&B_gpu, arrayBytes));
    CUDA_CALL(cudaMalloc((void**)&C_gpu, arrayBytes));

    // TODO: Copy data from host to device. Hint: use cudaMemcpy
    CUDA_CALL(cudaMemcpy(A_gpu, A_cpu, arrayBytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(B_gpu, B_cpu, arrayBytes, cudaMemcpyHostToDevice));
    auto alloc = std::chrono::high_resolution_clock::now();

    // TODO: define the thread dimentions 
    uint32_t blockSize = blkSize / 4; // <--------------------------- Remove /4 if you switch back to float in VectorAdd
    uint32_t gridSize = N / blkSize; // 32 x 32 = 1024

    // TODO: Issue the kernel on the GPU 
    VectorAdd<<<gridSize, blockSize>>>((float4 const *)A_gpu, (float4 const *)B_gpu, (float4 *)C_gpu);
    CUDA_CALL(cudaDeviceSynchronize());
    auto kerTime = std::chrono::high_resolution_clock::now();
    
    // TODO: Copy the computed results from device to host
    CUDA_CALL(cudaMemcpy(C_cpu, C_gpu, arrayBytes, cudaMemcpyDeviceToHost));

    // Free GPU Memory
    CUDA_CALL(cudaFree(A_gpu));
    CUDA_CALL(cudaFree(B_gpu));
    CUDA_CALL(cudaFree(C_gpu));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> allocTime = alloc - start;
    std::chrono::duration<double, std::micro> kernelTime = kerTime - alloc;
    std::chrono::duration<double, std::micro> feedbackTime = end - kerTime;
    return {
        allocTime.count(),
        kernelTime.count(),
        feedbackTime.count()
    };
}

Timings matmulKernel(matmulKernel_input const &in) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> alloc;
    std::chrono::time_point<std::chrono::high_resolution_clock> kernel;
    std::chrono::time_point<std::chrono::high_resolution_clock> feedback;

    uint32_t const A_numBytes = in.A_numel * sizeof(float);
    uint32_t const B_numBytes = in.B_numel * sizeof(float);
    uint32_t const C_numBytes = in.C_numel * sizeof(float);
    uint32_t const numBytes =A_numBytes + B_numBytes + C_numBytes;

    float *A_gpu, *B_gpu, *C_gpu;
    dim3 blockSize(in.numRowsFst, in.numColsFst);

    // create device pointers
    CUDA_CALL(cudaMalloc((void**)&A_gpu, numBytes));
    B_gpu = A_gpu + in.A_numel;
    C_gpu = A_gpu + in.A_numel + in.B_numel;

    // copy data to GPU global memory
    CUDA_CALL(cudaMemcpy(A_gpu, in.A_cpu, A_numBytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(B_gpu, in.B_cpu, B_numBytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(C_gpu, in.C_cpu, C_numBytes, cudaMemcpyHostToDevice));
    alloc = std::chrono::high_resolution_clock::now();

    matmul<<<1, blockSize>>>(A_gpu, B_gpu, C_gpu, in.numRowsFst, in.numColsFst, in.numColsSnd);
    CUDA_CALL(cudaDeviceSynchronize());
    kernel = std::chrono::high_resolution_clock::now();

    CUDA_CALL(cudaMemcpy(in.C_cpu, C_gpu, C_numBytes, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(A_gpu));
    feedback = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> const allocDuration    = alloc - start;
    std::chrono::duration<double, std::micro> const kernelDuration   = kernel - alloc;
    std::chrono::duration<double, std::micro> const feedbackDuration = feedback - kernel;
    
    return {
        allocDuration.count(),
        kernelDuration.count(),
        feedbackDuration.count()
    };
}

