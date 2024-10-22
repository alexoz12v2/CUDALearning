#pragma once

#include <cstdint>

struct Timings {
    double allocTime;
    double kernelTime;
    double feedbackTime;
};

struct matmulKernel_input {
    float const *A_cpu;      // m x k
    float const *B_cpu;      // k x n 
    float       *C_cpu;
    uint32_t     numRowsFst; // m
    uint32_t     numColsFst; // k
    uint32_t     numColsSnd; // n
    uint32_t     A_numel;
    uint32_t     B_numel;
    uint32_t     C_numel;
};

extern Timings kernel(float const *A_cpu, float const *B_cpu, float *C_cpu, uint32_t N, uint32_t blkSize);
extern Timings matmulKernel(matmulKernel_input const &in);
