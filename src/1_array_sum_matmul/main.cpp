// Include C++ header files.
#include <cstdint>
#include <cstdlib>
#include <random>
#include <stdio.h>
#include <chrono>
#include <cmath>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"

/********************** RESULTS ***********************
 * For array length of 1024, CPU takes ~3 microseconds, while GPU 100k microseconds
 * For array length of 1U << 20U = 104576 CPU took 1715 microseconds, GPU took ~89129 microseconds
 * How about if we try to change addition with division? Nothing, CPU is still faster, you need a more complicated formula
 * Now let's try to play with the number of threads per block (or block size). Up till now we used 32
 * Basically, we noted that we can't go over 512 (or 1024 on newer GPUs), and that incrementing such number didn't 
 * improve our specs. Actually, these are the limits:
 *  - Maximum threads in X direction: 512 (1024 for compute capability >= 2.0)
 *  - Maximum threads in Y direction: 512 (1024 for compute capability >= 2.0)
 *  - Maximum threads in Z direction: 64
 *
 * CUDA 11 introduced the asynchronous memcpy operations!
 * Blocks are the unit of scheduling, and once they are scheduled to run they occupy N warps, enough to fill all of these
 * threads in the block, which can get up to 1024x1024x64, hence a lot of threads!
 *
 ******************************************************/
void printFirstN(char const *prefix, float const *arr, uint32_t N) {
    printf("%s", prefix);
    for (uint32_t i = 0; i != N; ++i) {
        printf("%.3f ", arr[i]);
    }
    printf("\n");
}

/**
 * Comando di compilazione:
 * >> nvcc main.cpp src/cuda_kernel.cu -I include -o lab
 */
static void addExercise(uint32_t arrayLength) {
    printf("---------------------- Add Exercise -----------------------------");
    printf("Array Length: %zu\n", arrayLength);
    // TODO: Create empty arrays on the host (CPU)
    float *A_cpu = (float *)malloc(sizeof(float) * arrayLength);
    float *B_cpu = (float *)malloc(sizeof(float) * arrayLength);
    float *C_cpu = (float *)malloc(sizeof(float) * arrayLength);
    if (!(A_cpu && B_cpu && C_cpu)) {
        fprintf(stderr, "Couldn't allocate system memory for all the CPU arrays\n");
        exit(1);
    }

    // TODO: initialize the arrays with random data (e.g., use rand function)
    auto start = std::chrono::high_resolution_clock::now();
    srand(clock());
    for (uint32_t i = 0; i != arrayLength; ++i) {
        A_cpu[i] = (float)rand() / RAND_MAX;
        B_cpu[i] = (float)rand() / RAND_MAX;
        C_cpu[i] = 0.f;
    }
    printFirstN("A_cpu: ", A_cpu, 10);
    printFirstN("B_cpu: ", B_cpu, 10);

    // TODO: Peform the computation on the CPU 
    start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i != arrayLength; ++i) {
        // This complicated, nonsensical formula is the one which makes so that the time spent with the 
        // GPU is the same of the CPU, with 32 threads per block
        // C_cpu[i] = sin(tanh(A_cpu[i] / B_cpu[i])) * cosh(tanh(A_cpu[i])) / cosh(B_cpu[i]) + expm1(B_cpu[i]);
        C_cpu[i] = A_cpu[i] + B_cpu[i];
    }
        
    auto end = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double, std::micro> timeTakenCPU = end - start;
    printf("Finished computing for CPU, took %.5f microseconds\n", timeTakenCPU.count());
    printFirstN("C_cpu: ", C_cpu, 10);

    for (uint32_t i = 0; i != arrayLength; ++i) {
        C_cpu[i] = 0.f;
    }

    start = std::chrono::high_resolution_clock::now();

    // TODO: call a function passing the pointer of the arrays as arguments to compute on the GPU
    auto const &[allocTime, kernelTime, feedbackTime] = kernel(A_cpu, B_cpu, C_cpu, arrayLength, 32);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> timeTakenGPU = end - start;
    
    // TODO: verify the results.
    printf("Finished computing for GPU, took %.5f microseconds, of which \n\t%.5f are due to data movement\n"
            "\t%.5f are due to kernel computation\n\t%.5f are due to result transfer\n\n", 
            timeTakenGPU.count(), allocTime, kernelTime, feedbackTime);
    printFirstN("C_cpu: ", C_cpu, 10);

    printf("\nProgramm Finished!\n");
    free(A_cpu);
    free(B_cpu);
    free(C_cpu);
    printf("-----------------------------------------------------------------");
}

static void matrixMultiplicationExercise(uint32_t numRowsFst, uint32_t numColsFst, uint32_t numColsSnd, std::normal_distribution<float> random, std::mt19937 gen, bool doPrint) {
    std::chrono::time_point<std::chrono::high_resolution_clock> t;
    uint32_t const A_numel = numRowsFst * numColsFst;
    uint32_t const B_numel = numColsFst * numColsSnd;
    uint32_t const C_numel = numRowsFst * numColsSnd;
    auto constexpr printMatrix = [](char const *prefix, float const *mat, uint32_t numRows, uint32_t numCols){
        printf("%s", prefix);
        for (uint32_t i = 0; i != numRows; ++i) {
            for (uint32_t j = 0; j != numCols; ++j) {
                printf("%.3f ", mat[numCols * i + j]);
            }
            printf("\n");
        }
        printf("\n");
    };

    // Allocate matrices on the CPU
    float *A_cpu = (float *)malloc(A_numel * sizeof(float));
    float *B_cpu = (float *)malloc(B_numel * sizeof(float));
    float *C_cpu = (float *)malloc(C_numel * sizeof(float));
    if (!(A_cpu && B_cpu && C_cpu)) {
        fprintf(stderr, "Failed allocating system memory for 3 matrices\n");
        exit(1); 
    }

    // Perform Initialization and print of the matrices
    for (uint32_t i = 0; i != A_numel; ++i) {
        A_cpu[i] = random(gen);
    }
    for (uint32_t i = 0; i != B_numel; ++i) {
        B_cpu[i] = random(gen);
    }
    for (uint32_t i = 0; i != C_numel; ++i) {
        C_cpu[i] = 0.f;
    }
    if (doPrint) {
        printMatrix("A = \n", A_cpu, numRowsFst, numColsFst);
        printMatrix("B = \n", B_cpu, numColsFst, numColsSnd);
    }

    // perform computation on CPU, timed
    t = std::chrono::high_resolution_clock::now();

    // https://www.adityaagrawal.net/blog/architecture/matrix_multiplication
    for (uint32_t k = 0; k != numColsFst; ++k) {
        for (uint32_t i = 0; i != numRowsFst; ++i) {
            // For each element in the row of B, update corresponding row of C
            for (uint32_t j = 0; j != numColsSnd; ++j) {
                C_cpu[numColsSnd * i + j] += A_cpu[numColsFst * i + k] * B_cpu[numColsSnd * k + j];
            }
        }
    }

    std::chrono::duration<double, std::micro> timeTakenCPU = std::chrono::high_resolution_clock::now() - t;
    printf("Computation performed on CPU in %.3f microseconds.\n", timeTakenCPU.count());
    if (doPrint) {
        printMatrix("C =\n", C_cpu, numRowsFst, numColsSnd);
    }

    for (uint32_t i = 0; i != C_numel; ++i) {
        C_cpu[i] = 0.f;
    }

    // Perform matrix multiplication on GPU
    t = std::chrono::high_resolution_clock::now();
    Timings const timings = matmulKernel({
        .A_cpu = A_cpu,
        .B_cpu = B_cpu,
        .C_cpu = C_cpu,
        .numRowsFst = numRowsFst,
        .numColsFst = numColsFst,
        .numColsSnd = numColsSnd,
        .A_numel = A_numel,
        .B_numel = B_numel,
        .C_numel = C_numel 
    });
    std::chrono::duration<double, std::micro> timeTakenGPU = std::chrono::high_resolution_clock::now() - t;
    printf("Computation performed on GPU in %.3f microseconds, of which"
            "\n\t%.3f allocation\n\t%.3f kernel execution\n\t%.3f feedback\n", 
            timeTakenGPU.count(), timings.allocTime, timings.kernelTime, timings.feedbackTime);
    if (doPrint) {
        printMatrix("C =\n", C_cpu, numRowsFst, numColsSnd);
    }
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist{1.f, 0.5f};

    //addExercise(1U << 20U);
    matrixMultiplicationExercise(2u<<10, 2u<<10, 2u<<10, dist, gen, false);
}

