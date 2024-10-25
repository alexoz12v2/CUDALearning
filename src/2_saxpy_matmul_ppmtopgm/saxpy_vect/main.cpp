// Include local CUDA header files.
#include "include/cuda_kernel.cuh"

// Include C++ header files.
#include <iostream>
#include <random>

inline uint32_t constexpr N = 10000;

// Single precision A X plus Y
// saxpy: z_i = a \times x_i + y_i
void cpu_saxpy_vect(float const *x, float const *y, float a, float *z, uint32_t n) {
    for (int i = 0; i < n; i++) {
        z[i] = a * x[i] + y[i];
    }
}

int main(void) {
    static uint32_t constexpr threshold = 10e-4;
    std::random_device seed;
    std::mt19937 gen{seed()};
    std::uniform_real_distribution<float> random;
    float A[N];
    float B[N];
    float C[N], C_cpu[N];
    float const scalar = random(gen);

    for(int i = 0; i < N; i++) {
        A[i] = random(gen);
        B[i] = random(gen);
        C_cpu[i] = 0;
        C[i] = 0;
    }

    std::cout << "Starting saxpy computation on the CPU...\n" << std::flush;
    cpu_saxpy_vect(A, B, scalar, C_cpu, N);
    std::cout << "Done!\n";

    std::cout << "Starting saxpy computation on the GPU...\n" << std::flush;
    kernel(A, B, scalar, C, N);
    std::cout << "Done!\n\nshowing first 4 elements of each result:\nCPU[0:3] = \n";

    for (uint32_t i = 0; i != 4; ++i) {
        std::cout << C_cpu[i] << ' ';
    }
    std::cout << "\n\nGPU[0:3] = \n";
    for (uint32_t i = 0; i != 4; ++i) {
        std::cout << C[i] << ' ';
    }
    std::cout << "\n\n" << std::flush;
    
    bool error = false;
    float diff = 0.0;
    for(int i = 0; i < N; i++){
        diff = abs(C[i] - C_cpu[i]);
        if (diff > threshold) {
            error = true;
            std::cout << i << " " << diff << " " << C[i] << " " << C_cpu[i] << std::endl;
        }     
    }

    if (error) {
        std::cout << "\nThe Results are Different!\n";
    } else {
        std::cout << "\nThe Results match!\n";
    }

    std::cout << "\nProgramm Finished!\n";
}