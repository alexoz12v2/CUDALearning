// Include local CUDA header files.
#include "include/cuda_kernel.cuh"

// Include C++ header files.
#include <iostream>
#include <random>

inline uint32_t constexpr M = 500;
inline uint32_t constexpr N = 1000;
inline uint32_t constexpr K = 300;

void cpu_MatMul(float *A, float *B, float *C, int m, int n, int k){
    for(int row=0; row<m; row++){
        for (int col=0; col<n; col++){
            for (int ii = 0; ii < k; ii++) {
                C[row * n + col] += A[row * k + ii] * B[ii * n + col];
            }
        }
    }
}

int main() {
    auto constexpr printMat = [](char const *prefix, float const *mat, unsigned m, unsigned n) {
        std::cout << prefix;
        for (unsigned i = 0; i != m; ++i) {
            for (unsigned j = 0; j != n; ++j) {
                std::cout << mat[N * i + j] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n' << std::flush;
    };

    float *A;
    float *B;
    float *C, *C_cpu;
    std::random_device seed;
    std::mt19937 gen{seed()};
    std::uniform_real_distribution<float> random{0.f, 1.f};

    A = (float *)malloc(M*K*sizeof(float));
    B = (float *)malloc(K*N*sizeof(float));
    C = (float *)malloc(M*N*sizeof(float));
    C_cpu = (float *)malloc(M*N*sizeof(float));

    for(int i=0;i<M*K; i++) {
        A[i] = random(gen);
    }

    for (int i = 0; i != K * N; ++i) {
        B[i] = random(gen);
    }

    for(int i = 0; i < M * N; i++) {
        C[i] = 0;
        C_cpu[i] = 0;
    }

    std::cout << "Computing matrix multiplication on CPU...\n" << std::flush;
    cpu_MatMul(A,B,C_cpu,M,N,K);
    std::cout << "Done\n\n" << std::flush;

    std::cout << "Computing matrix multiplication on GPU...\n" << std::flush;
    kernel(A,B,C,M,N,K);
    std::cout << "Done\n\nShowing First 3 rows of each\n" << std::flush;
    printMat("CPU[0:2, :] = \n", C_cpu, 3, 4);
    printMat("GPU[0:2, :] = \n", C, 3, 4);

    bool error = false;
    float diff = 0.0;
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            diff = abs(C[i * N + j] - C_cpu[i * N + j]);
            if (diff > 10e-4) { 
                error = true;
                std::cout << i << " " << j << " " << diff << " " << C[i*N+j] << " " << C_cpu[i*N+j] << std::endl;
           }           
        }
    }

    if (error) {
        std::cout << "\nThe Results are Different!\n";
    } else {
        std::cout << "\nThe Results match!\n";
    }

    std::cout << "\nProgramm Finished!\n";
}