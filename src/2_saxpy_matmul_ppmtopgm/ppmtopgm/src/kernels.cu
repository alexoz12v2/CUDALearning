#include "kernels.cuh"

#include <cstdio>
#include <cassert>

inline uint32_t constexpr logPixelsPerThread = 6u;
inline uint32_t constexpr pixelsPerThread = 1u << logPixelsPerThread;

#define CUDA_CALL(stmt) checkCudaCall((stmt), __FILE__, __LINE__)

__host__ static void checkCudaCall(cudaError_t err, char const *file, uint32_t line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[%s:%u]: (%s) %s\n", file, line, cudaGetErrorName(err), cudaGetErrorString(err));
        cudaDeviceReset();
        exit(1);
    }
}

__constant__ uint32_t width;
__constant__ uint32_t height;

__global__ static void
rgbToGrayscale_kernel(Pixel const *rgb, unsigned char *grayscale) {
    // compute tid
    uint32_t tid = (blockDim.x * blockIdx.x + threadIdx.x) * pixelsPerThread;

    unsigned char localPixels[pixelsPerThread];
    for (uint32_t i = 0; i != pixelsPerThread; ++i) {
        if ((tid + i) < width * height) {
            Pixel const &pixel = rgb[tid + i];
            localPixels[i] = static_cast<unsigned char>(
                    (0.21f * (pixel.r / 255.f) + 0.71f * (pixel.g / 255.f) + 0.07 * (pixel.b / 255.f)) * 255);
        } else {
            return;
        }
    }

    memcpy(grayscale + tid, localPixels, pixelsPerThread * sizeof(unsigned char));
}

uint32_t nextPOT(uint32_t v) {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
}

void rgbToGrayscale_gpu(RGBImage const *rgbImage, GrayImage *output) {
    assert(rgbImage->width == output->width && rgbImage->height == output->height &&
           "input and output dimensions must match");
    static uint32_t constexpr defBlockSize = 256;
    size_t const numPixels = rgbImage->width * rgbImage->height;
    size_t const numBytesRgb = numPixels * sizeof(decltype(*rgbImage->data));
    size_t const numBytesGray = numPixels * sizeof(decltype(*output->data));
    size_t const numBytes = numBytesRgb + numBytesGray;
    Pixel *d_rgb;
    unsigned char *d_grayscale;

    // prepare data
    CUDA_CALL(cudaMalloc(&d_rgb, numBytes));
    d_grayscale = (unsigned char *) d_rgb + numBytesRgb;

    CUDA_CALL(cudaMemcpy(d_rgb, rgbImage->data, numBytesRgb, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(width, &rgbImage->width, sizeof(decltype(width))));
    CUDA_CALL(cudaMemcpyToSymbol(height, &rgbImage->height, sizeof(decltype(height))));

    // compute grid size
    // since images are stored in row-major order, each block will have a 1D array of threads
    // the number of threads will be equal to min(nextPOT(width), 256)
    // the number of blocks will be equal to floor(numPixels / numThreads) + 1
    dim3 const blockSize{std::min(nextPOT(rgbImage->width), defBlockSize), 1, 1};
    dim3 const gridSize{(static_cast<uint32_t>(static_cast<float>(numPixels) / static_cast<float>(blockSize.x))
            << logPixelsPerThread) + 1, 1, 1};
    rgbToGrayscale_kernel<<<gridSize, blockSize>>>(d_rgb, d_grayscale);

    // copy back to CPU result and free
    CUDA_CALL(cudaMemcpy(output->data, d_grayscale, numBytesGray, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_rgb));
}
