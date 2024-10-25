#include "kernels.cuh"
#include "imglib/img.h"

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <type_traits>
#include <thread>
#include <chrono>

template<typename T> requires (!std::is_pointer_v<T>)
class Janitor {
public:
    explicit Janitor(T *ptr, void (*destroy)(T *ptr)) : m_destroy(destroy), m_ptr(ptr) {}
    Janitor(Janitor const &) = delete;
    Janitor(Janitor &&) noexcept = default;
    Janitor &operator=(Janitor const &) = delete;
    Janitor &operator=(Janitor &&) = default;
    ~Janitor() {
        m_destroy(m_ptr);
    }

private:
    void (*m_destroy)(T *ptr);

    mutable T *m_ptr;
};

void rgbToGrayscale(RGBImage const *input, GrayImage *output) {
    assert(input->width == output->width && input->height == output->height &&
           "input and output dimensions must match");
    for (uint64_t i = 0; i != input->width * input->height; ++i) {
        Pixel const &pixel = input->data[i];
        unsigned char &outPix = output->data[i];
        outPix = static_cast<unsigned char>((0.21f * (pixel.r / 255.f) + 0.71f * (pixel.g / 255.f) + 0.07 * (pixel.b / 255.f)) * 255);
    }
}

// https://gist.github.com/bunnyDrug/41e9e56f20e1033e700986ec85f560f1
int main() {
    namespace sc = std::chrono;
    using timens = sc::time_point<sc::system_clock, sc::nanoseconds>;

    // Allocate all the data
    RGBImage *rgbImage = readPPM("image.ppm");
    if (!rgbImage) {
        fprintf(stderr, "Couldn't open image");
        return 1;
    }
    Janitor<RGBImage> const rgbJanitor{rgbImage, destroyPPM};

    GrayImage *grayImage = createPGM(rgbImage->width, rgbImage->height);
    if (!grayImage) {
        fprintf(stderr, "Couldn't allocate memory for grayscale image");
        return 1;
    }
    Janitor<GrayImage> const grayJanitor{grayImage, destroyPGM};

    GrayImage *grayImage_cpu = createPGM(rgbImage->width, rgbImage->height);
    if (!grayImage) {
        fprintf(stderr, "Couldn't allocate memory for grayscale image");
        return 1;
    }
    Janitor<GrayImage> const grayJanitor_cpu{grayImage_cpu, destroyPGM};

    // perform conversion on the CPU
    timens cpuStart = sc::high_resolution_clock::now();
    rgbToGrayscale(rgbImage, grayImage_cpu);
    timens cpuEnd = sc::high_resolution_clock::now();
    auto cpuDuration = sc::duration_cast<sc::microseconds>(cpuEnd - cpuStart).count();

    // Perform conversion on the GPU
    timens gpuStart = sc::high_resolution_clock::now();
    rgbToGrayscale_gpu(rgbImage, grayImage);
    timens gpuEnd = sc::high_resolution_clock::now();
    auto gpuDuration = sc::duration_cast<sc::microseconds>(gpuEnd - gpuStart).count();

    printf("CPU Processing Time: %ld µs\n", cpuDuration);
    printf("GPU Processing Time: %ld µs\n", gpuDuration);

    // Compare and Print the First 4 Elements
    printf("First 4 elements comparison (CPU vs GPU):\n");
    for (int i = 0; i < 4; ++i) {
        printf("CPU: %u, GPU: %u\n", grayImage_cpu->data[i], grayImage->data[i]);
    }

    // Compare all elements
    bool allEqual = true;
    for (uint64_t i = 0; i < rgbImage->width * rgbImage->height; ++i) {
        if (grayImage_cpu->data[i] != grayImage->data[i]) {
            printf("Mismatch found at index %llu: CPU = %u, GPU = %u\n", i, grayImage_cpu->data[i], grayImage->data[i]);
            allEqual = false;
            break; // Stop at first error found (remove break to check all errors)
        }
    }

    if (allEqual) {
        printf("All elements match between CPU and GPU outputs.\nWriting image \"gray.pgm\" to disk\n");
        writePGM("gray.pgm", grayImage);
    } else {
        printf("Discrepancies found between CPU and GPU outputs.\n");
    }
}
