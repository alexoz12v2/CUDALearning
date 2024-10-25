#pragma once

#include <cstdint>

// List wrapper function callable by .cpp file.
void kernel(float const *A, float const *B, float scalar, float *C, uint32_t N);
