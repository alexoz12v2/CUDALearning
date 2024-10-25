#pragma once

#include <cstdint>

struct Pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};
static_assert(sizeof(Pixel) == 3 && alignof(Pixel) == 1);

// data is a vector of dimension width*height*channels (RGB has 3 channels)
// pixels are stored as RGB one after the other ( r_0,g_0,b_0,r_1,g_1,b_1...)
struct RGBImage {
    Pixel *data;
    uint32_t width;
    uint32_t height;
};

// data is a vector of dimension width*height*channels (GrayImage has 1 channels)
// pixels are stored one after the other ( gray_0, gray_1, gray_2...)
struct GrayImage {
    unsigned char *data;
    uint32_t width;
    uint32_t height;
};

RGBImage *readPPM(const char *filename);

GrayImage *readPGM(const char *filename);

GrayImage *createPGM(uint32_t width, uint32_t height);

RGBImage *createPPM(uint32_t width, uint32_t height);

void destroyPGM(GrayImage *img);

void destroyPPM(RGBImage *img);

void writePGM(char const *filename, GrayImage const *img);

void writePPM(char const *filename, RGBImage const *img);
