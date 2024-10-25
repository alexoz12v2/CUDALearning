#include "img.h"
#include <cstdio>
#include <cstdlib>

inline uint32_t constexpr MAX_COLOR_DEPTH = 255;

RGBImage *readPPM(const char *filename) {
    char buff[16];
    RGBImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    int w, h;

    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        return nullptr;
    }


    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        return nullptr;
    }

    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        return nullptr;
    }


    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n');
        c = getc(fp);
    }

    ungetc(c, fp);

    if (fscanf(fp, "%d %d", &w, &h) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        return nullptr;
    }


    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
        return nullptr;
    }

    if (rgb_comp_color != MAX_COLOR_DEPTH) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        return nullptr;
    }

    img = createPPM(w, h);

    while (fgetc(fp) != '\n');

    if (fread(img->data, 3 * img->width, img->height, fp) != img->height) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        return nullptr;
    }

    fclose(fp);
    return img;
}

GrayImage *readPGM(const char *filename) {
    char buff[16];
    GrayImage *img;
    FILE *fp;
    int c, depth;
    int w, h;

    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        return nullptr;
    }

    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        return nullptr;
    }


    if (buff[0] != 'P' || buff[1] != '5') {
        fprintf(stderr, "Invalid image format (must be 'P5')\n");
        return nullptr;
    }


    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n');
        c = getc(fp);
    }

    ungetc(c, fp);

    if (fscanf(fp, "%d %d", &w, &h) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        return nullptr;
    }


    if (fscanf(fp, "%d", &depth) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
        return nullptr;
    }

    if (depth != MAX_COLOR_DEPTH) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        return nullptr;
    }

    while (fgetc(fp) != '\n');
    img = createPGM(w, h);


    if (fread(img->data, img->width, img->height, fp) != img->height) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        return nullptr;
    }

    fclose(fp);
    return img;
}

void writePGM(const char *filename, GrayImage const *img) {
    FILE *fp;

    fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        return;
    }


    fprintf(fp, "P5\n");


    fprintf(fp, "# Comment\n");


    fprintf(fp, "%d %d\n", img->width, img->height);

    fprintf(fp, "%d\n", MAX_COLOR_DEPTH);

    printf("%d %d\n", img->width, img->height);

    fwrite(img->data, img->width, img->height, fp);
    fclose(fp);
}

void writePPM(const char *filename, RGBImage const *img) {
    FILE *fp;

    fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        return;
    }


    fprintf(fp, "P6\n");

    fprintf(fp, "%d %d\n", img->width, img->height);

    fprintf(fp, "%d\n", MAX_COLOR_DEPTH);

    fwrite(img->data, 3 * img->width, img->height, fp);
    fclose(fp);
}

GrayImage *createPGM(uint32_t width, uint32_t height) {

    GrayImage *img;

    img = (GrayImage *) malloc(sizeof(GrayImage));

    if (!img) {
        fprintf(stderr, "malloc failure\n");
        return nullptr;
    }

    img->width = width;
    img->height = height;
    img->data = (unsigned char *) malloc(img->width * img->height * sizeof(unsigned char));

    if (!img->data) {
        fprintf(stderr, "malloc failure\n");
        return nullptr;
    }
    return img;
}

RGBImage *createPPM(uint32_t width, uint32_t height) {
    RGBImage *img;

    img = (RGBImage *) malloc(sizeof(RGBImage));

    if (!img) {
        fprintf(stderr, "malloc failure\n");
        return nullptr;
    }

    img->width = width;
    img->height = height;
    img->data = (Pixel *) malloc(img->width * img->height * sizeof(Pixel));

    if (!img->data) {
        fprintf(stderr, "malloc failure\n");
        return nullptr;
    }
    return img;
}

void destroyPPM(RGBImage *img) {
    free(img->data);
    free(img);
}

void destroyPGM(GrayImage *img) {
    free(img->data);
    free(img);
}
