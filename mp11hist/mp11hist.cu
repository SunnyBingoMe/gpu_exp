// Histogram Equalization

/*
usage:
*/

#define TIMER_OK
#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "../include/wb.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

/* general define for cuda */
#define localId_colIndex threadIdx.x
#define localId_rowIndex threadIdx.y
#define blockId_colIndex blockIdx.x
#define blockId_rowIndex blockIdx.y
#define globalId_colIndex blockIdx.x * blockDim.x + localId_colIndex
#define globalId_rowIndex blockIdx.y * blockDim.y + localId_rowIndex

/* specific define for this project */
#define HISTOGRAM_LENGTH 256

//@@ insert kernel code here

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here

    wbSolution(args, outputImage);

    //@@ insert code here

    return 0;
}
