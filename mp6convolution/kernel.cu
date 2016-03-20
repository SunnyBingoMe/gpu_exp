#define TIMER_OK
//#define BLOCK_SIZE 256

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "../include/wb.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define Out_tile_width 16
#define Mask_width  5
#define Mask_radius Mask_width/2
#define In_tile_width (Out_tile_width + Mask_width - 1)
#define Block_width In_tile_width

//@@ INSERT CODE HERE
__global__
void imageConvolution_global(float *pd_inputData, float *pd_outputData, float *pd_maskData, int imageWidth, int imageHeight, int imageChannels, int maskWidth) {
    //
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage_struct;
    wbImage_t outputImage_struct;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * pd_inputData;
    float * pd_outputData;
    float * pd_maskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage_struct = wbImport(inputImageFile);
    hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage_struct);
    imageHeight = wbImage_getHeight(inputImage_struct);
    imageChannels = wbImage_getChannels(inputImage_struct);

    outputImage_struct = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage_struct);
    hostOutputImageData = wbImage_getData(outputImage_struct);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **)&pd_inputData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **)&pd_outputData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **)&pd_maskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(pd_inputData,
        hostInputImageData,
        imageWidth * imageHeight * imageChannels * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(pd_maskData,
        hostMaskData,
        maskRows * maskColumns * sizeof(float),
        cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    dim3 DimBlock(Block_width, Block_width, 1);
    dim3 DimGrid((imageWidth - 1) / DimBlock.x + 1, (imageHeight - 1) / DimBlock.y + 1, 1);
    imageConvolution_global <<< DimGrid, DimBlock >>> (pd_inputData, pd_outputData, pd_maskData, imageWidth, imageHeight, imageChannels, maskWidth);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
        pd_outputData,
        imageWidth * imageHeight * imageChannels * sizeof(float),
        cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage_struct);

    cudaFree(pd_inputData);
    cudaFree(pd_outputData);
    cudaFree(pd_maskData);

    free(hostMaskData);
    wbImage_delete(outputImage_struct);
    wbImage_delete(inputImage_struct);

    return 0;
}
