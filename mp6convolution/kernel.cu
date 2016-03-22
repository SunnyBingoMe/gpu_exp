/* usage:
.\mp6convolution.exe .\mp1\0\input0.ppm .\mp1\0\input1.csv .\mp1\0\output.ppm
*/

# pragma warning (disable:4819)
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

#define threadColIndex threadIdx.x
#define threadRowIndex threadIdx.y
#define blockColIndex blockIdx.x
#define blockRowIndex blockIdx.y

#define Out_tile_width  12
#define Channel_nr      3
#define Mask_width      5
#define Mask_radius     Mask_width/2
#define In_tile_width   (Out_tile_width + Mask_width - 1)
#define Block_width     In_tile_width

//@@ INSERT CODE HERE
__global__
void imageConvolution_global(float *pd_inputData, float *pd_outputData, const float * __restrict__ pd_maskData, int imageRowWidth, const int imageColHeight) { // !!! const float * __restrict__
    int outColIndex = blockColIndex * Out_tile_width + threadColIndex;  // outColIndex for this thread to handle !!! for this thread
    int outRowIndex = blockRowIndex * Out_tile_width + threadRowIndex;  // outRowIndex .........................
    int inColIndex = outColIndex - Mask_radius;                         // inColIndex  .........................
    int inRowIndex = outRowIndex - Mask_radius;                         // inRowIndex  .........................
    __shared__ float deviceShare_inputTileData_matrix[In_tile_width][In_tile_width][Channel_nr];

    // load input to share
    for (int channelIndex = 0; channelIndex < Channel_nr; channelIndex++){                                              // boundry of step 1 (loading)
        if ((inRowIndex >= 0) && (inRowIndex < imageColHeight) && (inColIndex >= 0) && (inColIndex < imageRowWidth))
            deviceShare_inputTileData_matrix[threadRowIndex][threadColIndex][channelIndex] = pd_inputData[(inRowIndex * imageRowWidth + inColIndex) * Channel_nr + channelIndex]; // !!! general format: (D1_index * D2_width + D2_index) * D3_width + D3_index ... 
        else
            deviceShare_inputTileData_matrix[threadRowIndex][threadColIndex][channelIndex] = 0.0;
    }

    // calculate & write
    __syncthreads();                                                    // !!! I forgot to sync. !!!
    if (threadRowIndex < Out_tile_width && threadColIndex < Out_tile_width){                                            // boundry of step 2 (calculation)
        if (outRowIndex < imageColHeight && outColIndex < imageRowWidth){                                               // boundry of step 3 (write output)
            for (int channelIndex = 0; channelIndex < Channel_nr; channelIndex++){
                float sum = 0.0; // was wrong. init should after/inside each channel, not before/outside.
                for (int i = 0; i < Mask_width; i++){
                    for (int j = 0; j < Mask_width; j++){
                        sum += pd_maskData[i * Mask_width + j] * deviceShare_inputTileData_matrix[threadRowIndex + i][threadColIndex + j][channelIndex]; // !!! important index relationship +i , +j
                    }
                }
                pd_outputData[(outRowIndex * imageRowWidth + outColIndex) * Channel_nr + channelIndex] = sum; // pd_outputData[outRowIndex][outColIndex][channelIndex] = sum;
            }
        }
    }
    __syncthreads();
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

    wbLog(TRACE, "size of size_t: ", sizeof(size_t));

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
    wbLog(TRACE, "imageWidth: ", imageWidth);
    wbLog(TRACE, "imageHeight: ", imageHeight);
    wbLog(TRACE, "imageChannels: ", imageChannels);

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
    dim3 DimBlock(In_tile_width, In_tile_width, 1);                                                 // OBS: *Block* size depends on the *bigger*  tile.
    dim3 DimGrid((imageWidth - 1) / Out_tile_width + 1, (imageHeight - 1) / Out_tile_width + 1, 1); // OBS: *Grid*  size depends on the *smaller* tile.
//  dim3 DimGrid((imageWidth - 1) / blockDim.y     + 1, (imageHeight - 1) / blockDim.x     + 1, 1); // I was wrong !!!  !!!
    imageConvolution_global <<< DimGrid, DimBlock >>> (pd_inputData, pd_outputData, pd_maskData, imageWidth, imageHeight);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    wbCheck(cudaMemcpy(hostOutputImageData,
        pd_outputData,
        imageWidth * imageHeight * imageChannels * sizeof(float),
        cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage_struct);

    wbCheck(cudaFree(pd_inputData));
    cudaFree(pd_outputData);
    cudaFree(pd_maskData);

    free(hostMaskData);
    wbImage_delete(outputImage_struct);
    wbImage_delete(inputImage_struct);

    return 0;
}
