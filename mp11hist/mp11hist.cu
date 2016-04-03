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

/* general define for cuda,  OBS: maybe not suitable for 3D */
#define Grid_width  gridDim.x
#define Grid_height gridDim.y
#define Grid_pages  gridDim.z
#define Grid_size   (Grid_width * Grid_height)
#define Block_width   blockDim.x
#define Block_height  blockDim.y
#define Block_pages   blockDim.z
#define BLOCK_SIZE    (Block_width * Block_height)
#define localId_colIndex       threadIdx.x
#define localId_rowIndex       threadIdx.y
#define localId_overallIndex2D (localId_rowIndex * Block_width + localId_colIndex)
#define blockId_colIndex         blockIdx.x
#define blockId_rowIndex         blockIdx.y
#define blockId_overallIndex2D   (blockId_rowIndex * Grid_width + blockId_colIndex)
#define globalId_colIndex           (blockId_colIndex * Block_width + localId_colIndex)
#define globalId_rowIndex           (blockId_rowIndex * Block_height + localId_rowIndex)
#define globalId_overallIndex2D     (globalId_rowIndex * Block_width*Grid_width + globalId_colIndex)
#define __syncdevice cudaDeviceSynchronize

/* specific define for this project */
#define Tile_width       16  // obs: related with HISTOGRAM_LENGTH
#define HISTOGRAM_LENGTH 256 // set to 256 to easy hist related thread actions. when HISTOGRAM_LENGTH < 256, hard to code. when HISTOGRAM_LENGTH > 256, need to add "if" statements
#define Channel_nr       3
#define Debug            1
#define Scan_alternative 1

//@@ insert kernel code here
__global__
void preScan_global(float *pd_inputData, float *pd_outputData, float * pd_hist_dividedToFloatProbility, float * pd_minMax, int imageRowWidth, int imageColHeight) {
    if (Debug && globalId_overallIndex2D == 0) pd_minMax[1] = 0.0;
    __shared__ int pd_shared_hist[256]; pd_shared_hist[localId_overallIndex2D] = 0;
    float d_onePixelAllChannels[3];

    // load input and convert to unsigned char & calculate to local pd_shared_hist
    __syncthreads();
    if ((globalId_rowIndex < imageColHeight) && (globalId_colIndex < imageRowWidth)){
        int onePixelStartIndex = (globalId_rowIndex * imageRowWidth + globalId_colIndex) * Channel_nr;
        for (int channelIndex = 0; channelIndex < Channel_nr; channelIndex++){
            d_onePixelAllChannels[channelIndex] = (unsigned char)(255 * pd_inputData[onePixelStartIndex + channelIndex]);
        }
        unsigned char t_grayValueAsCharAsIndex = (unsigned char)(0.21 * d_onePixelAllChannels[0] + 0.71 * d_onePixelAllChannels[1] + 0.07 *  d_onePixelAllChannels[2]);

        //calculate to local pd_shared_hist
        atomicAdd(&pd_shared_hist[t_grayValueAsCharAsIndex], 1);
    }

    // division ; then atom-write to global hist probility pd_hist_dividedToFloatProbility pdf
    __syncthreads();
    float t_localHistOneFloatValue = ((float)(pd_shared_hist[localId_overallIndex2D])) / (imageRowWidth * imageColHeight);
    atomicAdd(&pd_hist_dividedToFloatProbility[localId_overallIndex2D], t_localHistOneFloatValue);

    // scan global hist probility pd_hist_dividedToFloatProbility to cdf
    __syncthreads();
    if (Debug && globalId_overallIndex2D == 0) pd_minMax[1] += 2^0;
}

__global__
void scan_global(float *pd_inputData, float *pd_outputData, float * pd_hist_dividedToFloatProbility, float * pd_minMax, int imageRowWidth, int imageColHeight) {
    __shared__ float pd_shared_histFloat[256];
    if (Debug && globalId_overallIndex2D == 0) pd_minMax[1] += 2^1;
    if (blockId_overallIndex2D == 0){
        if (Debug && globalId_overallIndex2D == 0) pd_minMax[1] += 2^2;
        int stride;
        int perThread_index;

        // load hist from global to local shared mem
        pd_shared_histFloat[localId_overallIndex2D] = pd_hist_dividedToFloatProbility[localId_overallIndex2D];
        __syncthreads();

        // use only 128 threads to do scan
        if (localId_overallIndex2D < 128){

            // scan-> reduce
            for (stride = 1; stride <= 128; stride *= 2){
                __syncthreads();
                perThread_index = (localId_colIndex + 1) * (stride * 2) - 1;
                if (perThread_index < 256){
                    pd_shared_histFloat[perThread_index] += pd_shared_histFloat[perThread_index - stride];
                }
            }

            // scan-> reverse reduce
            __syncthreads();
            for (stride = 128 / 2; stride >= 1; stride /= 2){
                __syncthreads();
                perThread_index = (localId_colIndex + 1) * (stride * 2) - 1;
                if (perThread_index + stride < 256){
                    pd_shared_histFloat[perThread_index + stride] += pd_shared_histFloat[perThread_index];
                }
            }

        } // if (localId_overallIndex2D < 128){

        // save back to global mem
        __syncthreads();
        if (pd_shared_histFloat[localId_overallIndex2D] > 0.0){
            atomicMin(pd_minMax, pd_shared_histFloat[localId_overallIndex2D]);
        }
        __syncthreads();
        pd_hist_dividedToFloatProbility[localId_overallIndex2D] = pd_shared_histFloat[localId_overallIndex2D];

    } // if (block == 0){
    if (Debug && globalId_overallIndex2D == 0) pd_minMax[1] += 2^3;
    __syncthreads();
}

__global__ // normalization
void normalization_global(float *pd_inputData, float *pd_outputData, float * pd_hist_dividedToFloatProbility, float * pd_minMax, int imageRowWidth, int imageColHeight) {
    if (Debug && globalId_overallIndex2D == 0) pd_minMax[1] += 2^4;
    __shared__ float pd_shared_histFloat[256];

    //// load cdf hist from global to local shared mem
    pd_shared_histFloat[localId_overallIndex2D] = pd_hist_dividedToFloatProbility[localId_overallIndex2D];
    __syncthreads();

    //// normalize
    if ((globalId_rowIndex < imageColHeight) && (globalId_colIndex < imageRowWidth)){
        int onePixelStartIndex = (globalId_rowIndex * imageRowWidth + globalId_colIndex) * Channel_nr;
        for (int channelIndex = 0; channelIndex < Channel_nr; channelIndex++){
            unsigned char colorValueAsCharAsIndex = (unsigned char)(255 * pd_inputData[onePixelStartIndex + channelIndex]);
            unsigned char newColorValue = 255 * (pd_shared_histFloat[colorValueAsCharAsIndex] - pd_minMax[0]) / (1 - pd_minMax[0]);
            pd_outputData[onePixelStartIndex + channelIndex] = (float)newColorValue / 255;
        }
    }
    __syncthreads();
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage_struct;
    wbImage_t outputImage_struct;
    const char * inputImage_structFile;

    //@@ Insert more code here
    float * ph_inputData;
    float * ph_outputData;
    float * pd_hist_dividedToFloatProbility;
    float   ph_hist_dividedToFloatProbility[256];
    float * pd_inputData;
    float * pd_outputData;
    float   ph_minMax[2]; ph_minMax[0] = 1.0; ph_minMax[1] = 0.0;
    float * pd_minMax;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImage_structFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage_struct = wbImport(inputImage_structFile);
    imageWidth = wbImage_getWidth(inputImage_struct);
    imageHeight = wbImage_getHeight(inputImage_struct);
    imageChannels = wbImage_getChannels(inputImage_struct);
    outputImage_struct = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");
    wbLog(TRACE, "imageWidth: ", imageWidth);
    wbLog(TRACE, "imageHeight: ", imageHeight);
    wbLog(TRACE, "imageChannels: ", imageChannels);

    //@@ insert code here
    ph_inputData = wbImage_getData(inputImage_struct);
    ph_outputData = wbImage_getData(outputImage_struct);


    wbTime_start(GPU, "Doing GPU/CPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    wbCheck(cudaMalloc((void **)&pd_inputData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **)&pd_outputData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **)&pd_hist_dividedToFloatProbility, 256 * sizeof(float)));
    wbCheck(cudaMalloc((void **)&pd_minMax, 2 * sizeof(float)));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    wbCheck(cudaMemcpy(pd_inputData, ph_inputData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(pd_minMax, ph_minMax, 2 * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemset(pd_hist_dividedToFloatProbility, 0, 256 * sizeof(float)));
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU preScan_global");
    //@@ INSERT CODE HERE
    dim3 DimBlock(Tile_width, Tile_width, 1);
    dim3 DimGrid((imageWidth - 1) / Tile_width + 1, (imageHeight - 1) / Tile_width + 1, 1);
    preScan_global <<< DimGrid, DimBlock >>> (pd_inputData, pd_outputData, pd_hist_dividedToFloatProbility, pd_minMax, imageWidth, imageHeight); __syncdevice();
    wbTime_stop(Compute, "Doing the computation on the GPU preScan_global");

    if (Debug || Scan_alternative)
        wbCheck(cudaMemcpy(ph_hist_dividedToFloatProbility, pd_hist_dividedToFloatProbility, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    if (0){ // pass
        for (int i = 0; i < 256; i++){
            wbLog(TRACE, "pdf hist ", i, ": ", ph_hist_dividedToFloatProbility[i]);
        }
    }
    if (Scan_alternative){
        wbTime_start(Compute, "Scan by CPU");
        float t_float_sum = 0.0;
        ph_minMax[0] = 1.0;
        for (int i = 0; i < 256; i++){
            if (ph_hist_dividedToFloatProbility[i] > 0){
                ph_minMax[0] = min(ph_minMax[0], ph_hist_dividedToFloatProbility[i]);
            }
            t_float_sum += ph_hist_dividedToFloatProbility[i];
            wbLog(TRACE, "pdf hist sum ", t_float_sum);
            ph_hist_dividedToFloatProbility[i] = t_float_sum;
        }
        wbCheck(cudaMemcpy(pd_minMax, ph_minMax, 1 * sizeof(float), cudaMemcpyHostToDevice));
        wbCheck(cudaMemcpy(pd_hist_dividedToFloatProbility, ph_hist_dividedToFloatProbility, 256 * sizeof(float), cudaMemcpyHostToDevice));
        wbTime_stop(Compute, "Scan by CPU");
    }
    else{
        wbTime_start(Compute, "Scan by GPU");
        scan_global              <<< DimGrid, DimBlock >>> (pd_inputData, pd_outputData, pd_hist_dividedToFloatProbility, pd_minMax, imageWidth, imageHeight); __syncdevice();
        wbTime_stop(Compute, "Scan by GPU");
    }

    if (Debug){
        wbCheck(cudaMemcpy(ph_hist_dividedToFloatProbility, pd_hist_dividedToFloatProbility, 256 * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 256; i++){
            wbLog(TRACE, "cdf hist ", i, ": ", ph_hist_dividedToFloatProbility[i]);
        }
    }

    wbTime_start(Compute, "Doing the computation on the GPU normalization_global");
    normalization_global         <<< DimGrid, DimBlock >>> (pd_inputData, pd_outputData, pd_hist_dividedToFloatProbility, pd_minMax, imageWidth, imageHeight); __syncdevice();
    wbTime_stop(Compute, "Doing the computation on the GPU normalization_global");

    wbTime_start(Copy, "Copying data from the GPU");
    wbCheck(cudaMemcpy(ph_outputData, pd_outputData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU/CPU Computation (memory + compute)");

    if (Debug){
        wbCheck(cudaMemcpy(&(ph_minMax[1]), &(pd_minMax[1]), 1 * sizeof(float), cudaMemcpyDeviceToHost));
        wbLog(TRACE, "ph_minMax[0]: ", ph_minMax[0]);
        wbLog(TRACE, "ph_minMax[1]: ", ph_minMax[1]);
    }

    wbSolution(args, outputImage_struct);

    //@@ insert code here
    wbCheck(cudaFree(pd_inputData));
    wbCheck(cudaFree(pd_outputData));
    wbCheck(cudaFree(pd_hist_dividedToFloatProbility));
    wbCheck(cudaFree(pd_minMax));

    return 0;
}
