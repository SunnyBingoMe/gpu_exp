#define TIMER_OK

#include "../include/wb.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/* usage:
.\vectorAdd.exe .\mp1\0\input1.raw .\mp1\0\input0.raw .\mp1\0\output.raw
*/


__global__
void vecAdd(float * in1, float * in2, float * out, int len)
{
    //@@ Insert code to implement vector addition here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
        out[i] = in1[i] + in2[i];
}

int main(int argc, char ** argv)
{
    wbArg_t args;
    cudaError_t returned;
    int inputLength;
    int inputByteSize;

    float * hostInput1;
    float * hostInput2;
    float * hostOutput;

    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "input length: ", inputLength);
    inputByteSize = inputLength * sizeof(float);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    returned = cudaMalloc(&deviceInput1, inputByteSize);
    if (returned != cudaSuccess) wbLog(ERROR, "cudaMalloc(&deviceInput1");
    returned = cudaMalloc(&deviceInput2, inputByteSize);
    if (returned != cudaSuccess) wbLog(ERROR, "cudaMalloc(&deviceInput2");
    returned = cudaMalloc(&deviceOutput, inputByteSize);
    if (returned != cudaSuccess) wbLog(ERROR, "cudaMalloc(&deviceOutput");
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    returned = cudaMemcpy(deviceInput1, hostInput1, inputByteSize, cudaMemcpyHostToDevice);
    if (returned != cudaSuccess) wbLog(ERROR, "cudaMemcpy(deviceInput1");
    returned = cudaMemcpy(deviceInput2, hostInput2, inputByteSize, cudaMemcpyHostToDevice);
    if (returned != cudaSuccess) wbLog(ERROR, "cudaMemcpy(deviceInput2");
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(inputByteSize / 256 + 1, 1, 1);
    dim3 DimBlock(256, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    vecAdd <<<DimGrid, DimBlock >>>(deviceInput1, deviceInput2, deviceOutput, inputByteSize);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    returned = cudaMemcpy(hostOutput, deviceOutput, inputByteSize, cudaMemcpyDeviceToHost);
    if (returned != cudaSuccess) wbLog(ERROR, "cudaMemcpy(hostOutput");
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
