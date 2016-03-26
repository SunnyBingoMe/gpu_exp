// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

/*
usage:
*/

# pragma warning (disable:4819)
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
#define threadId_colIndex threadIdx.x
#define threadId_rowIndex threadIdx.y
#define blockId_colIndex blockIdx.x
#define blockId_rowIndex blockIdx.y

/* specific define for this project */
#define BLOCK_SIZE      512 // you can modify this
#define NrItemEachThred 2   // do not modify this

__global__ 
void scan_global(float * pd_input, float * pd_output, int numElements) {
    //@@ Modify the body of this function to complete the functionality of the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this function and call them from here
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * pd_input;
    float * pd_output;
    int numElements; // number of elements in the list
    int sizeByte;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
    sizeByte = numElements * sizeof(float);
    hostOutput = (float*)malloc(sizeByte);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&pd_input, sizeByte));
    wbCheck(cudaMalloc((void**)&pd_output, sizeByte));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(pd_output, 0, sizeByte));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(pd_input, hostInput, sizeByte, cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(numElements / (BLOCK_SIZE * NrItemEachThred), 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan on the deivce
    scan_global <<<dimGrid, dimBlock>>> (pd_input, pd_output, numElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, pd_output, sizeByte, cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(pd_input);
    cudaFree(pd_output);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
