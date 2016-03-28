// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

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
#define BLOCK_SIZE      512 // you can modify this, make sure it is power of 2.
#define NrItemEachThred 2   // do not modify this
//#define ShowLastSubSum 1  // show previous sub.sum for info or debug

__global__
#ifdef ShowLastSubSum
void scan_global(float * pd_input, float * pd_output, int offset, int listLengthTotal, float * pd_previousRunLastSum) {
#else
void scan_global(float * pd_input, float * pd_output, int offset, int listLengthTotal) {
#endif
    //@@ Modify the body of this function to complete the functionality of the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this function and call them from here
    __shared__ float d_shared_partialList[BLOCK_SIZE * NrItemEachThred];
    __shared__ float previousRunLastSum;
    previousRunLastSum = 0.0;
    int stride = 1;
    int perThread_index = 0;

    // 1. load data to shared mem
    // in this MP, localId_colIndex === globalId_colIndex, cuz the grid size x === 1.
    if (globalId_colIndex + offset < listLengthTotal){
        d_shared_partialList[globalId_colIndex] = pd_input[globalId_colIndex + offset];
    }
    else{
        d_shared_partialList[globalId_colIndex] = 0.0;
    }
    if (globalId_colIndex + offset + BLOCK_SIZE < listLengthTotal){
        d_shared_partialList[globalId_colIndex + BLOCK_SIZE] = pd_input[globalId_colIndex + offset + BLOCK_SIZE];
    }
    else{
        d_shared_partialList[globalId_colIndex + BLOCK_SIZE] = 0.0;
    }

    __syncthreads(); // before calculation 2

    // 2. reduce
    for (stride = 1; stride <= BLOCK_SIZE; stride *= 2){
        __syncthreads();
        perThread_index = (localId_colIndex + 1) * (stride * 2) - 1;                                    // !!! important index relationship. the 1st localId_colIndex is 0, so "+ 1" is necessary. (stride * 2) is easy to understand. "- 1" is easy.
        if (perThread_index < BLOCK_SIZE * NrItemEachThred){
            d_shared_partialList[perThread_index] += d_shared_partialList[perThread_index - stride];    // !!! important, together with above one. see figure 2 in notes 4.6 Work-Efficient Scan (But Resource-Inefficient).
        }
    }

    __syncthreads(); // before calculation 3 
    // 3. reverse reduce
    for (stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2){
        __syncthreads();
        perThread_index = (localId_colIndex + 1) * (stride * 2) - 1;
        if (perThread_index + stride < BLOCK_SIZE * NrItemEachThred){
            d_shared_partialList[perThread_index + stride] += d_shared_partialList[perThread_index];
        }
    }

    // get previousRunLastSum
    if (offset > 0 && globalId_colIndex == 0){
        previousRunLastSum = pd_output[offset - 1];
    }
    __syncthreads(); // after calculation all

    // 4. write to output
    perThread_index = globalId_colIndex + offset;
    if (globalId_colIndex + offset < listLengthTotal){
        pd_output[perThread_index] = d_shared_partialList[globalId_colIndex] + previousRunLastSum;
    }
    perThread_index += BLOCK_SIZE;
    if (globalId_colIndex + offset + BLOCK_SIZE < listLengthTotal){
        pd_output[perThread_index] = d_shared_partialList[globalId_colIndex + BLOCK_SIZE] + previousRunLastSum;
    }

#ifdef ShowLastSubSum
    __syncthreads();
    pd_previousRunLastSum[0] = pd_output[offset + BLOCK_SIZE * NrItemEachThred - 1];
#endif
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * pd_input;
    float * pd_output;
    int numElements; // number of elements in the list
    int sizeByte;
    int runCount;
#ifdef ShowLastSubSum
    float previousRunLastSum = 0.0;
    float * pd_previousRunLastSum;
#endif

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
#ifdef ShowLastSubSum
    wbCheck(cudaMalloc((void**)&pd_previousRunLastSum, sizeof(float)));
#endif
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(pd_output, 0, sizeByte));
#ifdef ShowLastSubSum
    wbCheck(cudaMemset(pd_previousRunLastSum, 0, sizeof(float)));
#endif
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(pd_input, hostInput, sizeByte, cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(1, 1, 1);
    runCount = (numElements - 1) / (BLOCK_SIZE * NrItemEachThred) + 1;
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan on the deivce
    for (int runIndex = 0; runIndex < runCount; runIndex++){
        int offset = BLOCK_SIZE * NrItemEachThred * runIndex;
#ifdef ShowLastSubSum
        scan_global <<<dimGrid, dimBlock>>> (pd_input, pd_output, offset, numElements, pd_previousRunLastSum);
        wbCheck(cudaMemcpy(&previousRunLastSum, pd_previousRunLastSum, sizeof(float), cudaMemcpyDeviceToHost));
        wbLog(TRACE, "previousRunLastSum: ", previousRunLastSum);
#else
        scan_global <<<dimGrid, dimBlock>>> (pd_input, pd_output, offset, numElements);
#endif
    }
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
