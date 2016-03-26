// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];


/* 
This is the "better/improved reduction. [* (Resource Efficient) *] 
Related: lecture 4.3.

usage:
.\mp4reduction.exe .\mp4\8\input0.ppm .\mp4\8\output.ppm

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
#define BLOCK_SIZE      512 // do not modify this
#define NrItemEachThred 2   // do not modify this

__global__
void total(float * d_input, float * d_output, int numInputElements) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ float ds_partialSum[BLOCK_SIZE * NrItemEachThred];
    int perThread_inputIndex = blockId_colIndex * BLOCK_SIZE * NrItemEachThred + threadId_colIndex;
    if (perThread_inputIndex < numInputElements){
        ds_partialSum[threadId_colIndex] = d_input[perThread_inputIndex];
    }
    else{
        ds_partialSum[threadId_colIndex] = 0.0;
    }
    perThread_inputIndex += BLOCK_SIZE;
    if (perThread_inputIndex < numInputElements){
        ds_partialSum[threadId_colIndex + BLOCK_SIZE] = d_input[perThread_inputIndex];
    }
    else{
        ds_partialSum[threadId_colIndex + BLOCK_SIZE] = 0.0;
    }
    __syncthreads();

    //@@ Traverse the reduction tree
    for (int stride = BLOCK_SIZE; stride >= 1; stride /= 2){
        if (threadId_colIndex < BLOCK_SIZE){
            ds_partialSum[threadId_colIndex] += ds_partialSum[threadId_colIndex + stride];
        }
        __syncthreads();
    }

    //@@ Write the computed sum of the block to the output vector at the correct index
    if (threadId_colIndex == 0){
        d_output[blockId_colIndex] = ds_partialSum[0];
    }
}

int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * pd_inputVector;
    float * pd_outputVector;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    hostOutput = (float*)malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    int inputSizeByte = numInputElements * sizeof(float);
    int outputSizeByte = numOutputElements * sizeof(float);
    wbCheck(cudaMalloc(&pd_inputVector, inputSizeByte));
    wbCheck(cudaMalloc(&pd_outputVector, outputSizeByte));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy(pd_inputVector, hostInput, inputSizeByte, cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(numOutputElements, 1, 1); // why numOutputElements ???

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    total <<<dimGrid, dimBlock>>> (pd_inputVector, pd_outputVector, numInputElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy(hostOutput, pd_outputVector, outputSizeByte, cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
    * Reduce output vector on the host
    * NOTE: One could also perform the reduction of the output vector
    * recursively and support any size input. For simplicity, we do not
    * require that for this lab.
    ********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }
    wbLog(TRACE, hostOutput[0]);

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck(cudaFree(pd_inputVector));
    wbCheck(cudaFree(pd_outputVector));
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}
