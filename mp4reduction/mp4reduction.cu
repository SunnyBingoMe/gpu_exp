// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];


/* usage:
//@@ TODO
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
#define threadColIndex threadIdx.x
#define threadRowIndex threadIdx.y
#define blockColIndex blockIdx.x
#define blockRowIndex blockIdx.y

/* specific define for this project */
#define BLOCK_SIZE 256

void total(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
}

int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
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

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here

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

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}
