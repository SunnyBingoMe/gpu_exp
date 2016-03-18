//#include "../../include/sunny_function_linux.h"
#include "../include/wb.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    wbTime_start(GPU, "getting gpu data.");

    for (int deviceIndex = 0; deviceIndex < deviceCount; deviceIndex++){
        cudaDeviceProp deviceProperties;

        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        if (deviceIndex == 0){
            if (deviceProperties.major == 9999 && deviceProperties.minor == 9999){
                wbLog(TRACE, "no cuda gpu detected.");
                wbLog(TRACE, "There are ", deviceCount, " devices supporting CUDA");
                return -1;
            }
            else if (deviceCount == 1){
                wbLog(TRACE, "only 1 device found.");
            }
            else {
                wbLog(TRACE, "There are ", deviceCount, " devices supporting CUDA");
            }
        }

        wbLog(TRACE, "\n");
        wbLog(TRACE, "device index:             ", deviceIndex, ", name: ", deviceProperties.name);
        wbLog(TRACE, "capabilities:             ", deviceProperties.major, ".", deviceProperties.minor);
        wbLog(TRACE, "max totalGlobalMem:       ", deviceProperties.totalGlobalMem >> 20, " MB");
        wbLog(TRACE, "max total Constant Mem:   ", deviceProperties.totalConstMem >> 10, " KB");
        wbLog(TRACE, "max sharedMemPerBlock:    ", deviceProperties.sharedMemPerBlock >> 10, " KB");
        wbLog(TRACE, "max block dim:            ", deviceProperties.maxThreadsDim[0], " * ", deviceProperties.maxThreadsDim[1], " * ", deviceProperties.maxThreadsDim[2]);
        wbLog(TRACE, "max grid dim:             ", deviceProperties.maxGridSize[0], " * ", deviceProperties.maxGridSize[1], " * ", deviceProperties.maxGridSize[2]);
        wbLog(TRACE, "Warp size:                ", deviceProperties.warpSize);
    }

    wbTime_stop(GPU, "Getting GPU Data."); //@@ stop the timer

    return 0;
}

