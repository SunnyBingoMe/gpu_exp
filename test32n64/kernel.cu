# pragma warning (disable:4819)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
__global__ 
void doNothing(){
    ;
}
int main()
{
    printf("size of int: %d \n", sizeof(int));
    printf("size of __int64: %d \n", sizeof(__int64));
    printf("size of size_t: %d \n", sizeof(size_t));
    printf("size of float: %d \n", sizeof(float));
    printf("size of double: %d \n", sizeof(double));
    return 0;
}
