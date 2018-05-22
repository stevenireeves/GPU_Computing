#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#define BLOCK_SIZE 16

__global__ void matvec(const float *A, const float *x, float *y)
{
	tid = threadIdx.x + blockIdx.x*blockDim.x; 
	

}
