#include <iostream>
#include <stdlib.h>
#include "mat.h"

/* 
    Kernel: Adds two matrices and stores them in a third.
    A, B inputs
    C output 
*/ 
__global__ void MatAddKernel(const Matrix A, const Matrix B, Matrix C)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x; //thread in x
	int idy = threadIdx.y + blockDim.y*blockIdx.y; //thread in y
	int tid = idx + A.width*idy; // Memory is 1D 

	if(idx < A.width && idy < A.height)
	{
		C.elements[tid] = A.elements[tid] + B.elements[tid]; 
	}
}

/* 
    Driver for launching MatAddKernel
*/
void MatAdd(const Matrix A, const Matrix B, Matrix C)
{
    int Gpu = 1; 
    //Use Copy Constructor to allocate and copy from host to device
    int w = A.width, h = A.height; 
    Matrix d_A(w, h, Gpu);
    d_A.load(A, Gpu); /*Cuda Memcpy */ 

    Matrix d_B(w, h, Gpu); 
    d_B.load(B, Gpu);

    Matrix d_C(w, h, Gpu); 
 
    dim3 dimBlock(16, 16); 
    dim3 dimGrid(A.width/dimBlock.x, A.height/dimBlock.y);
    MatAddKernel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C); 
    hipMemcpy(C.elements, d_C.elements, C.width*C.height*sizeof(float), hipMemcpyDeviceToHost); 

	//Free Memory
    d_A.gpu_deallocate();
    d_B.gpu_deallocate(); 
    d_C.gpu_deallocate(); 
}


//Main program 
int main()
{
// Set up matrices

    int N = 8192;
    int M = 8192;
    Matrix A(M,N), B(M,N), C(M,N); 

	for( int i = 0; i < A.height; i++){
		for( int j = 0; j < A.width; j++)
			{
    			A.elements[i*A.width + j] = 1.0f;
	    		B.elements[i*B.width + j] = 1.0f;
			}
	}

    MatAdd(A,B,C);
    A.cpu_deallocate();
    B.cpu_deallocate();
    C.cpu_deallocate();
}
