#include <iostream>
#include <stdlib.h>
#include <ctime>

typedef struct{
	int width;
	int height;
	float* elements;
} Matrix; 

__global__ void MatAddKernel(Matrix A, Matrix B, Matrix C)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x; //thread in x
	int idy = threadIdx.y + blockDim.y*blockIdx.y; //thread in y
	int tid = idx + A.width*idy; 

	if(idx < A.width && idy < A.height)
	{
		C.elements[tid] = A.elements[tid] + B.elements[tid]; 
	}
}

void MatAdd(const Matrix A, const Matrix B, Matrix C)
{
	Matrix d_A, d_B, d_C; 
	d_A.width = A.width;
	d_A.height = A.height;
	d_B.width = B.width;
	d_B.height = B.height; 
	size_t size = A.width*A.height*sizeof(float); 
	cudaMalloc(&d_A.elements,size);
        cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	size = B.width * B.height * sizeof(float);
        cudaMalloc(&d_B.elements,size);
        cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	//Allocate C in device memory
        d_C.width =  C.width; d_C.height = C.height;
        size = C.width * C.height * sizeof(float);
        cudaMalloc(&d_C.elements, size);

	dim3 dimBlock(16, 16); 
	dim3 dimGrid(A.width/dimBlock.x, A.height/dimBlock.y);
	
	float gpuElapsedTime;
	clock_t start, stop; 
	cudaEvent_t gpuStart, gpuStop;
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuStop);
	cudaEventRecord(gpuStart, 0);
	//Matrix Addition
	start = clock();
	MatAddKernel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C); 
	cudaEventRecord(gpuStop,0);
	cudaDeviceSynchronize();
	stop = clock();
	std::cout<< (double(stop) - double(start))/CLOCKS_PER_SEC << std::endl;
	cudaEventSynchronize(gpuStop);
	cudaEventElapsedTime(&gpuElapsedTime, gpuStart, gpuStop); //time in milliseconds
	cudaEventDestroy(gpuStart);
	cudaEventDestroy(gpuStop);

	std::cout<<"GPU TIME ELAPSED = " << gpuElapsedTime <<"ms" << std::endl;
		
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost); 

	//Free Memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

//Main program 
int main()
{
// Set up matrices

	Matrix A, B, C; 
	int N = 8192;
	int M = 8192;

	A.width = N;
	B.width = N; 
	C.width = N; 
	
	A.height = M;
	B.height = M;
	C.height = M;

	
	size_t asize = A.width * A.height * sizeof(float);
	size_t bsize = B.width * B.height * sizeof(float);
	size_t csize = C.width * C.height * sizeof(float);
	
	A.elements = (float*)malloc(asize);
	B.elements = (float*)malloc(bsize);
	C.elements = (float*)malloc(csize);
	for( int i = 0; i < A.height; i++){
		for( int j = 0; j < A.width; j++)
			{
			A.elements[i*A.width + j] = 1.0f;
			B.elements[i*B.width + j] = 1.0f;
			}
	}


	MatAdd(A,B,C);
        std::cout<<C.elements[0]<<std::endl;	

	free(A.elements);
	free(B.elements);
	free(C.elements);
}
