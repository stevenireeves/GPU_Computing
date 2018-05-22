/*Matmul routine for AMS148, written by Steven Reeves, March 10 2018,
  major routines referenced from CUDA Programming Guide. */

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstring>
#include <ctime>
#include <omp.h>

// Thread block size
#define BLOCK_SIZE 32

//Struct for shared matrix multiplication
typedef struct{
	int width;
	int height;
	int stride;
	float* elements;
} Matrix;

//Struct for naive implementation
typedef struct{
	int width; 
	int height;
	float* elements;	
} nMatrix;

//Get matrix element
__device__ void SetElement(Matrix A, int row, int col, float value)
	{
		 A.elements[row * A.stride + col] = value; 
	}

//Set a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
	{
		return A.elements[row*A.stride + col];
	}

// Get the BLOCK_SIZE x BLOCK_SIZE submatrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
//  from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
	{ 
		Matrix Asub; 
		Asub.width = BLOCK_SIZE;
		Asub.height = BLOCK_SIZE; 
		Asub.stride = A.stride; 
		Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col]; 
		return Asub;
	 }


// Forward declaration of the mat mul kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix); 
__global__ void naivekernel(const nMatrix, const nMatrix, nMatrix); 

// Matrix multiplication host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{

	//Load A and B to device memory 
	Matrix d_A, d_B, d_C;
	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float); 
	cudaMalloc(&d_A.elements,size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice); 
	
	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float); 
	cudaMalloc(&d_B.elements,size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice); 
	
	//Allocate C in device memory
	d_C.width = d_C.stride = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float); 
	cudaMalloc(&d_C.elements, size); 

	// Invoke Kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height/ dimBlock.y); 
	clock_t begin = clock();
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); 
	cudaDeviceSynchronize();
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout<<"Run Time! "<< elapsed_secs << std::endl;

	// Read C from Device memory 
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost); 
	
	//Free device memory 

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// Matrix Multiplication Kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	//Static shared memory for Asub and Bsub
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE]; //Great name for an array


	// Block row and column;
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	//Thread block computes one sub matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	// Each thread computes one element of Csub
	// By accumulating results into Cvalue
	float Cvalue = 0.0f; 

	//Thread row and column index within the submatrix
	int row = threadIdx.y;
	int col = threadIdx.x; 

	// Loop over submatrices of A and B that are required for Csub
	//Multiply each pair of sub-matrices together
	//and summ the results
	for (int m = 0; m < (A.width/BLOCK_SIZE); m++){
		
		//Get A submatrix
		Matrix Asub = GetSubMatrix(A, blockRow, m);

		//Get B submatrix 
		Matrix Bsub = GetSubMatrix(B, m ,blockCol); 
		

		//Load Asub and Bsub from global memory into shared; 

		As[row][col] = GetElement(Asub,row,col);
		Bs[row][col] = GetElement(Bsub,row,col);

		//Always sync threads when loading shared memory before doing computation
		__syncthreads();

		//Multiply the submatrices
		for (int e = 0; e < BLOCK_SIZE; e++)
			Cvalue += As[row][e]*Bs[e][col];

		//synchronize to make sure all threads are done computing
		__syncthreads();
	}
	//write Csub back into global memory 
	//each thread writes one element
	SetElement(Csub, row, col, Cvalue);
}

__global__ void naivekernel(nMatrix A, nMatrix B, nMatrix C)
{
	// Each Thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0.0f;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x; 
	for (int e = 0; e<A.width; e++)
		Cvalue += A.elements[row*A.width + e]*B.elements[e*B.width + col];
	C.elements[row*C.width + col] = Cvalue;
}

void NaiveMatMul(const nMatrix A, const nMatrix B, nMatrix C)
{

	//Load A and B to device memory
	nMatrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	nMatrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	// Allocate C in device memory
	nMatrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size); 
	// Invoke kernel 
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	clock_t start = clock();
	naivekernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	cudaDeviceSynchronize();
	clock_t end = clock();
	double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
	std::cout<<"Naive Run Time! "<< elapsed_secs << std::endl;
	// Read C from device memory 
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost); 
	// Free device memory 
	cudaFree(d_A.elements); 
	cudaFree(d_B.elements); 
	cudaFree(d_C.elements);
}

void serialMatMul(const Matrix A, const Matrix B, Matrix C)
{
	for(int i = 0; i < A.width; i++){
		for(int j = 0; j < B.height; j++)
		{
			float Cvalue = 0.0f;
			for(int k = 0; k < A.width; k++)
				Cvalue += A.elements[i*A.width + k]*B.elements[k*B.width + j];
			C.elements[i*C.width + j] = Cvalue;
		}
	}
}

void CPUMatMul(const Matrix A, const Matrix B, Matrix C)
{
int i ,j ,k;
#pragma omp parallel for private(j, k)
	for(i = 0; i < A.width; i++){
		for(j = 0; j < B.height; j++)
		{
			float Cvalue = 0.0f;
			for(k = 0; k < A.width; k++)
			{
				Cvalue += A.elements[i*A.width + k]*B.elements[k*B.width + j];
			}
			C.elements[i*C.width + j] = Cvalue;
		}
	}
}

//Main program 
int main()
{
// Set up matrices

	Matrix A, B, C, Ds, D; 
	nMatrix nA, nB, nC;
	int N = 1024;
	int M = 1024;

	A.width = N;
	B.width = N; 
	C.width = N; 
	nA.width = A.width;
	nB.width = B.width;
	nC.width = C.width;
	Ds.width = C.width;
	D.width = C.width;
	
	A.height = M;
	B.height = M;
	C.height = M;
	nA.height = A.height;
	nB.height = B.height;
	nC.height = C.height;
	Ds.height = C.height;
	D.height = C.height;


	A.stride = A.width;
	B.stride = B.width;
	C.stride = C.width;
	Ds.stride = C.width;
	D.stride = C.width;
	
	size_t asize = A.width * A.height * sizeof(float);
	size_t bsize = B.width * B.height * sizeof(float);
	size_t csize = C.width * C.height * sizeof(float);
	
	A.elements = (float*)malloc(asize);
	nA.elements = (float*)malloc(asize);
	B.elements = (float*)malloc(bsize);
	nB.elements = (float*)malloc(bsize);
	C.elements = (float*)malloc(csize);
	nC.elements = (float*)malloc(csize);
	Ds.elements = (float*)malloc(csize);
	D.elements = (float*)malloc(csize);
	

	//set values for A and B 
	for( int i = 0; i < A.height; i++){
		for( int j = 0; j < A.width; j++)
			{
			A.elements[i*A.stride + j] = 1.0f;
			nA.elements[i*A.width + j] = 1.0f;
			B.elements[i*B.stride + j] = 1.0f;
			nB.elements[i*B.width + j] = 1.0f;
			}
	}


// Call matrix multiplication. 

//Serial 
	clock_t sstart = clock();	//Serial Start
	serialMatMul(A,B,Ds);
	clock_t send = clock(); 	//Serial End
	double serial = double(send - sstart) / CLOCKS_PER_SEC;	
	std::cout<< " Serial Time = " << serial << std::endl;

//OpenMP
	clock_t begin = clock();	
	CPUMatMul(A,B,D);
	clock_t end = clock();
	double fullcpu = double(end - begin) / (CLOCKS_PER_SEC*8);
	std::cout<< " CPU Time = " << fullcpu << std::endl; //*/

//Naive CUDA
	NaiveMatMul(nA,nB,nC);

//SharedMemCUDA
	MatMul(A,B,C);
	

//Deallocate Memory
	free(A.elements);
	free(B.elements);
	free(C.elements);
	free(Ds.elements);
	free(D.elements);
	free(nA.elements);
	free(nB.elements);
	free(nC.elements);
}
