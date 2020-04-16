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

/* Use Matrix Class! */
#include "mat.h"

// Thread block size
#define BLOCK_SIZE 32




// Forward declaration of the mat mul kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix); 
__global__ void naivekernel(const Matrix, const Matrix, Matrix); 

// Matrix multiplication host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE


/* Shared Matrix Multiplication Routines */ 

/* MatMul with shared memory 
   :inputs: Matrix A, Matrix B
   :outputs: Matrix C = AB
*/ 
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    int Gpu = 1; 
    int toDev = 1, fromDev = 2; 
	//Load A and B to device memory 
    //Allocate Matrix C
	Matrix d_A(A.width, A.height, A.stride, Gpu);
    Matrix d_B(B.width, B.height, B.stride, Gpu);
    Matrix d_C(C.width, C.height, C.stride, Gpu);
    d_A.load(A, toDev);
    d_B.load(B, toDev); 
	
    

	// Invoke Kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height/ dimBlock.y); 
    //Use Cuda Events for timing
    cudaEvent_t start, stop; 
    float time; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    cudaEventRecord(start, 0); 

	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&time, start, stop); 
    std::cout<< " Shared Memory Matrix Multiplication time =" << '\t' 
             << time << "ms" << std::endl; 

	// Read C from Device memory 
    C.load(d_C, fromDev);
	
	//Free device memory 
    d_A.dealloc(Gpu);
    d_B.dealloc(Gpu);
    d_C.dealloc(Gpu);
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
	subMatrix Csub(C, BLOCK_SIZE,  blockRow, blockCol);

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
		subMatrix Asub(A, BLOCK_SIZE, blockRow, m);

		//Get B submatrix 
		subMatrix Bsub(B, BLOCK_SIZE, m ,blockCol);  
		

		//Load Asub and Bsub from global memory into shared; 

		As[row][col] = Asub.GetElem(row,col);
		Bs[row][col] = Bsub.GetElem(row,col); 

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
	Csub.SetElem(row, col, Cvalue);
}

__global__ void naivekernel(const Matrix A, const Matrix B, Matrix C)
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

void NaiveMatMul(const Matrix A, const Matrix B, Matrix C)
{

    int Gpu=1, toDev = 1, fromDev = 2; 
	//Load A and B to device memory
	Matrix d_A(A.width, A.height,0, Gpu);
    d_A.load(A, toDev); 
	Matrix d_B(B.width, B.height,0, Gpu);
    d_B.load(B, toDev); 

	// Allocate C in device memory
	Matrix d_C(C.width, C.height,0, Gpu);

	// Invoke kernel 
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

    // Use cudaEvet type for timing
    cudaEvent_t start, stop; 
    float elapsed_secs; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    cudaEventRecord(start, 0); 

	naivekernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&elapsed_secs, start, stop); 

	std::cout<<" Naive GPU MatMul Time = "<< elapsed_secs << "ms" << std::endl;
	// Read C from device memory 
    C.load(d_C, fromDev); 
	// Free device memory 
    d_A.dealloc(Gpu);
    d_B.dealloc(Gpu);
    d_C.dealloc(Gpu); 
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
    int Cpu = 0;
	int N = 1024;
	int M = 1024;

    Matrix A(N, M, N, Cpu), B(M, N, M, Cpu), C(N, N, N, Cpu);
    Matrix Ds(N, M, N, Cpu), D(N,M,N, Cpu);
    Matrix nC(N, N, N, Cpu); 
	

	//set values for A and B 
	for( int i = 0; i < A.height; i++){
		for( int j = 0; j < A.width; j++)
			{
                A.elements[i*A.stride + j] = 1.0f;
                B.elements[i*B.stride + j] = 1.0f;
			}
	}


// Call matrix multiplication. 

//Serial 
	clock_t sstart = clock();	//Serial Start
	serialMatMul(A,B,Ds);
	clock_t send = clock(); 	//Serial End
	double serial = double(send - sstart) / CLOCKS_PER_SEC;	
	std::cout<< " Serial Time = " << serial << "s" << std::endl;

//OpenMP
	clock_t begin = clock();	
	CPUMatMul(A,B,D);
	clock_t end = clock();
	double fullcpu = double(end - begin) / (CLOCKS_PER_SEC*12);
	std::cout<< " CPU Time = " << fullcpu << "s" << std::endl; //*/

//Naive CUDA
	NaiveMatMul(A,B,nC);

//SharedMemCUDA
	MatMul(A,B,C);
	

//Deallocate Memory
    A.dealloc();
    B.dealloc();
    C.dealloc();
    Ds.dealloc();
    D.dealloc();
    nC.dealloc(); 
}
