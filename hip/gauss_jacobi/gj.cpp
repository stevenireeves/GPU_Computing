#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>

#include "mat.h"
#define N 1024 
#define BLOCK_SIZE 16 

/* 
    CPU function for Gauss-Jacobi algorithm
    Input: FP32 Matrix A, FP32 array x, FP32 array b, FP32 epsilon
    Output: FP32 array x
*/
void CpuGJ(const Matrix A, std::vector<float> &x, const std::vector<float> &b, float eps)
{
    float res = 1.0f; 
    float summ1, summ2;
    std::vector<float> temp(A.width, 0.f); 
    int counter = 0; 
    while(res > eps)
    {
        summ2 = 0.0f; 
        for(int i = 0; i < A.width; i++)
        {
            summ1 = 0.0f; 
            for(int k =0; k < A.width; k++)
                if(k!=i) summ1 += A.elements[k + i*A.width]*x[k]; 
            temp[i] = 1/A.elements[i+i*A.width]*(b[i] - summ1); 
            summ2 += abs(temp[i] - x[i]);  
        }
        for(int i = 0; i < A.width; i++) x[i] = temp[i]; 
        res = summ2;
        counter++; 
        if(counter==A.width)
            break; 
    }
    std::cout<<"Steps Taken to Convergence = "<< counter<<std::endl;
}

/* Function to load elements from filesystem into Matrix. */
void LoadMatrix(std::string file, Matrix A)
{
    std::ifstream f;
    f.open(file);
    for( int i = 0; i <A.height; i++)
            for(int j = 0; j < A.width; j++)
            {
                    f >> A.elements[j + A.width*i];
            }
    f.close();
}


/* 
    Kernel: Gauss-Jacobi algorithm that uses shared memory
    Inputs: FP32 Matrix A, FP32 array x, FP32 array xout, FP32 array b
    Output: FP32 array xout
*/
__global__ void SharedGJ(const Matrix A, const float x[], float xOut[], const float b[]) //Computes one iteration of GJ
{
    int row = threadIdx.x;
    int tidx = row + blockIdx.x*blockDim.x;
    if (tidx >= A.height)
            return; // thread outside bounds.
    __shared__ float ASub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float xSub[BLOCK_SIZE];
    float yVal = 0.0f; 
    for (int block = 0; block < (A.width+BLOCK_SIZE -1)/BLOCK_SIZE; block++)
    {
            // grab shared local data for operations
            for(int j = 0; j < BLOCK_SIZE; j++)
                ASub[row][j] = A.elements[tidx*A.width + block*BLOCK_SIZE + j ];
            xSub[row] = x[block * BLOCK_SIZE + row];
            // sync threads, all are ready now to compute
            __syncthreads ();

            // multiply sub matrix and sub vector
            for (int e = 0; e < BLOCK_SIZE; e++){
                    int tileId = block*BLOCK_SIZE + e; 
                    if(tileId!=tidx){
                        yVal +=  ASub[row][e] * xSub[e];
                    }
            }
            __syncthreads ();
    }
    xOut[tidx] = 1.0f/A.elements[tidx + tidx*A.width]*(b[tidx] - yVal);
}

/* 
    Kernel: Unoptimized Gauss-Jacobi algorithm
    Inputs: FP32 Matrix, FP32 array x, FP32 array xout, FP32 array b
    Output: FP32 array xout
*/
__global__ void NaiveGJ(const Matrix A, const float x[], float xOut[], const float b[]) //Computes one iteration of GJ
{
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	float summ1 = 0.0f; 
	for (int k =0; k < A.width; k++)
	{
		if(k!= gid)
			summ1 += A.elements[k + gid*A.width]*x[k]; //dot product 
	} 
	xOut[gid] = 1.0f/A.elements[gid + gid*A.width]*(b[gid] - summ1);
}

/* 
    Kernel: Compute the residual between iterations
    Inputs: FP32 array xold, FP32 array xnew
    Output: FP32 array xold
*/
__global__ void ComputeR(float xOld[], const float xNew[]) //store abs(diff) in xold
{
	int gid = threadIdx.x + blockDim.x*blockIdx.x; 
	xOld[gid] = fabs(xNew[gid] - xOld[gid]);
}

/*
    Kernel: Computes the sum reduction of the residual
    Inputs: FP32 array d_out, FP32 array d_in
    Output: FP32 array d_out
*/
__global__ void ReduceR(float dOut[], const float dIn[])
{
    // sdata is allocated in the kernel call: via dynamic shared memeory
    extern __shared__ float sData[];

    int myId = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x;

    //load shared mem from global mem
    sData[tid] = dIn[myId];
    __syncthreads(); // always sync before using sdata

    //do reduction over shared memory
    for(int s = blockDim.x/2; s>0; s >>=1)
    {
        if(tid < s)
        {
           sData[tid] += sData[tid + s];
        }
        __syncthreads(); //make sure all additions are finished
    }

    //only tid 0 writes out result!
    if(tid == 0)
    {
       dOut[blockIdx.x] = sData[0];
    }
}

/*
    Kernel: Fills xout with xin's contents
    Inputs: FP32 array xout, FP32 array xin
    Output: FP32 array xout
*/
__global__ void fill(float xOut[], const float xIn[])
{
	int gid = threadIdx.x + blockDim.x*blockIdx.x; 
	xOut[gid] = xIn[gid];
}

/*
    Driver frunction for Gauss-Jacobi solver
    Inputs: FP32 Matrix A, FP32 array x, FP32 array b, FP32 scalar eps
    Output: FP32 array x
*/
void ParGJ(const Matrix A, std::vector<float> &x, const std::vector<float> &b, float eps)
{
    float res = 1.0f;
    int counter = 0;
    Matrix dA(A.width, A.height, 1);
    float *dX, *dB, *dXNew;
    hipMalloc(&dX, A.width*sizeof(float));
    hipMalloc(&dB, A.height*sizeof(float));
    hipMalloc(&dXNew, A.width*sizeof(float));

    hipMemcpy(dA.elements,A.elements,A.width*A.height*sizeof(float),hipMemcpyHostToDevice);
    hipMemcpy(dX, x.data(), A.width*sizeof(float),hipMemcpyHostToDevice);
    hipMemcpy(dB, b.data(), A.height*sizeof(float),hipMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((A.width+ dimBlock.x - 1)/dimBlock.x);
    float time; 
    hipEvent_t start, stop; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start);         
    while(res>eps)
    {
        //Compute x^{n+1}
//        naive_gj<<<dimGrid, dimBlock>>>(d_A, d_x, d_xnew, d_b);
        SharedGJ<<<dimGrid, dimBlock>>>(dA, dX, dXNew, dB);

        //Compute vector of residuals
        ComputeR<<<dimGrid,dimBlock>>>(dX, dXNew); //Store r in d_x

        //Reduce vector of residuals to find norm
        ReduceR<<<1,N, N*sizeof(float)>>>(dX, dX);
        hipMemcpy(&res, dX, sizeof(float), hipMemcpyDeviceToHost);
        std::cout<<res<<std::endl; 
        //X = Xnew
        fill<<<dimGrid,dimBlock>>>(dX, dXNew);
        counter++;
        if(counter==A.width)
           break;
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&time, start, stop); 
	std::cout<<"Steps Taken to Convergence = "<< counter<<std::endl;
    std::cout<<"Time for execution = " << time <<"ms" << std::endl; 
    //export X
    hipMemcpy(x.data(), dX, A.width*sizeof(float), hipMemcpyDeviceToHost);
    hipFree(dX);
    hipFree(dXNew);
    hipFree(dB);
}

int main()
{
// Matrix stuff! 
	Matrix A(N, N); 
	LoadMatrix("matrix.dat", A);

// Vector stuff!
    std::vector<float> x(N, 0.f); 
    std::vector<float> b(N, 1.f); 

// Gauss-Jacobi Parameters
	float eps = 1e-7; 	

// Call the Gauss-Jacobi algorithms
	ParGJ(A, x, b, eps); 

	std::cout<<"Soln X = "<<std::endl;
	for(int i = 0; i <10; i++)
		std::cout<< x[i] <<std::endl; //  */
}

