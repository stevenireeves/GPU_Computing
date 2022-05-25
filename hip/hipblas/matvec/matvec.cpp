#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipblas.h>

#include "mat.h"

#define BLOCK_SIZE 32

/*
    Kernel: Matrix Vector Multiplication
    Input: Matrix A, FP32 array x, FP32 array y
    Output: FP32 array y
*/
__global__ void
matVec (Matrix A, float *x, float *y) {
    int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    if (tidx > A.width)
        return; // thread outside bounds.
	float yval = 0.0f; 
    for (int e = 0; e < A.width ; e++) {
        yval +=  A.elements[A.width * tidx + e]* x[e];
    }
    y[tidx] = yval;
}

/*
    Kernel: Matrix Vector Multiplication with LDS
    Input: Matrix A, FP32 array x, FP32 array y
    Output: FP32 array y
*/
__global__ void
smemMatVec (Matrix A, float *x, float *y) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    int tid = threadIdx.x;
    if (gid > A.width)
        return; // thread outside bounds.
    
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float xsub[BLOCK_SIZE];

	float yval = 0.0f;
    for (int i = 0; i < (A.width + BLOCK_SIZE -1)/BLOCK_SIZE; i++){
        for (int e = 0; e < BLOCK_SIZE; e++)
            Asub[tid][e] = A.elements[A.width * gid + (i*BLOCK_SIZE + e)];
        xsub[tid] = x[gid];
        __syncthreads(); 
        for (int e = 0; e < BLOCK_SIZE; e++) {
            yval +=  Asub[tid][e]* xsub[e];
        }
    }
    y[gid] = yval;
}



int main()
{
    int N = std::pow(2,10);
    float time_instance, my_mv_time=0.f, smem_mv_time=0.f, hipblas_mv_time=0.f;
    double my_mv_bandwidth, smem_mv_bandwidth, hipblas_mv_bandwidth;

    std::vector<float> host_x(N, 1.f); 
    Matrix A(N, N);
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            A.elements[i*A.width + j] = 1.f;
        

    float *d_x, *d_y, *d_x1;

    Matrix d_A(N, N, 1);    
    hipMalloc(&d_x, N*sizeof(float));
    hipMalloc(&d_y, N*sizeof(float));
    hipMalloc(&d_x1, N*sizeof(float));
   
    hipMemcpy(d_x, host_x.data(), N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_x1, host_x.data(), N*sizeof(float), hipMemcpyHostToDevice);
    d_A.load(A, 1);   

    dim3 dimGrid(N/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    for(int i = 0; i < 10; i++){
        hipEventRecord(start);
        matVec<<<dimGrid, dimBlock>>>(d_A, d_x, d_y);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&time_instance, start, stop);
        if( i > 0) my_mv_time += time_instance;
    }
    my_mv_time /= 9; 

    for(int i = 0; i < 10; i++){
        hipEventRecord(start);
        smemMatVec<<<dimGrid, dimBlock>>>(d_A, d_x1, d_y);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&time_instance, start, stop);
        if( i > 0) smem_mv_time += time_instance;
    }
    smem_mv_time /= 9; 

    hipblasHandle_t handle;
    hipblasCreate(&handle);
    hipblasOperation_t op = HIPBLAS_OP_N;
    float alpha = 1.f, beta = 1.f;
    
    for(int i = 0; i < 10; i++){
        hipEventRecord(start);
        hipblasSgemv(handle, op, N, N, &alpha, d_A.elements, N, d_x1, 1, &beta, d_y, 1);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&time_instance, start, stop);
        if( i > 0) hipblas_mv_time += time_instance; 
    }
    hipblas_mv_time /= 9;

    hipEventDestroy(start);
    hipEventDestroy(stop);
    
    hipFree(d_x);
    hipFree(d_x1);
    hipFree(d_y);
    d_A.gpu_deallocate();

    using lint = long long int;
    lint numer     = (lint)(N)*(lint)(N) + lint(2*N);
    my_mv_bandwidth      = 4*numer/(my_mv_time*1e6);
    smem_mv_bandwidth      = 4*numer/(smem_mv_time*1e6);
    hipblas_mv_bandwidth = 4*numer/(hipblas_mv_time*1e6);


    std::cout<< "matVec time = " << my_mv_time << " ms" << std::endl;
    std::cout<< "matVec Bandwidth = " << my_mv_bandwidth << " GB/s" <<'\n' << std::endl;

    std::cout<< "My smemMatVec time = " << smem_mv_time << " ms" << std::endl;
    std::cout<< "smemMatVec Bandwidth = " << smem_mv_bandwidth << " GB/s" << '\n' << std::endl;

    std::cout<< "HipBlas Sgemv time = " << hipblas_mv_time << " ms" << std::endl;
    std::cout<< "hipblasSgemv Bandwidth = " << hipblas_mv_bandwidth << " GB/s" << std::endl;
}
