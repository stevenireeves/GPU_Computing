#include <iostream>
#include <cmath>

#include <hip/hip_runtime.h>

#include "reduce.h"

__global__ void fill(float x[])
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    x[tid] = (float)(tid % 16)/16.f;
}

template <typename T>
__global__ void saxpy(T x[], T y[], T alpha)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    y[tid] += alpha*x[tid];
}

int main()
{
    hipDeviceProp_t devProp0, devProp1;
    hipGetDeviceProperties(&devProp0, 0);
    hipGetDeviceProperties(&devProp1, 1);
    std::cout<< devProp0.name << '\t' << devProp1.name << std::endl; 
    int N = std::pow(2,20);
    int shared_buffer = 1024*sizeof(float);
    float *d_x0, *d_x1;
    float *d_r0, *d_r1, *d_rc;
    dim3 dimBlock(1024);
    dim3 dimGrid(N/1024);
    dim3 gridReduc(dimGrid.x/64);
    hipSetDevice(0);
    hipMalloc(&d_rc, 1024*sizeof(float));
    hipMalloc(&d_x0, N*sizeof(float));
    fill<<<dimGrid, dimBlock>>>(d_x0);

    hipSetDevice(1);
    hipMalloc(&d_x1, N*sizeof(float));
    fill<<<dimGrid, dimBlock>>>(d_x1);
    
    hipSetDevice(0);
    hipMalloc(&d_r0, 1024*sizeof(float));
    reduce<<<gridReduc, dimBlock, shared_buffer>>>(d_r0, d_x0, N);

    hipSetDevice(1);
    hipMalloc(&d_r1, 1024*sizeof(float));
    reduce<<<gridReduc, dimBlock, shared_buffer>>>(d_r1, d_x1, N);

    hipSetDevice(0);
    unsigned int flags;
    hipDeviceEnablePeerAccess(1, flags);
    std::cout<< flags << std::endl;
    hipMemcpyPeer(d_rc, 0, d_r1, 1, 1024*sizeof(float));
    saxpy<<<1, 1024>>>(d_r0, d_rc, 1.f);
    reduce<<<1, 1024, shared_buffer>>>(d_rc, d_rc, 1024);
    float val;
    hipMemcpy(&val, d_rc, sizeof(float), hipMemcpyDeviceToHost);
    std::cout<< "Reduction value = " << val << std::endl;

    hipFree(d_x0);
    hipFree(d_x1);
    hipFree(d_r0);
    hipFree(d_r1);
    hipFree(d_rc);
}
