#include<iostream>
#include<vector>

#include<hip/hip_runtime.h>
#include<hipblas.h>

__global__ void saxpy(const float x[], float y[], const float alpha)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    y[tid] += alpha*x[tid];
}

int main()
{
    int N = std::pow(2,30);
    float time_instance, my_saxpy_time=0.f, hipblas_saxpy_time=0.f;
    double my_saxpy_bandwidth, hipblas_saxpy_bandwidth;

    std::vector<float> host_x(N, 1.f); 
    std::vector<float> host_y(N, 1.f);
    float alpha = 2.5f;
    
    float *d_x, *d_y, *d_x1;

    hipMalloc(&d_x, N*sizeof(float));
    hipMalloc(&d_y, N*sizeof(float));
    hipMalloc(&d_x1, N*sizeof(float));
   
    hipMemcpy(d_x, host_x.data(), N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_x1, host_x.data(), N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, host_y.data(), N*sizeof(float), hipMemcpyHostToDevice);
   
    dim3 dimGrid(N/256);
    dim3 dimBlock(256);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    for(int i = 0; i < 10; i++){
        hipEventRecord(start);
        saxpy<<<dimGrid, dimBlock>>>(d_x, d_y, alpha);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&time_instance, start, stop);
        if( i > 0) my_saxpy_time += time_instance;
    }
    my_saxpy_time /= 9; 

    hipblasHandle_t handle;
    hipblasCreate(&handle);
    for(int i = 0; i < 10; i++){
        hipEventRecord(start);
        hipblasSaxpy(handle, N, &alpha, d_x1, 1, d_y, 1);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&time_instance, start, stop);
        if( i > 0) hipblas_saxpy_time += time_instance; 
    }

    hipblas_saxpy_time /= 9;

    hipEventDestroy(start);
    hipEventDestroy(stop);
    
    hipFree(d_x);
    hipFree(d_x1);
    hipFree(d_y);

    std::cout<< "My saxpy time = " << my_saxpy_time << " ms" << std::endl;
    std::cout<< "HipBlas saxpy time = " << hipblas_saxpy_time << " ms" << std::endl;

    long long int numer     = (long long int)(N)*2*3;
    my_saxpy_bandwidth      = numer/(my_saxpy_time*1e6);
    hipblas_saxpy_bandwidth = numer/(hipblas_saxpy_time*1e6);

    std::cout<< "My saxpy Bandwidth = " << my_saxpy_bandwidth << " GB/s" << std::endl;
    std::cout<< "HipBlas saxpy Bandwidth = " << hipblas_saxpy_bandwidth << " GB/s" << std::endl;

}
