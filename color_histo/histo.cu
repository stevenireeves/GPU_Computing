#include <iostream>
#include <stdlib.h>

__global__ void faux_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
	int myId = threadIdx.x + blockDim.x*blockIdx.x; 
	int myItem = d_in[myId]; 
	int myBin = myItem % BIN_COUNT; 
	d_bins[myBin]++; 
}

__global__ void simple_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
	int myId = threadIdx.x + blockDim.x*blockIdx.x; 
	int myItem = d_in[myId]; 
	int myBin = myItem % BIN_COUNT; 
	atomicAdd(&(d_bins[myBin]),1);
}


__global__ void smem_histo(int *d_bins, const int *d_in, const int BIN_COUNT, const int size)
{
        //Create Private copies of histo[] array; 
        extern __shared__ unsigned int histo_private[];

        int tid = threadIdx.x;
        if(threadIdx.x < BIN_COUNT)
                histo_private[tid] = 0;
        __syncthreads();

        int i = threadIdx.x + blockDim.x*blockIdx.x;
        //stride total number of threads
	int stride = blockDim.x*gridDim.x;
        while( i < size)
        {
                int buffer = i % BIN_COUNT; 
                atomicAdd(&(histo_private[buffer]), 1);
                i += stride;
        }
        __syncthreads();

        //Build Final Histogram using private histograms.

        if(tid < BIN_COUNT)
        {
                atomicAdd(&(d_bins[tid]), histo_private[tid]);
        }
}

int main()
{
	int *bins, *in, *d_bins, *d_in; 
	const int ARRAY_SIZE = 65536;
	const int ARRAY_BYTES = ARRAY_SIZE*sizeof(int);
	const int BIN_COUNT = 16;
	const int BIN_BYTES = BIN_COUNT*sizeof(int);

	bins = (int*)malloc(BIN_BYTES);
	in = (int*)malloc(ARRAY_BYTES);
	cudaMalloc(&d_in, ARRAY_BYTES); 
	cudaMalloc(&d_bins, BIN_BYTES);

	for(int i = 0; i < ARRAY_SIZE; i++)
		in[i] = i; 

	cudaMemcpy(d_in, in, ARRAY_BYTES, cudaMemcpyHostToDevice); 
	 float gpuElapsedTime;
        cudaEvent_t gpuStart, gpuStop;
        cudaEventCreate(&gpuStart);
        cudaEventCreate(&gpuStop);
        cudaEventRecord(gpuStart, 0);
       	//Launch Kernel
//	faux_histo<<<ARRAY_SIZE/BIN_COUNT, BIN_COUNT>>>(d_bins, d_in, BIN_COUNT);
//	simple_histo<<<ARRAY_SIZE/BIN_COUNT, BIN_COUNT>>>(d_bins, d_in, BIN_COUNT);
	smem_histo<<<ARRAY_SIZE/(32*BIN_COUNT), BIN_COUNT, BIN_BYTES>>>(d_bins, d_in, BIN_COUNT, ARRAY_SIZE);
	cudaEventRecord(gpuStop,0);
        cudaEventSynchronize(gpuStop);
        cudaEventElapsedTime(&gpuElapsedTime, gpuStart, gpuStop); //time in milliseconds
        cudaEventDestroy(gpuStart);
        cudaEventDestroy(gpuStop);
	std::cout<< "Time Taken = " << gpuElapsedTime << "ms" << std::endl;
	cudaMemcpy(bins,d_bins, BIN_BYTES, cudaMemcpyDeviceToHost); 

	std::cout<< "Histogram =" <<std::endl;
	for(int i = 0; i < BIN_COUNT; i++)
		std::cout<< "Bin " << i << " = " << bins[i]<<std::endl;

	free(bins);
	free(in);
	cudaFree(d_bins);
	cudaFree(d_in);	
}
