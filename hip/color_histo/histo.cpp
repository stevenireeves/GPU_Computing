#include <iostream>
#include <stdlib.h>
#include <hip/hip_runtime.h>

/* Kernel example of a bad implementation of histogram. 
   Inputs: Int array d_bins, Int array d_in, Int BIN_COUNT
   Ouput: Int array d_bins
*/
__global__ void faux_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
	int myId = threadIdx.x + blockDim.x*blockIdx.x; 
	int myItem = d_in[myId]; 
	int myBin = myItem % BIN_COUNT; 
	d_bins[myBin]++; 
}

/* Kernel, example of a simple but unoptimized implementation of histogram. 
   Inputs: Int array d_bins, Int array d_in, Int BIN_COUNT
   Ouput: Int array d_bins
*/
__global__ void simple_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
	int myId = threadIdx.x + blockDim.x*blockIdx.x; 
	int myItem = d_in[myId]; 
	int myBin = myItem % BIN_COUNT; 
	atomicAdd(&(d_bins[myBin]),1);
}

/* Kernel, example of a more optimized implementation of histogram using shared memory
   Inputs: Int array d_bins, Int array d_in, Int BIN_COUNT
   Ouput: Int array d_bins
*/
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

/*
    Driver function which compares implementations of histogram. 
*/
int main()
{
	int *bins, *in, *d_bins, *d_in; 
	const int ARRAY_SIZE = 65536;
	const int ARRAY_BYTES = ARRAY_SIZE*sizeof(int);
	const int BIN_COUNT = 16;
	const int BIN_BYTES = BIN_COUNT*sizeof(int);

	bins = new int[BIN_BYTES];
	in   = new int[ARRAY_BYTES];
	hipMalloc(&d_in, ARRAY_BYTES); 
	hipMalloc(&d_bins, BIN_BYTES);

	for(int i = 0; i < ARRAY_SIZE; i++)
		in[i] = i; 

	hipMemcpy(d_in, in, ARRAY_BYTES, hipMemcpyHostToDevice); 
	float gpuElapsedTime;
    hipEvent_t gpuStart, gpuStop;
    hipEventCreate(&gpuStart);
    hipEventCreate(&gpuStop);
    hipEventRecord(gpuStart, 0);
//  Launch Kernel
//	faux_histo<<<ARRAY_SIZE/BIN_COUNT, BIN_COUNT>>>(d_bins, d_in, BIN_COUNT);
//	simple_histo<<<ARRAY_SIZE/BIN_COUNT, BIN_COUNT>>>(d_bins, d_in, BIN_COUNT);
	smem_histo<<<ARRAY_SIZE/(32*BIN_COUNT), BIN_COUNT, BIN_BYTES>>>(d_bins, d_in, BIN_COUNT, ARRAY_SIZE);
	hipEventRecord(gpuStop,0);
    hipEventSynchronize(gpuStop);
    hipEventElapsedTime(&gpuElapsedTime, gpuStart, gpuStop); //time in milliseconds
    hipEventDestroy(gpuStart);
    hipEventDestroy(gpuStop);
	std::cout<< "Time Taken = " << gpuElapsedTime << "ms" << std::endl;
	hipMemcpy(bins,d_bins, BIN_BYTES, hipMemcpyDeviceToHost); 

	std::cout<< "Histogram =" <<std::endl;
	for(int i = 0; i < BIN_COUNT; i++)
		std::cout<< "Bin " << i << " = " << bins[i]<<std::endl;

    delete bins, in;
	hipFree(d_bins);
	hipFree(d_in);	
}
