#include <iostream>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <vector>
/* Kernel example of a bad implementation of histogram. 
   Inputs: Int array d_bins, Int array d_in, Int BIN_COUNT
   Ouput: Int array d_bins
*/
__global__ void FauxHisto(unsigned int *dBins, const int *dIn, const int binCount)
{
	int myId = threadIdx.x + blockDim.x*blockIdx.x; 
	int myItem = dIn[myId]; 
	int myBin = myItem % binCount; 
	dBins[myBin]++; 
}

/* Kernel, example of a simple but unoptimized implementation of histogram. 
   Inputs: Int array d_bins, Int array d_in, Int BIN_COUNT
   Ouput: Int array d_bins
*/
__global__ void SimpleHisto(unsigned int *dBins, const int *dIn, const int binCount)
{
	int myId = threadIdx.x + blockDim.x*blockIdx.x; 
	int myItem = dIn[myId]; 
	int myBin = myItem % binCount; 
	atomicAdd(&(dBins[myBin]),1);
}

/* Kernel, example of a more optimized implementation of histogram using shared memory
   Inputs: Int array d_bins, Int array d_in, Int BIN_COUNT
   Ouput: Int array d_bins
*/
__global__ void SmemHisto(unsigned int *dBins, const int *dIn, const int binCount, const int size)
{
    //Create Private copies of histo[] array; 
    extern __shared__ unsigned int histoLDS[];

    int tid = threadIdx.x;
    if(tid < binCount)
       histoLDS[tid] = 0;
    __syncthreads(); 

    int gid = threadIdx.x + blockDim.x*blockIdx.x;
    int buffer = gid % binCount; 
    buffer = 0;
    atomicAdd(&(histoLDS[buffer]), 1);
    __syncthreads();

    //Build Final Histogram using private histograms.

    if(tid < binCount)
    {
       atomicAdd(&(dBins[tid]), histoLDS[tid]);
    }
}

/*
    Driver function which compares implementations of histogram. 
*/
int main()
{
    using uint = unsigned int;
    uint *dBins;
	int *dIn; 
	const int arraySize  = 65536;
	const int arrayBytes = arraySize*sizeof(int);
	const int binCount   = 16;
	const int binBytes   = binBytes*sizeof(uint);

    std::vector<uint> bins(binCount, 0);
    std::vector<int> in(arraySize);
	hipMalloc(&dIn, arrayBytes); 
	hipMalloc(&dBins, binBytes);

	for(int i = 0; i < arraySize; i++)
		in[i] = i; 

	hipMemcpy(dIn, in.data(), arrayBytes, hipMemcpyHostToDevice); 
	hipMemcpy(dBins, bins.data(), binBytes, hipMemcpyHostToDevice);
	float gpuElapsedTime;
    hipEvent_t gpuStart, gpuStop;
    hipEventCreate(&gpuStart);
    hipEventCreate(&gpuStop);
    hipEventRecord(gpuStart, 0);
//  Launch Kernel
//	fauxHisto<<<arraySize/binCount, binCount>>>(dBins, dIn, binCount);
	SimpleHisto<<<arraySize/binCount, binCount>>>(dBins, dIn, binCount);
//	SmemHisto<<<arraySize/binCount, binCount, binBytes>>>(dBins, dIn, binCount, arraySize);
	hipEventRecord(gpuStop,0);
    hipEventSynchronize(gpuStop);
    hipEventElapsedTime(&gpuElapsedTime, gpuStart, gpuStop); //time in milliseconds
    hipEventDestroy(gpuStart);
    hipEventDestroy(gpuStop);
	std::cout<< "Time Taken = " << gpuElapsedTime << "ms" << std::endl;
	hipMemcpy(bins.data(), dBins, binBytes, hipMemcpyDeviceToHost); 

	std::cout<< "Histogram =" <<std::endl;
	for(int i = 0; i < binCount; i++)
		std::cout<< "Bin " << i << " = " << bins[i]<<std::endl;

	hipFree(dBins);
	hipFree(dIn);	
}
