#include <hip/hip_runtime.h>

#define  blockSize 1024

//multiple elements per thread
template <class T>
__global__ void reduce(T *d_out, const T *d_in, unsigned int n)
{
    // sdata is allocated in the kernel call: via dynamic shared memeory
    extern __shared__ T sdata[];

    int myId = threadIdx.x + (blockDim.x*2)*blockIdx.x;
    int tid = threadIdx.x;
	int gridSize = blockDim.x*2*gridDim.x; 
	sdata[tid] = 0; 

        //load shared mem from global mem
	while(myId < n)
	{
		sdata[tid] += d_in[myId] + d_in[myId + blockDim.x]; 
		myId += gridSize; 
	}
	__syncthreads(); 

        //do reduction over shared memory
	
	if(blockSize >= 512){
		if(tid < 256)
		{
			sdata[tid] += sdata[tid + 256]; 
		}
		__syncthreads();

	}
	if(blockSize >= 256){
		if(tid < 128)
		{
			sdata[tid] += sdata[tid + 128]; 
		}
		__syncthreads();

	}
	if(blockSize >= 128){
		if(tid < 64)
		{
			sdata[tid] += sdata[tid + 64]; 
		}
		__syncthreads();

	}


	if(tid < 32)
	{
	   if(blockSize >= 64) sdata[tid] += sdata[tid+32];
       if(blockSize >= 32) sdata[tid] += sdata[tid+16];
       if(blockSize >= 16) sdata[tid] += sdata[tid+8];
       if(blockSize >= 8) sdata[tid] += sdata[tid+4];
       if(blockSize >= 4) sdata[tid] += sdata[tid+2];
       if(blockSize >= 2) sdata[tid] += sdata[tid+1];
	}

    //only tid 0 writes out result!
    if(tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}
