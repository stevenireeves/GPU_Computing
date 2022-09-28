#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <hip/hip_runtime.h>

/* Device function to apply root(1-x^2), accepts float x, and returns float */
inline __device__ float RootOneMinusX(float x)
{
	return sqrtf(1.0f - x*x); 
}

/*  Applies RootOneMinusX to to data generated from xbeg to xbeg + n*dx 
    computes Reimann Rectangles for numerical integral
    and inputs into float pointer f1.
    Inputs: xbeg float, dx float, n int, float array f1.
    Outputs: float array f1.
 */
__global__ void GenTrapezoids(float xBeg, float dx, int n, float *f1)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x; 
	if(tid >= n) return; 

	float2 x; 
	x.x = xBeg + tid*dx; 
	x.y = xBeg + (tid + 1)*dx; 
	f1[tid] = RootOneMinusX(x.x)*dx*0.5f + RootOneMinusX(x.y)*dx*0.5f;
}

/*  Computes the reduction of FP32 array d_in. 
    Ouput is FP32 array d_out. If input array is larger than 1024 floats
    A partial reduction is computed to the 1024th entries of d_out. 
    If less, the full reduction can be found in d_out[0].
    In this application d_in contains the Reimainn rectangles 
    to compute the numerical integral.  
*/
__global__ void ShmemReduceKernel(float * dOut, const float *dIn)
{
    // sdata is allocated in the kernel call: via dynamic shared memeory
    extern __shared__ float sData[];
    
    int myId = threadIdx.x + blockDim.x*blockIdx.x;
    int tId = threadIdx.x;
    
    //load shared mem from global mem
    sData[tId] = dIn[myId];
    __syncthreads(); // always sync before using sdata
    
    //do reduction over shared memory
    for(int s = blockDim.x/2; s>0; s >>=1)
    {
        if(tId < s)
        {
            sData[tId] += sData[tId + s];
        }
        __syncthreads(); //make sure all additions are finished
    }
    
    //only tid 0 writes out result!
    if(tId == 0)
    {
            dOut[blockIdx.x] = sData[0];
    }
}

/* Host code to compute the digits of pi.
   Inputs: n int, the resolution of the integral to calculate pi
   Outputs: pi float, the estimate of pi.
*/ 
float MmmPi(int n)
{
    //Initialization
    float value; 
    float *dData;
    float *dReduc; 
    size_t original = n*sizeof(float);
    size_t reduc = n/(1024)*sizeof(float);
    
    //Allocation    
    hipMalloc(&dData, original);
    hipMalloc(&dReduc, reduc);
    
    //Kernel Parameters
    dim3 blockDim(1024, 1, 1);
    dim3 gridDim(n/blockDim.x, 1, 1);
    
    //integration parameters
    float xBeg = -1.0f;
    float dx = (1.0f - xBeg)/(float)n;

    dim3 mapGrid(n/blockDim.x + 1, 1,1); 
    //map+stencil kernel Note because of the shift we need more threads.
    GenTrapezoids<<<mapGrid, blockDim>>>(xBeg, dx, n, dData);
    size_t size = blockDim.x*sizeof(float);
    ShmemReduceKernel<<<gridDim, blockDim,size>>>(dReduc, dData);
    //Recall that this makes a reduced array of size grid_dim/block_dim.
    //Second Stage of First sum! 
    ShmemReduceKernel<<<1, blockDim, size>>>(dReduc, dReduc);
    hipMemcpy(&value, dReduc, sizeof(float), hipMemcpyDeviceToHost); 
    //Recall that value now = pi/2
    value *= 2.0f;
    //Free memory
    hipFree(dReduc);
    hipFree(dData);
    return value;
}

/* Driver for the computation of pi. */
int main()
{
        int n = pow(2,20);
        float pi = MmmPi(n);
        std::cout<<" Pi = "<< pi <<std::endl;
}

