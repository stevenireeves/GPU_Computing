#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <hip/hip_runtime.h>

/* Device function to apply root(1-x^2), accepts float x, and returns float */
__device__ float myfun(float x)
{
	return sqrtf(1.0f - x*x); 
}

/*  Applies myfun to to data generated from xbeg to xbeg + n*dx 
    computes Reimann Rectangles for numerical integral
    and inputs into float pointer f1.
    Inputs: xbeg float, dx float, n int, float array f1.
    Outputs: float array f1.
 */
__global__ void map(float xbeg, float dx, int n, float *f1)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x; 
	if(tid >= n) return; 

	float2 x; 
	x.x = xbeg + tid*dx; 
	x.y = xbeg + (tid + 1)*dx; 
	f1[tid] = myfun(x.x)*dx*0.5f + myfun(x.y)*dx*0.5f;
}

/*  Computes the reduction of FP32 array d_in. 
    Ouput is FP32 array d_out. If input array is larger than 1024 floats
    A partial reduction is computed to the 1024th entries of d_out. 
    If less, the full reduction can be found in d_out[0].
    In this application d_in contains the Reimainn rectangles 
    to compute the numerical integral.  
*/
__global__ void shmem_reduce_kernel(float * d_out, const float *d_in)
{
    // sdata is allocated in the kernel call: via dynamic shared memeory
    extern __shared__ float sdata[];
    
    int myId = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x;
    
    //load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads(); // always sync before using sdata
    
    //do reduction over shared memory
    for(int s = blockDim.x/2; s>0; s >>=1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); //make sure all additions are finished
    }
    
    //only tid 0 writes out result!
    if(tid == 0)
    {
            d_out[blockIdx.x] = sdata[0];
    }
}

/* Host code to compute the digits of pi.
   Inputs: n int, the resolution of the integral to calculate pi
   Outputs: pi float, the estimate of pi.
*/ 
float mmmmmm_pi(int n)
{
    //Initialization
    float value; 
    float *d_data;
    float *d_reduc; 
    size_t original = n*sizeof(float);
    size_t reduc = n/(1024)*sizeof(float);
    
    //Allocation    
    hipMalloc(&d_data, original);
    hipMalloc(&d_reduc, reduc);
    
    //Kernel Parameters
    dim3 block_dim(1024, 1, 1);
    dim3 grid_dim(n/block_dim.x, 1, 1);
    
    //integration parameters
    float xbeg = -1.0f;
    float dx = (1.0f - xbeg)/(float)n;

    dim3 map_grid(n/block_dim.x + 1, 1,1); 
    //map+stencil kernel Note because of the shift we need more threads.
    map<<<map_grid, block_dim>>>(xbeg, dx, n, d_data);
    size_t size = block_dim.x*sizeof(float);
    shmem_reduce_kernel<<<grid_dim, block_dim,size>>>(d_reduc, d_data);
    //Recall that this makes a reduced array of size grid_dim/block_dim.
    //Second Stage of First sum! 
    shmem_reduce_kernel<<<1, block_dim, size>>>(d_reduc, d_reduc);
    hipMemcpy(&value, d_reduc, sizeof(float), hipMemcpyDeviceToHost); 
    //Recall that value now = pi/2
    value *= 2.0f;
    //Free memory
    hipFree(d_reduc);
    hipFree(d_data);
    return value;
}

/* Driver for the computation of pi. */
int main()
{
        int n = pow(2,20);
        float pi = mmmmmm_pi(n);
        std::cout<<" Pi = "<< pi <<std::endl;
}

