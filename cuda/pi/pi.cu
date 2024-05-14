#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>

__device__ float myfun(float x)
{
	return sqrtf(1.0f - x*x); 
}

__global__ void map(float xbeg, float dx, int n, float *f1)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x; 
	if(tid < 1 && tid > n)
		return; 

	float2 x; 
	x.x = xbeg + tid*dx; 
	x.y = xbeg + (tid - 1)*dx; 
	float ftemp = myfun(x.x)*dx*0.5f + myfun(x.y)*dx*0.5f;
	f1[tid-1] = ftemp;  
}

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


float mmmmmm_pi(int n)
{

        //Initialization

        float value; 
        float *d_data;
        float *d_reduc, *h_reduc; 
        size_t original = n*sizeof(float);
        size_t reduc = n/(1024)*sizeof(float);

        //Allocation    
        h_reduc = (float*)malloc(reduc);
        cudaMalloc((void**)&d_data, original);
        cudaMalloc((void**)&d_reduc, reduc);

        //Kernel Parameters
        dim3 block_dim(1024,1,1);
        dim3 grid_dim(n/block_dim.x,1,1);

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
	cudaMemcpy(h_reduc,d_reduc, reduc, cudaMemcpyDeviceToHost); 
        value = h_reduc[0];
        //Recall that value now = pi/2
        value *= 2.0f;
        //Free memory
        free(h_reduc);
        cudaFree(d_reduc);
        cudaFree(d_data);

        return value;
}



int main()
{
        int n = pow(2,20);
        float pi = mmmmmm_pi(n);
        std::cout<<" Pi = "<< pi <<std::endl;
}

