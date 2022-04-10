#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <hip/hip_runtime.h>

#define M 1024 

/* 
    Kernel: Computes one step of the forward time centered space algorithm.
    Input: FP32 array f_new, FP32 array f, FP32 dx, FP32 k, FP32 dt
    Output: FP32 array f_new
*/
__global__ void ftcs(float f_new[], 
                     const float f[], const float dx,
                     const float k, const float dt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if(tid > 0 && tid < M-1) // Skip boundaries! 
	{
		float temp2 =  f[tid] + k*dt/(dx*dx)*(f[tid+1] - 2*f[tid] + f[tid-1]); 
		f_new[tid] = temp2;	
	}

}

/*
    Kernel: Applies the Initial Conditions
    Inputs: FP32 array f, FP32 array x, FP32 dx
    Output: FP32 array f
*/
__global__ void initialize(float f[], float x[], const float dx)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x; 
    float xt = -1.f + tid*dx; 
	x[tid] = xt; 
	f[tid] = exp(-0.5f*xt*xt);
}

/*
    Kernel: Fills f_new with the contents of f_old
    Inputs: FP32 array f_old, FP32 array f_new
    Output: FP32 array f_new
*/
__global__ void equate(float f_old[], const float f_new[]){
    int tid = threadIdx.x + blockIdx.x*blockDim.x; 
    f_old[tid] = f_new[tid]; 
}

/*
    Kernel: Applies the outflow boundary conditions.
    Inputs: FP32 array f
    Output: FP32 array f
*/
__global__ void bc(float f[])
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x; 
	if(tid == 0) //use only one thread for 1D BC
	{
		f[0] = f[1];
		f[M-1] = f[M-2]; 
	}
}

/*
    Helper function to write array to filesystem
*/
void io_fun(std::string file, float *x, float *f)
{
	std::ofstream myfile_tsN; 
	myfile_tsN.open(file); 
	for(int i = 0; i < M; i++)
	{
		myfile_tsN << x[i] << '\t';
		myfile_tsN << f[i] << std::endl;
	}	

	myfile_tsN.close(); 
}

/*
    Driver function to simulate the heat profile in a 1-D bar. 
*/
int main()
{
    //Integration Parameters 
	float k = 1.0f; //Thermal Conductivity
    /* Numerical Mesh Configuration */ 
	float dx = 2.0f/float(M); 
	float dt = 0.5f*(dx*dx)/k; 

	float tmax = 0.5f; 
	float t = 0.0f, tio = 0.125f; 
	
	//Allocate Memory 
	size_t sz = M*sizeof(float); 
	float *f, *x;
    f = new float[M];
    x = new float[M]; 
 
	float *d_f1, *d_f2, *d_x; 
	hipMalloc(&d_f1, sz);
    hipMalloc(&d_f2, sz); 
    hipMalloc(&d_x, sz); 
	
	//Kernel parameters
	dim3 dimBlock(16,1,1); 
	dim3 dimGrid(M/dimBlock.x, 1,1); 
	
	//Apply Initial Condition You could also create a kernel for this
    initialize<<<dimGrid, dimBlock>>>(d_f1, d_x, dx);
  
	//Copy for IO operation
 	hipMemcpy(x, d_x, sz, hipMemcpyDeviceToHost); 
   
    //device x is no longer needed 
    hipFree(d_x); 
	/*====================== Perform Integration =======================*/ 

	std::string f2;
	int kk = 0;
	while(t<tmax)
	{
	//Call the stencil routine
		ftcs<<<dimGrid, dimBlock>>>(d_f2, d_f1, dx, k, dt); 
		hipDeviceSynchronize(); 
	//Call BC
		bc<<<dimGrid, dimBlock>>>(d_f2); 
		hipDeviceSynchronize();
        equate<<<dimGrid, dimBlock>>>(d_f1, d_f2); 
		if(fmod(t, tio) == 0.0f)
		{
		//IO function
			f2 = "sol" + std::to_string(kk) + ".dat"; 
			hipMemcpy(f,d_f2, sz, hipMemcpyDeviceToHost);
			io_fun(f2, x, f); 
    		kk++;
		}

		t+=dt;
	}

	if(fmod(tmax,tio) != 0.0f)
	{//IO Function 
		f2 = "final_sol.dat"; 
		hipMemcpy(f,d_f1, sz, hipMemcpyDeviceToHost);
		io_fun(f2, x, f); 
	}

	//deallocate memory 
    delete x, f; 
	hipFree(d_f1);
    hipFree(d_f2); 
}

