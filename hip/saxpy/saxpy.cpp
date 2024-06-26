#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <hip/hip_runtime.h>

/* this is the vector addition kernel. 
   :inputs: n -> Size of vector, integer
            a -> constant multiple, float
	    x -> input 'vector', constant pointer to float
	    y -> input and output 'vector', pointer to float  */
__global__ void saxpy(int n, float a, const float x[], float y[])
{
	int id = threadIdx.x + blockDim.x*blockIdx.x; /* Performing that for loop */ 
	// check to see if id is greater than size of array
	if(id < n){
		 y[id] +=  a*x[id]; // y[id] = y[id] + a*x[id]; 
	} 
}

int main()
{
	int N = 256; 
	//create pointers and device
	float *d_x, *d_y; 
	
	const float a = 2.0f;

	//allocate and initializing memory on host
	std::vector<float> x(N, 1.f);
	std::vector<float> y(N, 1.f);
/*
	float *x, *y; 
	x = new float[N]; //C++
	(*float)Malloc(x, N*sizeof(float)); //C
*/
	//allocate our memory on GPU 
	hipMalloc(&d_x, N*sizeof(float));
	hipMalloc(&d_y, N*sizeof(float));
	
	//Memory Transfer! 
	hipMemcpy(d_x, x.data(), N*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(d_y, y.data(), N*sizeof(float), hipMemcpyHostToDevice); 

	//Launch the Kernel! In this configuration there is 1 block with 256 threads
	//Use gridDim = int((N-1)/256) in general  
	saxpy<<<1, 256>>>(N, a, d_x, d_y);

	//Transfering Memory back! 
	hipMemcpy(y.data(), d_y, N*sizeof(float), hipMemcpyDeviceToHost);
	std::cout<<"First Element of z = ax + y is " << y[0]<<std::endl; 
	hipFree(d_x);
	hipFree(d_y);
	std::cout<<"Done!"<<std::endl;  
	return 0;
}
