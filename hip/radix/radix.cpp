#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_BITS 32 
#define size 1024 

/*
    device function to compute Hillis and Steele plus scan.
    Input: class T array x
    Output: class T array x
*/
template <class T>
__device__ T plus_scan(T *x) //Hillis and Steele
{
	__shared__ T temp[2*size]; // allocated on invocation
	int tid = threadIdx.x;
	int pout = 0, pin = 1;
	int n = size; 
	// load input into shared memory.
	temp[tid] = x[tid];
	__syncthreads();
	for( int offset = 1; offset < n; offset <<= 1 )
	{
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;
		if (tid >= offset)
			temp[pout*n + tid] = temp[pin*n + tid] + temp[pin*n + tid - offset];
		else
			temp[pout*n + tid] = temp[pin*n + tid];
		__syncthreads();
	}
	x[tid] = temp[pout*n+tid]; // write output
	return x[tid]; 
}

/*
    Kernel: Partitions the array by bit
    Input: unsigned int array values, unsigned int scalar bit
    Output: unsgigned int array values
*/
__global__ void partition_by_bit(unsigned int *values, unsigned int bit)
{
	unsigned int tid = threadIdx.x; 
	unsigned int bsize = blockDim.x; 
	unsigned int x_i = values[tid];
	__syncthreads();
	unsigned int p_i = (x_i >> bit) & 0b001; //value of x_i in binary at bits place predicate step! 
	values[tid] = p_i; 
	__syncthreads();

	unsigned int T_before = plus_scan(values); //scatter index before trues
	__syncthreads();
	unsigned int T_t = values[bsize - 1]; //total "trues"
	unsigned int F_t = bsize - T_t;
	__syncthreads();
	if(p_i)
	{
		values[T_before - 1 + F_t] = x_i;
		__syncthreads(); 
	}
	else
	{
		values[tid - T_before] = x_i; 
		__syncthreads();
	}
}

/*
    Driver function for the Radix Sort algorithm.
    Inputs: unsigned int array values
    Output: sorted unsigned int array values
*/
void radix_sort(unsigned int *values)
{
	unsigned int *d_vals; 
 	unsigned int bit; 
	hipMalloc(&d_vals, size*sizeof(unsigned int)); 
 	hipMemcpy(d_vals, values, size*sizeof(unsigned int), hipMemcpyHostToDevice); 
	for(bit = 0; bit < NUM_BITS; bit++)
	{
		partition_by_bit<<<1,size>>>(d_vals, bit); 
		hipDeviceSynchronize();
	}
	hipMemcpy(values, d_vals, size*sizeof(unsigned int), hipMemcpyDeviceToHost); 
	hipFree(d_vals);
}


int main()
{
	unsigned int *h_vals = new unsigned int[size];

	std::cout<<"original array"<<std::endl;
	for(int i = 0; i<size; i++)
	{
		h_vals[i] = size - i;
	}

	radix_sort(h_vals); 
	
	std::cout<<"Sorted Array"<<std::endl;
	for(int i = 0; i<size; i++)
	{
		std::cout<<h_vals[i]<<'\t';
	}

	delete h_vals;
	return 0; 
}
