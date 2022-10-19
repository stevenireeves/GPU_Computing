#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>

#define NUM_BITS 32 
#define size 1024 

/*
    device function to compute Hillis and Steele plus scan.
    Input: class T array x
    Output: class T array x
*/
template <class T>
__device__ T PlusScan(T *x) //Hillis and Steele
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
__global__ void PartitionByBit(unsigned int *values, unsigned int bit)
{
	unsigned int tid = threadIdx.x; 
	unsigned int bSize = blockDim.x; 
	unsigned int xI = values[tid];
	__syncthreads();
	unsigned int pI = (xI >> bit) & 0b001; //value of x_i in binary at bits place predicate step! 
	values[tid] = pI; 
	__syncthreads();

	unsigned int tBefore = PlusScan(values); //scatter index before trues
	__syncthreads();
	unsigned int tT = values[bSize - 1]; //total "trues"
	unsigned int fT = bSize - tT;
	__syncthreads();
	if(pI)
	{
		values[tBefore - 1 + fT] = xI;
		__syncthreads(); 
	}
	else
	{
		values[tid - tBefore] = xI; 
		__syncthreads();
	}
}

/*
    Driver function for the Radix Sort algorithm.
    Inputs: unsigned int array values
    Output: sorted unsigned int array values
*/
void RadixSort(std::vector<unsigned int> &values)
{
	unsigned int *dVals; 
 	unsigned int bit; 
	hipMalloc(&dVals, size*sizeof(unsigned int)); 
 	hipMemcpy(dVals, values.data(), size*sizeof(unsigned int), hipMemcpyHostToDevice); 
	for(bit = 0; bit < NUM_BITS; bit++)
	{
		PartitionByBit<<<1,size>>>(dVals, bit); 
	}
	hipMemcpy(values.data(), dVals, size*sizeof(unsigned int), hipMemcpyDeviceToHost); 
	hipFree(dVals);
}


int main()
{
    std::vector<unsigned int> hVals(size);

	std::cout<<"original array"<<std::endl;
	for(int i = 0; i<size; i++)
	{
		hVals[i] = size - i;
	}

	RadixSort(hVals); 
	
	std::cout<<"Sorted Array"<<std::endl;
	for(int i = 0; i<size; i++)
	{
		std::cout<<hVals[i]<<'\t';
	}

	return 0; 
}
