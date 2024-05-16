#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#define NUM_BITS 32
#define size 1024

template <class T>
__device__ T plus_scan(T *x) // Hillis and Steele
{
  __shared__ T temp[2 * size]; // allocated on invocation
  int tid = threadIdx.x;
  int pout = 0, pin = 1;
  int n = size;
  // load input into shared memory.
  temp[tid] = x[tid];
  __syncthreads();
  for (int offset = 1; offset < n; offset <<= 1) {
    pout = 1 - pout; // swap double buffer indices
    pin = 1 - pout;
    if (tid >= offset)
      temp[pout * n + tid] = temp[pin * n + tid] + temp[pin * n + tid - offset];
    else
      temp[pout * n + tid] = temp[pin * n + tid];
    __syncthreads();
  }
  x[tid] = temp[pout * n + tid]; // write output
  return x[tid];
}

__global__ void partition_by_bit(unsigned int *values, unsigned int bit) {
  unsigned int tid = threadIdx.x;
  unsigned int bsize = blockDim.x;
  unsigned int x_i = values[tid];
  __syncthreads();
  unsigned int p_i =
      (x_i >> bit) &
      0b001; // value of x_i in binary at bits place predicate step!
  values[tid] = p_i;
  __syncthreads();

  unsigned int T_before = plus_scan(values); // scatter index before trues
  __syncthreads();
  unsigned int T_t = values[bsize - 1]; // total "trues"
  unsigned int F_t = bsize - T_t;
  __syncthreads();
  if (p_i) {
    values[T_before - 1 + F_t] = x_i;
    __syncthreads();
  } else {
    values[tid - T_before] = x_i;
    __syncthreads();
  }
}

void radix_sort(unsigned int *values) {
  unsigned int *d_vals;
  unsigned int bit;
  cudaMalloc(&d_vals, size * sizeof(unsigned int));
  cudaMemcpy(d_vals, values, size * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  for (bit = 0; bit < NUM_BITS; bit++) {
    partition_by_bit<<<1, size>>>(d_vals, bit);
  }
  cudaMemcpy(values, d_vals, size * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  cudaFree(d_vals);
}

int main() {
  unsigned int *h_vals;
  h_vals = (unsigned int *)malloc(size * sizeof(unsigned int));

  std::cout << "original array" << std::endl;
  for (int i = 0; i < size; i++) {
    h_vals[i] = size - i;
    //		int bit = (h_vals[i]>>1)&0b001;
    //		std::cout<<h_vals[i]<< "   " << bit << std::endl;
  }

  radix_sort(h_vals);

  std::cout << "Sorted Array" << std::endl;
  for (int i = 0; i < size; i++) {
    std::cout << h_vals[i] << '\t';
  }

  free(h_vals);
  return 0;
}
