#include <iostream>
#include <stdlib.h>
#include <vector>

/* two array reversal functions to illustrate the use of shared memory */

/* staticReverse uses statically allocated shared memory
   :inputs: array d, length n
   :outputs: reversed array d */
__global__ void staticReverse(int *d, int n) {
  __shared__ int s[64]; // static shared memory allocation
  int t = threadIdx.x;
  int tr = n - t - 1;
  if (t < n) {
    s[t] = d[t];
    __syncthreads(); // None shall pass
    d[t] = s[tr];
  }
}

/* dynamicReverse uses dynamically allocated shared memory
   :inputs: array d, length n
   :outputs: reversed array d */
__global__ void dynamicReverse(int *d, int n) {
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n - t - 1;
  if (t < n) {
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
  }
}

int main() {
  const int n = 64;
  std::vector<int> a(n, 0);
  int *d_a;
  cudaMalloc(&d_a, n * sizeof(int));

  for (int i = 0; i < n; i++)
    a[i] = i;

  cudaMemcpy(d_a, a.data(), n * sizeof(int),
             cudaMemcpyHostToDevice); // Transfer to device

  dynamicReverse<<<1, n, n * sizeof(int)>>>(d_a, n); // grid ,block ,shared

  cudaMemcpy(a.data(), d_a, n * sizeof(int),
             cudaMemcpyDeviceToHost); // bring it back

  std::cout << a[0] << std::endl;
  cudaFree(d_a);
  return 0;
}
