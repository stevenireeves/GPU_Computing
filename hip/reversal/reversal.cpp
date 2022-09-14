#include <hip/hip_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

#define PROB_SIZE 64
/* two array reversal functions to illustrate the use of shared memory */

/* staticReverse uses statically allocated shared memory
   :inputs: array d, length n
   :outputs: reversed array d */
__global__ void StaticReverse(int d[], int n) {
  __shared__ int s[PROB_SIZE]; // static shared memory allocation
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
__global__ void DynamicReverse(int d[], int n) {
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
  const int n = PROB_SIZE;
  std::vector<int> a(n);
  for (int i = 0; i < n; i++) {
    a[i] = i;
  }

  int *dA;
  hipMalloc(&dA, n * sizeof(int));
  hipMemcpy(dA, a.data(), n * sizeof(int),
            hipMemcpyHostToDevice); // Transfer to device

  StaticReverse<<<1, n>>>(dA, n); // grid ,block
  hipMemcpy(a.data(), dA, n * sizeof(int),
            hipMemcpyDeviceToHost); // bring it back

  std::cout << "First Element of a after static reverse" << '\t' << a[0]
            << std::endl;

  DynamicReverse<<<1, n, n * sizeof(int)>>>(dA, n); // grid ,block ,shared
  hipMemcpy(a.data(), dA, n * sizeof(int),
            hipMemcpyDeviceToHost); // bring it back

  std::cout << "First Element of a after dynamic reverse" << '\t' << a[0]
            << std::endl;
  hipFree(dA);
  return 0;
}
