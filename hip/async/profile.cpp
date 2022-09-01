#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <vector>

int main() {
  const unsigned int N = 1048576;
  const unsigned int bytes = N * sizeof(int);
  std::vector<int> hA(N, 0);
  int *dA;
  hipMalloc(&dA, bytes);

  hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice);
  hipMemcpy(hA.data(), dA, bytes, hipMemcpyDeviceToHost);

  hipFree(dA);
  return 0;
}
