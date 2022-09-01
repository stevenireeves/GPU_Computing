/* Adapted from CUDA */

#include <assert.h>
#include <hip/hip_runtime.h>
#include <stdio.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline hipError_t CheckHip(hipError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != hipSuccess) {
    fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
    assert(result == hipSuccess);
  }
#endif
  return result;
}

template <typename T> __global__ void Offset(T *a, int s) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + s;
  a[i] = a[i] + 1;
}

template <typename T> __global__ void Stride(T *a, int s) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * s;
  a[i] = a[i] + 1;
}

template <typename T> void RunTest(int deviceId, int nMB) {
  int blockSize = 256;
  float ms;

  T *d_a;
  hipEvent_t startEvent, stopEvent;

  int n = nMB * 1024 * 1024 / sizeof(T);

  // NB:  d_a(33*nMB) for Stride case
  CheckHip(hipMalloc(&d_a, n * 33 * sizeof(T)));

  CheckHip(hipEventCreate(&startEvent));
  CheckHip(hipEventCreate(&stopEvent));

  printf("Offset, Bandwidth (GB/s):\n");

  Offset<<<n / blockSize, blockSize>>>(d_a, 0); // warm up

  for (int i = 0; i <= 32; i++) {
    CheckHip(hipMemset(d_a, 0, n * sizeof(T)));

    CheckHip(hipEventRecord(startEvent, 0));
    Offset<<<n / blockSize, blockSize>>>(d_a, i);
    CheckHip(hipEventRecord(stopEvent, 0));
    CheckHip(hipEventSynchronize(stopEvent));

    CheckHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
    printf("%d, %f\n", i, 2 * nMB / ms);
  }

  printf("\n");
  printf("Stride, Bandwidth (GB/s):\n");

  Stride<<<n / blockSize, blockSize>>>(d_a, 1); // warm up
  for (int i = 1; i <= 32; i++) {
    CheckHip(hipMemset(d_a, 0, n * sizeof(T)));

    CheckHip(hipEventRecord(startEvent, 0));
    Stride<<<n / blockSize, blockSize>>>(d_a, i);
    CheckHip(hipEventRecord(stopEvent, 0));
    CheckHip(hipEventSynchronize(stopEvent));

    CheckHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
    printf("%d, %f\n", i, 2 * nMB / ms);
  }

  CheckHip(hipEventDestroy(startEvent));
  CheckHip(hipEventDestroy(stopEvent));
  hipFree(d_a);
}

int main(int argc, char **argv) {
  int nMB = 4;
  int deviceId = 0;
  bool bFp64 = false;

  for (int i = 1; i < argc; i++) {
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi((char *)(&argv[i][4]));
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }

  hipDeviceProp_t prop;

  CheckHip(hipSetDevice(deviceId));
  CheckHip(hipGetDeviceProperties(&prop, deviceId));
  printf("Device: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", nMB);

  printf("%s Precision\n", bFp64 ? "Double" : "Single");

  if (bFp64)
    RunTest<double>(deviceId, nMB);
  else
    RunTest<float>(deviceId, nMB);
}
