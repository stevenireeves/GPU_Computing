/* Adapted from CUDA */

#include <stdio.h>
#include <assert.h>
#include <hip/hip_runtime.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
hipError_t checkHip(hipError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != hipSuccess) {
    fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
    assert(result == hipSuccess);
  }
#endif
  return result;
}

template <typename T>
__global__ void offset(T* a, int s)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x + s;
  a[i] = a[i] + 1;
}

template <typename T>
__global__ void stride(T* a, int s)
{
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * s;
  a[i] = a[i] + 1;
}

template <typename T>
void runTest(int deviceId, int nMB)
{
  int blockSize = 256;
  float ms;

  T *d_a;
  hipEvent_t startEvent, stopEvent;
    
  int n = nMB*1024*1024/sizeof(T);

  // NB:  d_a(33*nMB) for stride case
  checkHip( hipMalloc(&d_a, n * 33 * sizeof(T)) );

  checkHip( hipEventCreate(&startEvent) );
  checkHip( hipEventCreate(&stopEvent) );

  printf("Offset, Bandwidth (GB/s):\n");
  
  offset<<<n/blockSize, blockSize>>>(d_a, 0); // warm up

  for (int i = 0; i <= 32; i++) {
    checkHip( hipMemset(d_a, 0, n * sizeof(T)) );

    checkHip( hipEventRecord(startEvent,0) );
    offset<<<n/blockSize, blockSize>>>(d_a, i);
    checkHip( hipEventRecord(stopEvent,0) );
    checkHip( hipEventSynchronize(stopEvent) );

    checkHip( hipEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  printf("\n");
  printf("Stride, Bandwidth (GB/s):\n");

  stride<<<n/blockSize, blockSize>>>(d_a, 1); // warm up
  for (int i = 1; i <= 32; i++) {
    checkHip( hipMemset(d_a, 0, n * sizeof(T)) );

    checkHip( hipEventRecord(startEvent,0) );
    stride<<<n/blockSize, blockSize>>>(d_a, i);
    checkHip( hipEventRecord(stopEvent,0) );
    checkHip( hipEventSynchronize(stopEvent) );

    checkHip( hipEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  checkHip( hipEventDestroy(startEvent) );
  checkHip( hipEventDestroy(stopEvent) );
  hipFree(d_a);
}

int main(int argc, char **argv)
{
  int nMB = 4;
  int deviceId = 0;
  bool bFp64 = false;

  for (int i = 1; i < argc; i++) {    
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi((char*)(&argv[i][4]));
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }
  
  hipDeviceProp_t prop;
  
  checkHip( hipSetDevice(deviceId) )
  ;
  checkHip( hipGetDeviceProperties(&prop, deviceId) );
  printf("Device: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", nMB);
  
  printf("%s Precision\n", bFp64 ? "Double" : "Single");
  
  if (bFp64) runTest<double>(deviceId, nMB);
  else       runTest<float>(deviceId, nMB);
}

