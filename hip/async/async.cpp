/* Adapted from CUDA
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
// Convenience function for checking HIP runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline hipError_t checkHip(hipError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != hipSuccess) {
    fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
    assert(result == hipSuccess);
  }
#endif
  return result;
}

__global__ void Kernel(float *a, int offset) {
  int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
  float x = (float)i;
  float s = sinf(x);
  float c = cosf(x);
  a[i] = a[i] + sqrtf(s * s + c * c);
}

float MaxError(float *a, int n) {
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i] - 1.0f);
    if (error > maxE)
      maxE = error;
  }
  return maxE;
}

int main(int argc, char **argv) {
  const int blockSize = 256, nStreams = 4;
  const int n = 4 * 1024 * blockSize * nStreams;
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);

  int devId = 0;
  if (argc > 1)
    devId = atoi(argv[1]);

  hipDeviceProp_t prop;
  checkHip(hipGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkHip(hipSetDevice(devId));

  // allocate pinned host memory and device memory
  float *a, *d_a;
  checkHip(hipMallocHost((void **)&a, bytes)); // host pinned
  checkHip(hipMalloc((void **)&d_a, bytes));   // device

  float ms; // elapsed time in milliseconds

  // create events and streams
  hipEvent_t startEvent, stopEvent, dummyEvent;
  hipStream_t stream[nStreams];
  checkHip(hipEventCreate(&startEvent));
  checkHip(hipEventCreate(&stopEvent));
  checkHip(hipEventCreate(&dummyEvent));
  for (int i = 0; i < nStreams; ++i)
    checkHip(hipStreamCreate(&stream[i]));

  // baseline case - sequential transfer and execute
  memset(a, 0, bytes);
  checkHip(hipEventRecord(startEvent, 0));
  checkHip(hipMemcpy(d_a, a, bytes, hipMemcpyHostToDevice));
  Kernel<<<n / blockSize, blockSize>>>(d_a, 0);
  checkHip(hipMemcpy(a, d_a, bytes, hipMemcpyDeviceToHost));
  checkHip(hipEventRecord(stopEvent, 0));
  checkHip(hipEventSynchronize(stopEvent));
  checkHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for sequential transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", MaxError(a, n));

  // asynchronous version 1: loop over {copy, kernel, copy}
  memset(a, 0, bytes);
  checkHip(hipEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkHip(hipMemcpyAsync(&d_a[offset], &a[offset], streamBytes,
                            hipMemcpyHostToDevice, stream[i]));
    Kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    checkHip(hipMemcpyAsync(&a[offset], &d_a[offset], streamBytes,
                            hipMemcpyDeviceToHost, stream[i]));
  }
  checkHip(hipEventRecord(stopEvent, 0));
  checkHip(hipEventSynchronize(stopEvent));
  checkHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", MaxError(a, n));

  // asynchronous version 2:
  // loop over copy, loop over kernel, loop over copy
  memset(a, 0, bytes);
  checkHip(hipEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkHip(hipMemcpyAsync(&d_a[offset], &a[offset], streamBytes,
                            hipMemcpyHostToDevice, stream[i]));
  }
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    Kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  }
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkHip(hipMemcpyAsync(&a[offset], &d_a[offset], streamBytes,
                            hipMemcpyDeviceToHost, stream[i]));
  }
  checkHip(hipEventRecord(stopEvent, 0));
  checkHip(hipEventSynchronize(stopEvent));
  checkHip(hipEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", MaxError(a, n));

  // cleanup
  checkHip(hipEventDestroy(startEvent));
  checkHip(hipEventDestroy(stopEvent));
  checkHip(hipEventDestroy(dummyEvent));
  for (int i = 0; i < nStreams; ++i)
    checkHip(hipStreamDestroy(stream[i]));
  hipFree(d_a);
  hipFreeHost(a);

  return 0;
}
