#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#define blockSize 1024

// interleaved addressing
template <class T> __global__ void reduce0(T *d_out, const T *d_in) {
  // sdata is allocated in the kernel call: via dynamic shared memeory
  extern __shared__ T sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = d_in[myId];
  __syncthreads(); // always sync before using sdata

  // do reduction over shared memory
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads(); // make sure all additions are finished
  }

  // only tid 0 writes out result!
  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}

// interleaved addressing with no thread divergence but shared memory bank
// conflicts
template <class T> __global__ void reduce1(T *d_out, const T *d_in) {
  // sdata is allocated in the kernel call: via dynamic shared memeory
  extern __shared__ T sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = d_in[myId];
  __syncthreads(); // always sync before using sdata

  // do reduction over shared memory
  for (int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid; // Strided indexing!
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads(); // make sure all additions are finished
  }

  // only tid 0 writes out result!
  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}

// sequential addressing
template <class T> __global__ void reduce2(T *d_out, const T *d_in) {
  // sdata is allocated in the kernel call: via dynamic shared memeory
  extern __shared__ T sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = d_in[myId];
  __syncthreads(); // always sync before using sdata

  // do reduction over shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads(); // make sure all additions are finished
  }

  // only tid 0 writes out result!
  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}

// First add during global load --  half the number of blocks
template <class T> __global__ void reduce3(T *d_out, const T *d_in) {
  // sdata is allocated in the kernel call: via dynamic shared memeory
  extern __shared__ T sdata[];

  int myId = threadIdx.x + (blockDim.x * 2) * blockIdx.x;
  int tid = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = d_in[myId] + d_in[myId + blockDim.x];
  __syncthreads(); // always sync before using sdata

  // do reduction over shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads(); // make sure all additions are finished
  }

  // only tid 0 writes out result!
  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}

// Unroll the last warp
template <class T> __global__ void reduce4(T *d_out, const T *d_in) {
  // sdata is allocated in the kernel call: via dynamic shared memeory
  extern __shared__ T sdata[];

  int myId = threadIdx.x + (blockDim.x * 2) * blockIdx.x;
  int tid = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = d_in[myId] + d_in[myId + blockDim.x];
  __syncthreads(); // always sync before using sdata

  // do reduction over shared memory
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads(); // make sure all additions are finished
  }
  if (tid < 32) {

    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
  }

  // only tid 0 writes out result!
  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}

// Unroll the whole loop
template <class T> __global__ void reduce5(T *d_out, const T *d_in) {
  // sdata is allocated in the kernel call: via dynamic shared memeory
  extern __shared__ T sdata[];

  int myId = threadIdx.x + (blockDim.x * 2) * blockIdx.x;
  int tid = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = d_in[myId] + d_in[myId + blockDim.x];
  __syncthreads(); // always sync before using sdata

  // do reduction over shared memory

  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
  }

  // only tid 0 writes out result!
  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}

// multiple elements per thread
template <class T>
__global__ void reduce6(T *d_out, const T *d_in, unsigned int n) {
  // sdata is allocated in the kernel call: via dynamic shared memeory
  extern __shared__ T sdata[];

  int myId = threadIdx.x + (blockDim.x * 2) * blockIdx.x;
  int tid = threadIdx.x;
  int gridSize = blockDim.x * 2 * gridDim.x;
  sdata[tid] = 0;

  // load shared mem from global mem
  while (myId < n) {
    sdata[tid] += d_in[myId] + d_in[myId + blockDim.x];
    myId += gridSize;
  }
  __syncthreads();

  // do reduction over shared memory

  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) {
    if (blockSize >= 64)
      sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
      sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
      sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
      sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
      sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
      sdata[tid] += sdata[tid + 1];
  }

  // only tid 0 writes out result!
  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}

int main() {
  constexpr int n = 2 << 20; // pow(2,20);
  std::cout << " n = " << n << std::endl;
  int *array, *reduced;
  cudaMalloc((int **)&array, n * sizeof(int));
  cudaMalloc((int **)&reduced, 1024 * sizeof(int));
  cudaMemset(array, 1, n * sizeof(int));
  dim3 dimBlock(1024);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  float red_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  reduce0<<<dimGrid, dimBlock, 1024 * sizeof(int)>>>(reduced, array);
  reduce0<<<1, dimBlock, 1024 * sizeof(int)>>>(reduced, reduced);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&red_time, start, stop);
  std::cout << "Reduce 0 Elapsed Time = " << red_time << " ms" << std::endl;
  std::cout << " Bandwidth in GB/s =" << (n + 1024 + 1025) * 4 / red_time / 1e6
            << std::endl;

  cudaEventRecord(start);
  reduce1<<<dimGrid, dimBlock, 1024 * sizeof(int)>>>(reduced, array);
  reduce1<<<1, dimBlock, 1024 * sizeof(int)>>>(reduced, reduced);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&red_time, start, stop);
  std::cout << "Reduce 1 Elapsed Time = " << red_time << " ms" << std::endl;
  std::cout << " Bandwidth in GB/s =" << (n + 1024 + 1025) * 4 / red_time / 1e6
            << std::endl;

  cudaEventRecord(start);
  reduce2<<<dimGrid, dimBlock, 1024 * sizeof(int)>>>(reduced, array);
  reduce2<<<1, dimBlock, 1024 * sizeof(int)>>>(reduced, reduced);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&red_time, start, stop);
  std::cout << "Reduce 2 Elapsed Time = " << red_time << " ms" << std::endl;
  std::cout << " Bandwidth in GB/s =" << (n + 1024 + 1025) * 4 / red_time / 1e6
            << std::endl;

  dim3 dimGrid3((n + dimBlock.x - 1) / (2 * dimBlock.x));
  cudaEventRecord(start);
  reduce3<<<dimGrid3, dimBlock, 1024 * sizeof(int)>>>(reduced, array);
  reduce3<<<1, dimBlock, 1024 * sizeof(int)>>>(reduced, reduced);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&red_time, start, stop);
  std::cout << "Reduce 3 Elapsed Time = " << red_time << " ms" << std::endl;
  std::cout << " Bandwidth in GB/s =" << (n + 1024 + 1025) * 4 / red_time / 1e6
            << std::endl;

  cudaEventRecord(start);
  reduce4<<<dimGrid3, dimBlock, 1024 * sizeof(int)>>>(reduced, array);
  reduce4<<<1, dimBlock, 1024 * sizeof(int)>>>(reduced, reduced);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&red_time, start, stop);
  std::cout << "Reduce 4 Elapsed Time = " << red_time << " ms" << std::endl;
  std::cout << " Bandwidth in GB/s =" << (n + 1024 + 1025) * 4 / red_time / 1e6
            << std::endl;

  cudaEventRecord(start);
  reduce5<<<dimGrid3, dimBlock, 1024 * sizeof(int)>>>(reduced, array);
  reduce5<<<1, dimBlock, 1024 * sizeof(int)>>>(reduced, reduced);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&red_time, start, stop);
  std::cout << "Reduce 5 Elapsed Time = " << red_time << " ms" << std::endl;
  std::cout << " Bandwidth in GB/s =" << (n + 1024 + 1025) * 4 / red_time / 1e6
            << std::endl;

  dim3 dimGrid6(dimGrid.x / (64));
  cudaEventRecord(start);
  reduce6<<<dimGrid6, dimBlock, 1024 * sizeof(int)>>>(reduced, array, 32);
  reduce6<<<1, dimBlock, 1024 * sizeof(int)>>>(reduced, reduced, 1);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&red_time, start, stop);
  std::cout << "Reduce 6 Elapsed Time = " << red_time << " ms" << std::endl;
  std::cout << " Bandwidth in GB/s =" << (n + 1024 + 1025) * 4 / red_time / 1e6
            << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(array);
  cudaFree(reduced);
}
