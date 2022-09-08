#include "mat.h"
#include <iostream>
#include <stdlib.h>

/*
    Kernel: Adds two matrices and stores them in a third.
    A, B inputs
    C output
*/
__global__ void MatAddKernel(const GpuMatrix A, const GpuMatrix B,
                             GpuMatrix C) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x; // thread in x
  int idy = threadIdx.y + blockDim.y * blockIdx.y; // thread in y
  int tid = idx + A.width * idy;                   // Memory is 1D

  if (idx < A.width && idy < A.height) {
    C.elements[tid] = A.elements[tid] + B.elements[tid];
  }
}

/*
    Driver for launching MatAddKernel
*/
void MatAdd(const CpuMatrix &A, const CpuMatrix &B, CpuMatrix &C) {
  int w = A.width, h = A.height;

  // GPU memory allocation is handled by constructor
  GpuMatrix dA(w, h);
  hipMemcpy(dA.elements, A.elements.data(), w * h * sizeof(float),
            hipMemcpyHostToDevice);
  GpuMatrix dB(w, h);
  hipMemcpy(dB.elements, B.elements.data(), w * h * sizeof(float),
            hipMemcpyHostToDevice);
  GpuMatrix dC(w, h);

  dim3 dimBlock(16, 16);
  dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y);
  MatAddKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);
  hipMemcpy(C.elements.data(), dC.elements, w * h * sizeof(float),
            hipMemcpyDeviceToHost);
  dA.deAllocate();
  dB.deAllocate();
  dC.deAllocate();
}

// Main program
int main() {
  // Set up matrices

  int N = 4096;
  int M = 4096;
  CpuMatrix A(M, N, 1.f), B(M, N, 1.f), C(M, N);
  MatAdd(A, B, C);
  std::cout << C.elements[0] << std::endl;
}
