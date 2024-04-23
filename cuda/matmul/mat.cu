#include "mat.h"

void GpuMatrix::load(const CpuMatrix oldMatrix) {
  size_t size = width * height * sizeof(float);
  cudaMemcpy(elements, oldMatrix.elements.data(), size, cudaMemcpyHostToDevice);
}

void GpuMatrix::deAllocate() { cudaFree(elements); }

void CpuMatrix::load(const GpuMatrix oldMatrix) {
  size_t size = width * height * sizeof(float);
  cudaMemcpy(elements.data(), oldMatrix.elements, size, cudaMemcpyDeviceToHost);
}

void CpuMatrix::deAllocate() { elements.clear(); }
