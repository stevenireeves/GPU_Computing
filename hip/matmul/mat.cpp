#include "mat.h"

void GpuMatrix::load(const CpuMatrix oldMatrix) {
  size_t size = width * height * sizeof(float);
  hipMemcpy(elements, oldMatrix.elements.data(), size, hipMemcpyHostToDevice);
}

void GpuMatrix::deAllocate() { hipFree(elements); }

void CpuMatrix::load(const GpuMatrix oldMatrix) {
  size_t size = width * height * sizeof(float);
  hipMemcpy(elements.data(), oldMatrix.elements, size, hipMemcpyDeviceToHost);
}

void CpuMatrix::deAllocate() { elements.clear(); }
