#ifndef MAT_H
#define MAT_H
#include <hip/hip_runtime.h>
#include <vector>

class Matrix {
public:
  /* Member Data */
  int width;
  int height;
};

class CpuMatrix;
class GpuMatrix;

class GpuMatrix : public Matrix {
public:
  float *elements;

  /* Constructor */
  GpuMatrix(const int w, const int h) {
    width = w;
    height = h;
    hipMalloc(&elements, width * height * sizeof(float));
  }

  void load(const CpuMatrix oldMatrix);
  void deAllocate();
};

class CpuMatrix : public Matrix {
public:
  std::vector<float> elements;
  /* Constructor */
  CpuMatrix(const int w, const int h, const float val = 0) {
    width = w;
    height = h;
    elements.resize(w * h, val);
  }

  void load(const GpuMatrix oldMatrix);
  void deAllocate();
};

#endif
