#ifndef SUBMAT_H
#define SUBMAT_H
#include "mat.h"
/* This class only is available on the GPU
   Gets the BLOCK_SIZE x BLOCK_SIZE submatrix of a matrix that is
   located col sub-matrices to the right and row sub-matrices down
   from the upper-left corner of A */

class subMatrix : public Matrix {
public:
  /* Member Data */
  int stride;
  float *elements;

  __device__ subMatrix(GpuMatrix A, int sub_size, int row, int col) {
    width = sub_size;
    height = sub_size;
    stride = A.width;
    // memory at spot
    elements = &A.elements[stride * width * row + height * col];
  }
  // Get matrix element
  __device__ inline float GetElem(const int row, const int col) {
    return elements[row * stride + col];
  }

  // Set a matrix element
  __device__ inline void SetElem(const int row, const int col,
                                 const float value) {
    elements[row * stride + col] = value;
  }
};
#endif
