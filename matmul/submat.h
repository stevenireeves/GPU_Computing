#ifndef SUBMAT_H
#define SUBMAT_H
#include "mat.h"
/* This class only is available on the GPU  
   Gets the BLOCK_SIZE x BLOCK_SIZE submatrix of a matrix that is
   located col sub-matrices to the right and row sub-matrices down
   from the upper-left corner of A */

class subMatrix{
    public:
    /* Member Data */ 
    int width; 
    int height; 
    int stride;
    float* elements; 

    __device__
    subMatrix(Matrix A, int sub_size, int row, int col)
    {
        width = sub_size;
        height = sub_size;
        stride = A.stride;
        // memory at spot
        elements = &A.elements[stride * width * row + height * col];
     }

//Get matrix element
    __device__ 
    float GetElem(const int row, const int col);


//Set a matrix element
    __device__ 
    void SetElem(const int row, const int col, const float value);
};
#endif
