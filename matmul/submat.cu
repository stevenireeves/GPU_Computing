#include "submat.h"



//Get matrix element
__device__ float subMatrix::GetElem(const int row, const int col)
{
    return elements[row*stride + col];
} 

//Set a matrix element
__device__ void subMatrix::SetElem(const int row, const int col, const float value)
{
     elements[row * stride + col] = value;
}



