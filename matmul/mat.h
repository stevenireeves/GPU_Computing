#ifndef MAT_H
#define MAT_H

/* This is a header file for matmul, since it is compiled with nvcc, it has the CUDA extensions for C++ */ 

class Matrix
{
public:

/* Member Data */ 
    int width; 
    int height; 
    int stride;
    int my_type; 
    float* elements; 

/* Constructor  we want this class to be able to be generated both on CPU and GPU*/ 
    __host__
    Matrix(const int w, const int h, const int s = 0, const int type = 0){
        width = w; 
        height = h;
        stride = (s==0)?w:s; 
        my_type = type; //Matrix knows if it's CPU or GPU 
        if(type == 0)
            elements = new float[width*height];
        else if(type == 1)
            cudaMalloc(&elements, width*height*sizeof(float)); 
    }

/* member functions */ 

    void load(const Matrix old_matrix, const int dir = 0){
        size_t size = width*height*sizeof(float);
        if(dir == 0){ //CPU copy
            memcpy(elements, old_matrix.elements, size); 
        }
        else if(dir == 1){ //GPU copy host to device
            cudaMemcpy(elements, old_matrix.elements, size, cudaMemcpyHostToDevice);  
        }
        else if(dir == 2){ //GPU copy device to host
            cudaMemcpy(elements, old_matrix.elements, size, cudaMemcpyDeviceToHost);  
        }
    }

    void dealloc(int Proc = 0){
        if(Proc == 0)
            delete elements;
        else
            cudaFree(elements); 
    }
};


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
    inline float GetElem(const int row, const int col)
	{
		return elements[row*stride + col];
	}

//Set a matrix element
    __device__ 
    inline void SetElem(const int row, const int col, const float value)
	{
		 elements[row * stride + col] = value; 
	}
};
#endif
