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

    void load(const Matrix old_matrix, const int dir=0);

    void dealloc(int Proc=0);
};


#endif
