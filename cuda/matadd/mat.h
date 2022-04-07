#ifndef MAT_H
#define MAT_H

class Matrix
{
public:

/* Member Data */ 
    int width; 
    int height; 
    int my_type; 
    float* elements; 

/* Constructors */ 
    Matrix(const int w, const int h, const int type = 0){
        width = w; 
        height = h;
        my_type = type; //Matrix knows if it's CPU or GPU 
        if(type == 0)
            elements = new float[width*height];
        else
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
    }

    void cpu_deallocate(){
        delete elements; 
    }

    void gpu_deallocate(){
        cudaFree(elements); //Do not use cudaFree 
    }
};
#endif
