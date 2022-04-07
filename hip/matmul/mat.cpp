#include "mat.h"

void Matrix::load(const Matrix old_matrix, const int dir /* = 0 */){
        size_t size = width*height*sizeof(float);
        if(dir == 0){ //CPU copy
            memcpy(elements, old_matrix.elements, size); 
        }
        else if(dir == 1){ //GPU copy host to device
            hipMemcpy(elements, old_matrix.elements, size, hipMemcpyHostToDevice);  
        }
        else if(dir == 2){ //GPU copy device to host
            hipMemcpy(elements, old_matrix.elements, size, hipMemcpyDeviceToHost);  
        }
}

void Matrix::dealloc(int Proc /* = 0*/){
        if(Proc == 0)
            delete elements;
        else
            hipFree(elements);
}
