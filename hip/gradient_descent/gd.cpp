#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include "mat.h"
#define N 1024 
#define BLOCK_SIZE 16 

/*
    Helper function to load data from filesystme into Matrix A.
*/
void load_Matrix(std::string file, Matrix A) {
    std::ifstream f;
    f.open(file);
    for( int i = 0; i <A.height; i++)
        for(int j = 0; j < A.width; j++)
        {
           f >> A.elements[j + A.width*i];
        }
    f.close();
}

/* 
    Kernel: Shared Memory Sum Reduction
    Input: class T array d_out, class T array d_in 
    Output: class T array d_out
*/
template <class T>
__global__ void reduce_sum(T * d_out, const T *d_in) {
    // sdata is allocated in the kernel call: via dynamic shared memeory
    extern __shared__ T sdata[];

    int myId = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x;

    //load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads(); // always sync before using sdata

    //do reduction over shared memory
    for(int s = blockDim.x/2; s>0; s >>=1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); //make sure all additions are finished
    }

    //only tid 0 writes out result!
    if(tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

/*
    Kernel: Pairwise multiplication
    Inputs: class T array x, class T array y, Class T array z
    Output: class T array z
*/
template <class T>
__global__ void pairwise_mult(const T *x, const T *y, T *z) {
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	z[gid] = x[gid]*y[gid]; 
}

/*
    Kernel: Templated AX + Y 
    Input: class T array x, class T array y, class T array a
    Ouput: class T array x
*/
template <class T> 
__global__ void saxpy(T *x, T *y, T *a) {
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	x[gid] = x[gid] + a[0]*y[gid]; 
}

// Pass by value
/*
    Kernel: Templated AX + Y 
    Input: class T array x, class T out, class T array y, class T scalar a
    Ouput: class T array out
*/
template <class T>
__global__ void saxpy2(T *out, const T *x, const T *y, T a) {
	int gid = threadIdx.x + blockDim.x*blockIdx.x; 
	out[gid] = x[gid] + a*y[gid]; 
}

/*
    Kernel: Matrix Vector Multiplication
    Input: Matrix A, FP32 array x, FP32 array y
    Output: FP32 array y
*/
__global__ void
mat_vec (Matrix A, float *x, float *y) {
    int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    if (tidx > A.width)
        return; // thread outside bounds.
	float yval = 0.0f; 
    for (int e = 0; e < A.width ; e++) {
        yval +=  A.elements[A.width * tidx + e]* x[e];
    }
    y[tidx] = yval;
}

/*
    Kernel: Applies absolute value to in data
    Input: FP32 array xout, FP32 array xin
    Output: FP32 array xout
*/
__global__ void vec_abs(float *xout, float *xin) {
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	xout[gid] = fabs(xin[gid]); 
}

/*
    Kernel: Divides two inputs elementwise
    Input: class T array xout, class T array num, class T denom
    Output: class T array xout
*/
template <class T>
__global__ void divide(T *xout, T *num, T* denom) {
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	xout[gid] = num[gid]/denom[gid]; 
}

/*
    Driver function to compute Gradient Descent.
    Input: Matrix A, FP32 array x, FP32 array b, FP32 scalar eps
    Output: FP32 array x
*/
void grad_descnt(Matrix A, float *x, float *b, float eps) {
    float *r, *d_b, *d_x, *temp1, *temp2;
    float *alpha, res;
    int counter = 0;
    Matrix d_A(A.width, A.height, 1);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(A.width/dimBlock.x);

    hipMalloc(&r, A.width*sizeof(float));
    hipMalloc(&d_b, A.width*sizeof(float));
    hipMalloc(&d_x, A.width*sizeof(float));
    hipMalloc(&alpha, sizeof(float));
	res = 1.0f; 
    hipMalloc(&temp1, A.width*sizeof(float));
    hipMalloc(&temp2, A.width*sizeof(float));

    hipMemcpy(d_b, b, A.width*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_x, x, A.width*sizeof(float), hipMemcpyHostToDevice);
    d_A.load(A, 1); 

//Compute R
    mat_vec<<<dimGrid, dimBlock>>>(d_A, d_x, temp1);
    hipDeviceSynchronize();
    saxpy2<<<dimGrid, dimBlock>>>(r, temp1, d_b, -1.0f);
    hipDeviceSynchronize(); 

    while(res > eps) {
//  Calculate Alpha ----------------------------------------------------------------
//  compute p^Tr
        pairwise_mult<<<dimGrid, dimBlock>>>(r, r, temp1);
        reduce_sum<<<1,N,N*sizeof(float)>>>(temp1, temp1);
		//compute p^TAp
        mat_vec<<<dimGrid, dimBlock>>>(d_A,r,temp2);
        pairwise_mult<<<dimGrid, dimBlock>>>(r, temp2, temp2);
        reduce_sum<<<1, N, N*sizeof(float)>>>(temp2, temp2);
        hipDeviceSynchronize();
		// alpha = r^Tr/p^TAp
        divide<<<1,1>>>(alpha,temp1,temp2);

//      X^(k+1) ---------------------------------------------------------------------------
//      X = X + alpha P
        saxpy<<<dimGrid, dimBlock>>>(d_x, r, alpha);
//      r = grad(f) -----------------------------------------------------------------------
//      Calculate new r
        mat_vec<<<dimGrid, dimBlock>>>(d_A, d_x, temp1);
        saxpy2<<<dimGrid, dimBlock>>>(r, d_b, temp1, -1.0f);

//      ||r|| ----------------------------------------------------------------------------
//      new r = |r|_i 
        vec_abs<<<dimGrid, dimBlock>>>(temp1, r);
//      sum abs(r)_i
        reduce_sum<<<1, N, N*sizeof(float)>>>(temp1, temp1);
//		res = sum
        hipMemcpy(&res, temp1, sizeof(float), hipMemcpyDeviceToHost);
        std::cout<<"residual = "<< res << std::endl; 
		counter++;
        if(counter>A.width)
            break;
    }
	std::cout<< "Steps Taken = " << counter << std::endl;
    hipMemcpy(x,d_x,A.width*sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_x);
    hipFree(r);
    hipFree(d_b);
    hipFree(temp1);
    hipFree(temp2);
    hipFree(alpha);
}

int main() {
// Matrix stuff! 
	Matrix A(N,N); 
	A.width = N; 
	A.height = N;
	load_Matrix("matrix.dat", A);

// Vector stuff! 
    std::vector<float> x(N, 1.f); 
    std::vector<float> b(N, 1.f);

// Gauss-Jacobi Parameters
	float eps = 1e-1; 	

// Call the Gauss-Jacobi algorithms
	grad_descnt(A, x, b, eps); 

	std::cout<<"Soln X = "<<std::endl;
	for(int i = 0; i <10; i++)
		std::cout<< x[i] <<std::endl; //  */
}
