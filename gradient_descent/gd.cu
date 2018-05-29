#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#define N 1024 
#define BLOCK_SIZE 16 

struct Matrix
{
        float           *elements;
        int                 width;
        int                height;
};


void load_Matrix(std::string file, Matrix A)
{
        std::ifstream f;
        f.open(file);
        for( int i = 0; i <A.height; i++)
                for(int j = 0; j < A.width; j++)
                {
                        f >> A.elements[j + A.width*i];
                }
        f.close();
}

template <class T>
__global__ void reduce_sum(T * d_out, const T *d_in)
{
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

template <class T>
__global__ void pairwise_mult(const T *x, const T *y, T *z)
{
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	z[gid] = x[gid]*y[gid]; 
}

// Pass by reference
template <class T> 
__global__ void saxpy(T *x, T *y, T *a)
{
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	x[gid] = x[gid] + a[0]*y[gid]; 
}

// Pass by value
template <class T>
__global__ void saxpy2(T *out,const T *x, const T *y, T a)
{
	int gid = threadIdx.x + blockDim.x*blockIdx.x; 
	out[gid] = x[gid] + a*y[gid]; 
}

__global__ void
mat_vec (Matrix A, float *x, float *y)
{
        int tidx = threadIdx.x + blockIdx.x*blockDim.x;
        if (tidx > A.width)
                return; // thread outside bounds.
	float yval = 0.0f; 
       for (int e = 0; e < A.width ; e++)
        {
          yval +=  A.elements[A.width * tidx + e]* x[e];
        }
        y[tidx] = yval;

}

__global__ void vec_abs(float *xout, float *xin)
{
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	xout[gid] = fabs(xin[gid]); 
}

template <class T>
__global__ void devide(T *xout, T *num, T* denom)
{
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	xout[gid] = num[gid]/denom[gid]; 
}

void grad_descnt(Matrix A, float *x, float *b, float eps)
{
        float *r, *d_b, *d_x, *temp1, *temp2;
        float *alpha, res, *dres;
        int counter = 0;
        Matrix d_A;
	d_A.width = A.width; 
	d_A.height = d_A.height; 
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(A.width/dimBlock.x);

        cudaMalloc(&r, A.width*sizeof(float));
        cudaMalloc(&d_b, A.width*sizeof(float));
        cudaMalloc(&d_x, A.width*sizeof(float));
        cudaMalloc(&d_A.elements, A.width*A.height*sizeof(float));
        cudaMalloc(&alpha, sizeof(float));
	dres = (float*)malloc(sizeof(float)); 
	res = 1.0f; 
        cudaMalloc(&temp1, A.width*sizeof(float));
        cudaMalloc(&temp2, A.width*sizeof(float));

        cudaMemcpy(d_b, b, A.width*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, A.width*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A.elements, A.elements, A.width*A.height*sizeof(float), cudaMemcpyHostToDevice);

//Compute R
        mat_vec<<<dimGrid, dimBlock>>>(d_A, d_x, temp1);
        cudaDeviceSynchronize();
        saxpy2<<<dimGrid, dimBlock>>>(r, temp1, d_b, -1.0f);
        cudaDeviceSynchronize(); 


        while(res > eps)
        {
//      Calculate Alpha ----------------------------------------------------------------
		//compute p^Tr
                pairwise_mult<<<dimGrid, dimBlock>>>(r, r, temp1);
                reduce_sum<<<1,N,N*sizeof(float)>>>(temp1, temp1);
		//compute p^TAp
                mat_vec<<<dimGrid, dimBlock>>>(d_A,r,temp2);
                pairwise_mult<<<dimGrid, dimBlock>>>(r, temp2, temp2);
                reduce_sum<<<1, N, N*sizeof(float)>>>(temp2, temp2);
                cudaDeviceSynchronize();
		// alpha = r^Tr/p^TAp
                devide<<<1,1>>>(alpha,temp1,temp2);

//      X^(k+1) ---------------------------------------------------------------------------
//      X = X + alpha P
                saxpy<<<dimGrid, dimBlock>>>(d_x, r, alpha);
//      r = grad(f) -----------------------------------------------------------------------
//      Calculate new r
                mat_vec<<<dimGrid, dimBlock>>>(d_A, d_x, temp1);
                saxpy2<<<dimGrid, dimBlock>>>(r, d_b, temp1, -1.0f);

//       ||r|| ----------------------------------------------------------------------------
//      	new r = |r|_i 
                vec_abs<<<dimGrid, dimBlock>>>(temp1, r);
//              sum abs(r)_i
                reduce_sum<<<1, N, N*sizeof(float)>>>(temp1, temp1);
//		res = sum
                cudaMemcpy(dres, temp1, sizeof(float), cudaMemcpyDeviceToHost);
		res = dres[0]; 
                std::cout<<"residual = "<< res << std::endl; 
		counter++;
                if(counter>A.width)
                        break;
        }
	std::cout<< "Steps Taken = " << counter << std::endl;
        cudaMemcpy(x,d_x,A.width*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_x);
        cudaFree(d_A.elements);
        cudaFree(r);
        cudaFree(d_b);
        cudaFree(temp1);
        cudaFree(temp2);
        cudaFree(alpha);
}




int main()
{

// Matrix stuff! 
	Matrix A; 
	A.width = N; 
	A.height = N;
	A.elements = (float*)malloc(N*N*sizeof(float)); 
	load_Matrix("matrix.dat", A);

// Vector stuff! 
	float *x, *b; 
	x = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float)); 

	for(int i =0; i < N; i++)
	{
		b[i] = 1.0f; 
		x[i] = 1.0f;
	}

// Gauss-Jacobi Parameters
	float eps = 1e-1; 	

// Call the Gauss-Jacobi algorithms
	grad_descnt(A, x, b, eps); 

	std::cout<<"Soln X = "<<std::endl;
	for(int i = 0; i <10; i++)
		std::cout<< x[i] <<std::endl; //  */
	free(x); 
	free(b);
	free(A.elements); 
}

