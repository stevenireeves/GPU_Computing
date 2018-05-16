#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <limits>
#include <fstream>
#include <string>
#include <cmath>

#define BLOCK_SIZE 32

typedef struct
{
        float		 *val; 
        int              *col; 
	int            *rwptr; 
	int             nvals; 
	int    	        nrow;
}  csrMatrix; 

typedef struct
{
	float		*elements;
	int 		    width; 
	int 		   height;
} Matrix; 


__global__ void csr_mat_vec(const csrMatrix A, const float *x, float *b)
{
	// Have kernel go over each row this will give step 1
	int row = blockDim.x*blockIdx.x + threadIdx.x;
	if(row < A.nrow){
		float dot = 0.0f; 
		int row_start =   A.rwptr[row];
		int row_end = A.rwptr[row + 1]; 
		__syncthreads(); 
		for(int jj = row_start; jj < row_end; jj++)
		{	int colId = A.col[jj]; 
			dot += A.val[jj] * x[colId]; // Steps 2, 3, and 4
		}
		b[row] = dot; 
	}
}

void spmatvec(const csrMatrix A, const float *x, float *b)
{

	int colId; 
	for(int row = 0; row <A.nrow; row++)
	{
		float dot = 0.0f; 
		int row_start = A.rwptr[row]; 
		int row_end = A.rwptr[row+1];
		for(int jj = row_start; jj < row_end; jj++)
		{
			colId = A.col[jj];
			dot += A.val[jj]*x[colId]; 
		}
		b[row] = dot; 
	}
}

/*__global__ void mat_vec(Matrix A, float *x, float *y)
{

  int bid = blockIdx.x;
  float y_val = 0.0;
  float* Asub;

  int row = threadIdx.x;
  __shared__ float x_shared[BLOCK_SIZE]; 

  for (unsigned int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    if (row < BLOCK_SIZE)
	{
     	 x_shared[row] = x[row + m*BLOCK_SIZE];
	}
	 Asub = A.elements + BLOCK_SIZE*(bid + m*A.width); 
	 __syncthreads();


    for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
      if (row < BLOCK_SIZE)
        y_val += Asub[row + e * A.width] * x_shared[e];
    	}
    
    __syncthreads();
  }

  y[row + bid*blockDim.x] = y_val; 

}//*/

__global__ void
mat_vec (Matrix A, float *x, float *y)
{
        int block_row = blockIdx.x;
        int row = threadIdx.x;
        int tidx = row + block_row*blockDim.x;
        if (!(tidx < A.height))
                return; // thread outside bounds.

        __shared__ volatile float xsub[BLOCK_SIZE];
        float yval = 0.0f; 
        for (int block = 0; block < (A.width+BLOCK_SIZE -1)/BLOCK_SIZE ; block++)
        {
                // grab shared local data for operations
                xsub[row] = x[block * BLOCK_SIZE + row];
                // sync threads, all are ready now to compute
                __syncthreads ();

                // multiply sub matrix and sub vector
                for (int e = 0; e < BLOCK_SIZE; e++)
                        yval +=  A.elements[A.height * tidx + block * BLOCK_SIZE + e]* xsub[e];
                __syncthreads ();
        } 
                y[tidx] = yval;

}// shared_mult

/*
__global__ void mat_vec(Matrix A, float *x, float *y)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float dot = 0; 
	for( int i = 0; i < A.width; i++)
		dot += A.elements[i + A.width*tid]*x[i];
	__syncthreads();
	y[tid] = dot; 
} //*/

void load_denseMatrix(std::string file, Matrix A)
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

void csrMatrixCount(std::string file, csrMatrix &X, int numrow, int numcol)
{
	float temp; 
	int k = 0;
	std::ifstream f;
	f.open(file); 

	for(int j = 0; j < numrow; j++)
	{
		for(int i = 0; i < numcol; i++)
		{
			f >> temp; 
			if(std::abs(temp) >= std::numeric_limits<float>::epsilon())
			{
				k++;
			}
		}
	}
	f.close();
	X.nvals = k; 

}

void load_csrMatrix(std::string file, csrMatrix X, int numrow, int numcol)
{
	float temp; 
	int  index = 0;
	std::ifstream f;
	f.open(file); 
	X.rwptr[0] = 0; 

	for(int j = 0; j < numrow; j++)
	{
		for(int i = 0; i < numcol; i++)
		{
			f >> temp;
			if(std::abs(temp) >= std::numeric_limits<float>::epsilon())
			{
				X.val[index] = temp;
				X.col[index] = i;
				index++;	 
			}
		}
			X.rwptr[j+1] = index;
	}
	f.close();
}


int main()
{
	Matrix A, d_A;
	csrMatrix X, d_X;  
	float *v, *d_v, *d_v_out, *spv, *d_spv, *d_spv_out; 

	A.width  = 1024; 
	A.height = 1024; 
	d_A.width  = A.width;
	d_A.height = A.height; 
	X.nrow     = A.height;
	d_X.nrow   = X.nrow; 

	A.elements = (float*)malloc(A.width*A.height*sizeof(float));
	cudaMalloc((void**)&d_A.elements,A.width*A.height*sizeof(float));
	load_denseMatrix("dmatrix.dat", A);
	
	csrMatrixCount("dmatrix.dat",X,A.height,A.width);
	X.val = (float*)malloc(X.nvals*sizeof(float)); 
	X.col = (int*)malloc(X.nvals*sizeof(int)); 
	X.rwptr = (int*)malloc((X.nrow + 1)*sizeof(int));
	load_csrMatrix("dmatrix.dat", X, A.height, A.width);
	
	cudaMalloc((void**)&d_X.val, X.nvals*sizeof(float)); 
	cudaMalloc((void**)&d_X.rwptr, (X.nrow + 1)*sizeof(int)); 
	cudaMalloc((void**)&d_X.col, X.nvals*sizeof(int));
	d_X.nvals = X.nvals;

	dim3 dimBlock(BLOCK_SIZE); 
	dim3 dimGrid((A.width + BLOCK_SIZE - 1)/BLOCK_SIZE); 
	v   = (float*)malloc(A.width*sizeof(float)); 
	spv = (float*)malloc(A.width*sizeof(float)); 
	cudaMalloc((void**)&d_v,   A.width*sizeof(float)); 
	cudaMalloc((void**)&d_v_out,   A.width*sizeof(float)); 
	cudaMalloc((void**)&d_spv, A.width*sizeof(float)); 
	cudaMalloc((void**)&d_spv_out, A.width*sizeof(float)); 

	for(int i = 0; i < A.width; i++)
	{
		v[i] = 1.0f; 
		spv[i] = 1.0f; 
	}

	//Dense Device Copy	
	cudaMemcpy(d_A.elements,A.elements,A.height*A.width*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v, A.width*sizeof(float),cudaMemcpyHostToDevice);

	//Sparse Device Copy
	cudaMemcpy(d_X.val, X.val, X.nvals*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_X.rwptr, X.rwptr, (X.nrow + 1)*sizeof(int), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_X.col, X.col, X.nvals*sizeof(int), cudaMemcpyHostToDevice); 
 	cudaMemcpy(d_spv, spv, A.width*sizeof(float),cudaMemcpyHostToDevice);
      
	float DenseElapsedTime, SpElapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0); 
	mat_vec<<<dimGrid, dimBlock>>>(d_A, d_v, d_v_out);
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&DenseElapsedTime, start, stop); 
	cudaDeviceSynchronize(); 
	cudaMemcpy(v,d_v_out,A.height*sizeof(float),cudaMemcpyDeviceToHost);
	
        cudaEventRecord(start, 0);
	csr_mat_vec<<<dimGrid, dimBlock>>>(d_X, d_spv, d_spv_out); 
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&SpElapsedTime, start, stop); 
	cudaEventDestroy(start); 
	cudaEventDestroy(stop); 
	cudaMemcpy(spv, d_spv_out, A.height*sizeof(float), cudaMemcpyDeviceToHost);

	std::cout<<" Dense Mv Time = "<< DenseElapsedTime << "ms"<<std::endl;
	std::cout<<" SpMv Time = "<<SpElapsedTime <<"ms"<<std::endl; 

	float norm = 0.0f; 
	for(int i = 0; i< A.height; i++)
	{
		norm = abs(spv[i] - v[i]);
		if(norm > 0.0)
		{	
			std::cout<< "Matrix Vector Production Incorrect! Error = " <<  norm <<std::endl;
			std::cout<< "spv = " << spv[i] << " dense = " << v[i] << " loc = "<<i<<std::endl;
			return -1; 
		}
		
	}
	cudaFree(d_A.elements);
	cudaFree(d_X.val);
	cudaFree(d_X.rwptr);
	cudaFree(d_X.col);
	cudaFree(d_spv);
	cudaFree(d_spv_out);
	cudaFree(d_v);
	cudaFree(d_v_out);
	free(A.elements);
	free(X.val);
	free(X.rwptr);
	free(X.col);
	free(v);
	free(spv);
}

