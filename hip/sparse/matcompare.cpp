#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <limits>
#include <fstream>
#include <string>
#include <cmath>
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 32

/* Structs for sparse demonstration */
typedef struct
{
    float            *val; 
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


/*
    Kernel: Computes CSR matrix vector product.
    Input: csrMatrix A, FP32 array x, FP32 array b
    Output: FP32 array b
*/
__global__ void CsrMatVec(const csrMatrix A, const float *x, float *b)
{
	// Have kernel go over each row this will give step 1
	int row = blockDim.x*blockIdx.x + threadIdx.x;
	if(row < A.nrow){
		float dot = 0.0f; 
		int rowStart =   A.rwptr[row];
		int rowEnd = A.rwptr[row + 1]; 
		__syncthreads(); 
		for(int jj = rowStart; jj < rowEnd; jj++)
		{	
            int colId = A.col[jj]; 
			dot += A.val[jj] * x[colId]; // Steps 2, 3, and 4
		}
		b[row] = dot; 
	}
}

/*
    Host function for Sparse Matrix Vector Product. 
    Input: csrMatrix A, FP32 array x, FP32 array b
    Output: FP32 array b
*/
void SPmV(const csrMatrix A, float *x, float *b)
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


/*
    Kernel: Computes Dense Matrix Vector Product. 
    Input: Matrix A, FP32 array x, FP32 array y
    Output: FP32 array y
*/

__global__ void
MatVec (Matrix A, float *x, float *y)
{
    int block_row = blockIdx.x;
    int row = threadIdx.x;
    int tidx = row + block_row*blockDim.x;
    if (!(tidx < A.height))
       return; // thread outside bounds.

    __shared__ float aSub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ volatile float xSub[BLOCK_SIZE];
    float yval = 0.0f; 
    for (int block = 0; block < (A.width+BLOCK_SIZE -1)/BLOCK_SIZE ; block++)
    {
        // grab shared local data for operations
        for (int e = 0; e < BLOCK_SIZE; e++)
            aSub[row][e] = A.elements[A.height * tidx + block * BLOCK_SIZE + e];
        xSub[row] = x[block * BLOCK_SIZE + row];
        // sync threads, all are ready now to compute
        __syncthreads ();

        // multiply sub matrix and sub vector
        for (int e = 0; e < BLOCK_SIZE; e++)
                yval +=  aSub[row][e]* xSub[e];
        __syncthreads ();
    } 
    y[tidx] = yval;
}// shared_mult

void loadDenseMatrix(std::string file, Matrix A)
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

/*
    Helper function to gauge the size of a csrMatrix from file.
    Input: string file, csrMatrix X, int numrow, int numcol
    Output: N/A
*/
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

/*
    Helper function to  of a csrMatrix from file.
    Input: string file, csrMatrix X, int numrow, int numcol
    Output: N/A
*/
void loadCsrMatrix(std::string file, csrMatrix X, int numrow, int numcol)
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

/*
    Main driver function to test the algorithm.
*/
int main()
{
    Matrix A, dA;
    csrMatrix X, dX;  
    float *v, *dV, *dVOut, *spv, *dSpv, *dSpvOut; 
    
    A.width  = 1024; 
    A.height = 1024; 
    dA.width  = A.width;
    dA.height = A.height; 
    X.nrow     = A.height;
    dX.nrow   = X.nrow; 
    
    A.elements = new float[A.width*A.height];
    hipMalloc((void**)&dA.elements,A.width*A.height*sizeof(float));
    loadDenseMatrix("dmatrix.dat", A);
    
    csrMatrixCount("dmatrix.dat",X,A.height,A.width);
    X.val   = new float[X.nvals]; 
    X.col   = new int[X.nvals];
    X.rwptr = new int[X.nrow + 1];
    loadCsrMatrix("dmatrix.dat", X, A.height, A.width);
    
    hipMalloc((void**)&dX.val, X.nvals*sizeof(float)); 
    hipMalloc((void**)&dX.rwptr, (X.nrow + 1)*sizeof(int)); 
    hipMalloc((void**)&dX.col, X.nvals*sizeof(int));
    dX.nvals = X.nvals;
    
    dim3 dimBlock(BLOCK_SIZE); 
    dim3 dimGrid((A.width + BLOCK_SIZE - 1)/BLOCK_SIZE); 
    v   = new float[A.width];
    spv = new float[A.width];
    hipMalloc((void**)&dV,   A.width*sizeof(float)); 
    hipMalloc((void**)&dVOut,   A.width*sizeof(float)); 
    hipMalloc((void**)&dSpv, A.width*sizeof(float)); 
    hipMalloc((void**)&dSpvOut, A.width*sizeof(float)); 
    
    for(int i = 0; i < A.width; i++) {
    	v[i] = 1.0f; 
    	spv[i] = 1.0f; 
    }
    
    //Dense Device Copy	
    hipMemcpy(dA.elements,A.elements,A.height*A.width*sizeof(float),hipMemcpyHostToDevice);
    hipMemcpy(dV, v, A.width*sizeof(float),hipMemcpyHostToDevice);
    
    //Sparse Device Copy
    hipMemcpy(dX.val, X.val, X.nvals*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dX.rwptr, X.rwptr, (X.nrow + 1)*sizeof(int), hipMemcpyHostToDevice); 
    hipMemcpy(dX.col, X.col, X.nvals*sizeof(int), hipMemcpyHostToDevice); 
    hipMemcpy(dSpv, spv, A.width*sizeof(float),hipMemcpyHostToDevice);
      
    float DenseElapsedTime, SpElapsedTime;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0); 
    MatVec<<<dimGrid, dimBlock>>>(dA, dV, dVOut);
    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&DenseElapsedTime, start, stop); 
    hipDeviceSynchronize(); 
    hipMemcpy(v,dVOut,A.height*sizeof(float),hipMemcpyDeviceToHost);
    
    hipEventRecord(start, 0);
    CsrMatVec<<<dimGrid, dimBlock>>>(dX, dSpv, dSpvOut); 
    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&SpElapsedTime, start, stop); 
    hipEventDestroy(start); 
    hipEventDestroy(stop); 
    hipMemcpy(spv, dSpvOut, A.height*sizeof(float), hipMemcpyDeviceToHost);
    
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
    hipFree(dA.elements);
    hipFree(dX.val);
    hipFree(dX.rwptr);
    hipFree(dX.col);
    hipFree(dSpv);
    hipFree(dSpvOut);
    hipFree(dV);
    hipFree(dVOut);
    delete A.elements, X.val, X.rwptr, X.col, v, spv;
}

