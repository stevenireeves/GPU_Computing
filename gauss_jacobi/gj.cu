#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#define N 1024 
#define BLOCK_SIZE 16 

typedef struct
{
        float           *elements;
        int                 width;
        int                height;
} Matrix;


void cpu_gj(Matrix A, float *x, float *b, float eps)
{
	float res = 1.0f; 
	float summ1, summ2;
	float *temp;
	temp = new float[A.width]; 
	int counter = 0; 
	while(res > eps)
	{
		summ2 = 0.0f; 
		for(int i = 0; i < A.width; i++)
		{
			summ1 = 0.0f; 
			for(int k =0; k < A.width; k++)
				if(k!=i)
					summ1 += A.elements[k + i*A.width]*x[k]; 
			temp[i] = 1/A.elements[i+i*A.width]*(b[i] - summ1); 
			summ2 += abs(temp[i] - x[i]);  
		}
		for(int i = 0; i < A.width; i++)
			x[i] = temp[i]; 
		res = summ2; 
		counter++; 
		if(counter==A.width)
			break; 
	}
	delete[] temp; 
	std::cout<<"Steps Taken to Convergence = "<< counter<<std::endl;
}

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


__global__ void naive_gj(Matrix A, float *x, float *xout, float *b) //Computes one iteration of GJ
{
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	float summ1 = 0.0f; 
	float temp; 
	for (int k =0; k < A.width; k++)
	{
		if(k!= gid)
			summ1 += A.elements[k + gid*A.width]*x[k]; 
	} 
	temp = 1.0f/A.elements[gid + gid*A.width]*(b[gid] - summ1);
	xout[gid] = temp; 
}

__global__ void compute_r(float *xold, const float *xnew) //store abs(diff) in xold
{
	int gid = threadIdx.x + blockDim.x*blockIdx.x; 
	float temp = fabs(xnew[gid] - xold[gid]); 
	xold[gid] = temp; 
}

__global__ void reduce_r(float * d_out, const float *d_in)
{
        // sdata is allocated in the kernel call: via dynamic shared memeory
        extern __shared__ float sdata[];

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

__global__ void fill(float *xout, float *xin)
{
	int gid = threadIdx.x + blockDim.x*blockIdx.x; 
	xout[gid] = xin[gid];
}

void par_gj(Matrix A, float *x, float *b, float eps)
{
        float res = 1.0f;
        int counter = 0;
        Matrix d_A;
        d_A.width = A.width;
        d_A.height = A.height;
        float *d_x, *d_b, *d_xnew;
	float *dres; 
	dres = (float*)malloc(sizeof(float));
        cudaMalloc((void**)&d_A.elements, A.width*A.height*sizeof(float));
        cudaMalloc((void**)&d_x, A.width*sizeof(float));
        cudaMalloc((void**)&d_b, A.height*sizeof(float));
        cudaMalloc((void**)&d_xnew, A.width*sizeof(float));

        cudaMemcpy(d_A.elements,A.elements,A.width*A.height*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, A.width*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, A.height*sizeof(float),cudaMemcpyHostToDevice);

        dim3 dimBlock(16);
        dim3 dimGrid((A.width+ dimBlock.x - 1)/dimBlock.x);
        while(res>eps)
        {
                //Compute x^{n+1}
                naive_gj<<<dimGrid,dimBlock>>>(d_A, d_x, d_xnew, d_b);
                cudaDeviceSynchronize();

                //Compute vector of residuals
                compute_r<<<dimGrid,dimBlock>>>(d_x,d_xnew); //Store r in d_x
                cudaDeviceSynchronize();

                //Reduce vector of residuals to find norm
                reduce_r<<<1,N, N*sizeof(float)>>>(d_x, d_x);
                cudaMemcpy(dres, d_x, sizeof(float), cudaMemcpyDeviceToHost);
		res = dres[0]; 
		std::cout<<res<<std::endl;

                //X = Xnew
                fill<<<dimGrid,dimBlock>>>(d_x, d_xnew);
                cudaDeviceSynchronize();
                counter++;
                if(counter==A.width)
                        break;
        }
        	
	std::cout<<"Steps Taken to Convergence = "<< counter<<std::endl;
	//export X
        cudaMemcpy(x, d_x, A.width*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_x);
        cudaFree(d_A.elements);
        cudaFree(d_xnew);
        cudaFree(d_b);
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
		x[i] = 0.0f;
	}

// Gauss-Jacobi Parameters
	float eps = 1e-7; 	

// Call the Gauss-Jacobi algorithms
	par_gj(A, x, b, eps); 

	std::cout<<"Soln X = "<<std::endl;
	for(int i = 0; i <10; i++)
		std::cout<< x[i] <<std::endl; //  */
	free(x); 
	free(b);
	free(A.elements); 
}

