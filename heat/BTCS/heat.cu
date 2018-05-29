#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#define N 128 
#define BLOCK_SIZE 32


__global__ void tridiag_gj(float *diag, float *subdiag, float *supdiag, float *x, float *xout, float *b)
{
        int gid = threadIdx.x + blockDim.x*blockIdx.x;
        float summ1 = 0.0f;
        float temp;

        if(gid>0 && gid < N-1)
        {
                summ1 = subdiag[gid]*x[gid-1] + supdiag[gid]*x[gid+1];
                temp = 1.0f/diag[gid]*(b[gid] - summ1);
        }
        else if(gid==0)
        {
                summ1 = supdiag[gid]*x[gid+1]; 
                temp = 1.0f/diag[gid]*(b[gid] - summ1);
        }
        else
        {
                summ1 = subdiag[gid]*x[gid-1];
                temp = 1.0f/diag[gid]*(b[gid] - summ1);
        }
        xout[gid] = temp;
}

void io_fun(std::string file, float *f)
{
        std::ofstream myfile_tsN;
        myfile_tsN.open(file);
        for(int i = 0; i < N; i++)
        {
                myfile_tsN << f[i] << std::endl;
        }

        myfile_tsN.close();
}

__global__ void bcfill(float *f)
{
        int tid = threadIdx.x + blockIdx.x*blockDim.x;
        if(tid == 0) //use only one thread for 1D BC
        {
                f[0] = f[1];
                f[N-1] = f[N-2];
        }
}

__global__ void create_tridiag(float *diag, float *sub, float *sup, float r)
{
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	if(gid == 0)
	{	
		sup[0] = 0.0f;
		diag[0] = 1.0f; 	
		sub[0] = -r; 
	}
	else if(gid < N-1)
	{
		diag[gid] = 1.0f + 2.0f*r;  
		sub[gid] = -r; 
		sup[gid] = -r; 
	}
	else
	{
		diag[gid] = 1.0f; 
		sup[gid] = -r;
		sub[gid] = 0.0f;  
	}
}

__global__ void fill(float *d_out, const float *d_in)
{
	int gid = threadIdx.x + blockIdx.x*blockDim.x; 
	d_out[gid] = d_in[gid]; 
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

void tridiag_par_gj(float *diag, float *sub, float *sup, float *x, float *b, float eps)
{
        float res = 1.0f;
        float *xnew;
        float *dres;
	int count=0; 
        dres = (float*)malloc(sizeof(float));
	cudaMalloc((float**)&xnew, N*sizeof(float)); 

        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(N/dimBlock.x);
        while(res>eps)
        {
                //Compute x^{n+1}
                tridiag_gj<<<dimGrid,dimBlock>>>(diag, sub, sup, x, xnew, b);

                //Compute vector of residuals
                compute_r<<<dimGrid,dimBlock>>>(x,xnew); //Store r in d_x

                //Reduce vector of residuals to find norm
                reduce_r<<<1,N, N*sizeof(float)>>>(x,x);
                cudaMemcpy(dres, x, sizeof(float), cudaMemcpyDeviceToHost);
                res = dres[0];
                //X = Xnew
                fill<<<dimGrid,dimBlock>>>(x, xnew);
        }
	cudaFree(xnew); 
}

void BTCS(float *f, float dt, float dx, float kappa, float tend, int tio)
{
        float r = kappa*dt/(dx*dx);
        float *sub;
        float *diag;
        float *sup;
        float *d_f, *d_f1;
        float t = 0.0f;
        int k =  0;
	float eps = 1e-3; 
	std::string f2; 
	size_t sz = N*sizeof(float); 
        cudaMalloc((void**)&sub, sz);
        cudaMalloc((void**)&diag, sz);
        cudaMalloc((void**)&sup, sz);
        cudaMalloc((void**)&d_f, sz);
        cudaMalloc((void**)&d_f1, sz);
	cudaMemcpy(d_f, f, sz, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_f1, f, sz, cudaMemcpyHostToDevice); 

        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(N/BLOCK_SIZE);

        create_tridiag<<<dimGrid,dimBlock>>>(diag, sub, sup, r); //creates tridiag for BTCS
        while(t<tend)
        {
                tridiag_par_gj(diag, sub, sup, d_f1, d_f, eps); //Calculate new value!
                // or
                // tridiag_grad_descent(args);
		
                fill<<<dimGrid,dimBlock>>>(d_f, d_f1); //u^n+1
                if(k%tio==0)
                {
                        f2 = "sol" + std::to_string(k) + ".dat";
                        cudaMemcpy(f,d_f, sz, cudaMemcpyDeviceToHost);
                        io_fun(f2, f); 
                }

                t+=dt;
                k++;
        }
        if(k%tio!=0)
        {
                f2 = "final_sol.dat";
                cudaMemcpy(f,d_f, sz, cudaMemcpyDeviceToHost);
                io_fun(f2, f);     
        }

        cudaFree(sub);
        cudaFree(sup);
        cudaFree(diag);
        cudaFree(d_f);
        cudaFree(d_f1);
}

int main()
{
        float k = 1.0f;
        float dx = 2.0f/float(N);
        float dt = 100.f*0.5f*(dx*dx)/k;
        float tmax = 2.0f;
	int tio = 20; 
        //Allocate Memory 
        size_t sz = N*sizeof(float);
        float *f;
	float x = -1.0f; 
        f = (float*)malloc(sz);

        //Apply Initial Condition You could also create a kernel for this
        for(int i=0; i < N; i++)
        {
		x += dx; 
                f[i] = exp(-0.5f*x*x);
        }

        /* IO Operations for IC */
        std::string f1 = "IC.dat";
        io_fun(f1, f);

        /*Perform Integration */
	BTCS(f, dt, dx, k, tmax, tio); 
	
        //deallocate memory 
        free(f);
}


