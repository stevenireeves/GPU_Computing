#include <iostream>

#include <hip/hip_runtime.h>

#include "mat.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8


__global__ void transposeReg(Matrix odata, const Matrix idata)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
	int width = gridDim.x * TILE_DIM;	

	odata.elements[tidx*width + tidy] = idata.elements[tidy*width + tidx];
}

__global__ void transposeNaive(Matrix odata, const Matrix idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata.elements[x*width + (y+j)] = idata.elements[(y+j)*width + x];
}

__global__ void transposeCoalesced(Matrix odata, const Matrix idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata.elements[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata.elements[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

double denom(const float time)
{
    return (double)time * 1e6;
}

int main()
{
	int N = 1024;
	Matrix A(N, N);
	Matrix d_A(N, N, 1);
	Matrix d_At(N, N, 1);
	
	d_At.load(d_A, 1);
	float time0, time1, time2;
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	dim3 dimGrid(N/TILE_DIM, N/TILE_DIM);
	dim3 block1(TILE_DIM, TILE_DIM);
	hipEventRecord(start);
	transposeReg<<<dimGrid, block1>>>(d_At, d_A);
	hipEventRecord(stop);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&time0, start, stop);

	dim3 block2(TILE_DIM, TILE_DIM/BLOCK_ROWS);
	hipEventRecord(start);
	transposeNaive<<<dimGrid, block2>>>(d_At, d_A);
	hipEventRecord(stop);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&time1, start, stop);

	hipEventRecord(start);
	transposeCoalesced<<<dimGrid, block2>>>(d_At, d_A);
	hipEventRecord(stop);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&time2, start, stop);


	std::cout<< "First try time = " << time0 << " ms"<<std::endl;
	std::cout<< "Second try time = " << time1 << " ms" << std::endl;
	std::cout<< "Third try time = " << time2 << " ms"<< std::endl;

    long int numer = (long int)(N * N)*4*2; 
    double bw0 = numer/denom(time0);
    double bw1 = numer/denom(time1);
    double bw2 = numer/denom(time2);

    std::cout<< "First try bandwidth = " << bw0 << "GB/s" << std::endl;
    std::cout<< "Second try bandwidth = " << bw1 << "GB/s" << std::endl;
    std::cout<< "Third try bandwidth = " << bw2 << "GB/s" << std::endl;

	hipEventDestroy(start);
	hipEventDestroy(stop);

	d_A.gpu_deallocate();
	d_At.gpu_deallocate();
	A.cpu_deallocate();
}
