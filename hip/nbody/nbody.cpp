#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream> 
#include <string> 
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <vector>

#define EPS2 0.0001
#define BLOCK_SIZE  256 
#define N  2048 

/*
    Device function: computes the body to body interaction. 
    Inputs: float4 bi, float4 bj, float3 ai
    Output: float ai
*/
__device__ void BodyBodyInteraction(float4 bi, float4 bj, float3 &ai)
{

  float3 r;
  // r_ij
  r.x = bj.x - bi.x;
  r.y = bj.y - bi.y;
  r.z = bj.z - bi.z;
  // distSqr = dot(r_ij, r_ij) + EPS^2
  float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
  // invDistCube =1/distSqr^(3/2)
  float distSixth = distSqr * distSqr * distSqr;
  float invDistCube = 1.0f/sqrtf(distSixth);
  // s = m_j * invDistCube
  float s = bj.w * invDistCube;
  // a_i =  a_i + s * r_ij
  ai.x += r.x * s;
  ai.y += r.y * s;
  ai.z += r.z * s;
}

/* 
    Device function: calculates the body to body interaction for the entire tile
    Inputs: float4 myPosition, float4 array shPosition, float3 accel
    Output: float4 array shPosition
*/
__device__ void TileCalculation(float4 myPosition, float4 *shPosition,  float3 &accel)
{
 
  int i;
  for (i = 0; i < blockDim.x; i++) {
    BodyBodyInteraction(myPosition, shPosition[i], accel);
  }
}

/*
    Device function: Calculates the total forces on the system and update acceleration
    Inputs: float4 array d_x, float3 array d_A
    Output: float3 array d_a
*/
__device__ void CalculateForces(float4 *dX, float3 *dA)
{
        __shared__ float4 shPosition[BLOCK_SIZE];
        float4 myPosition;
        int i, tile;
        float3 acc = {0.0f, 0.0f, 0.0f};
        int gtid = blockIdx.x * blockDim.x + threadIdx.x;
        myPosition = dX[gtid];
        for (i = 0, tile = 0; i < N; i += BLOCK_SIZE, tile++)
        {
                    int idx = tile * blockDim.x + threadIdx.x;
                    shPosition[threadIdx.x] = dX[idx];
                    __syncthreads();
                    TileCalculation(myPosition, shPosition ,acc);
                    __syncthreads();
          }
    // Save the result in global memory for the integration step.
    dA[gtid] = acc;
}

/*
    Device function: Advances the positions of the N-body system
    Inputs: float4 scalar X, float3 scalar V, float3 A, float dt
    Output float4 scalar X
*/
__device__ void PosAdvance(float4 &X, const float3 V, const float3 A, float dt)
{
	//this is called by each thread
	X.x += V.x*dt + 0.5f*A.x*dt*dt; 
	X.y += V.y*dt + 0.5f*A.y*dt*dt; 
	X.z += V.z*dt + 0.5f*A.z*dt*dt; 
	X.w = X.w; //Mass stays the same 
}

/*
    Device Function: Advances the velocities of the N-body system
    Inputs: float3 scalar V, float3 scalar A1, float3 scalar A2, float scalar dt
    Output: float3 scalar V 
*/
__device__ void VelAdvance(float3 &V, const float3 A1, const float3 A2, float dt)
{
	//called by every thread
	V.x += 0.5f*(A1.x + A2.x)*dt; 
	V.y += 0.5f*(A1.y + A2.y)*dt; 
	V.z += 0.5f*(A1.z + A2.z)*dt; 
}

/*
    Kernel: Apply the leapfrog algorithm to timestep the system
    Inputs: float4 array X, float3 array V, float3 array A, float scalar dt, int iter
    Output: float4 array X, float3 array V, float3 array A 
*/
__global__ void LeapFrog(float4 *X, float3 *V, float3 *A, float dt, int iter)
{
	int gid = threadIdx.x + blockIdx.x * blockDim.x ;
	if(iter == 0){ // Initial acceleration .
        CalculateForces(X, A) ;
        __syncthreads();
	}
	float3 temp;
// Store acceleration from x ^ n
	temp=A[gid];
	__syncthreads();
// Calculate x ^ n +1
	PosAdvance(X[gid], V[gid], temp, dt) ;
	__syncthreads();
// Calculate acceleration at the n +1 stage
	CalculateForces(X, A) ;
	__syncthreads();
// Calculate v ^ n +1
	VelAdvance(V[gid], temp, A[gid] ,dt) ;
}

/*
    Helper function to output solution set.
    inputs: string file, float4 array X, int M
    output: N/A 
*/
template<class T>
void IoFun(std::string file, T X, int M)
{
    std::ofstream myfile_tsN;
    myfile_tsN.open("data/" + file);
    for(int i = 0; i < M; i++)
    {
        myfile_tsN << X[i].x << '\t' << X[i].y<< '\t' << X[i].z << '\t' << X[i].w << std::endl;
    }

    myfile_tsN.close();
}

/*
    Driver function to calculate nbody system
    Input: float4 array X, float scalar dt, int tio, float tend
    Output: float4 array X, 
*/
void NBody(std::vector<float4> &X, float dt, int tio, float tend)
//X are the positions, dt = time step, tio = io iter, tend = end simulation time, N= #of bodies
{
    float4 *dX;
    float3 *dA, *dV;
    float t = 0.0f;
    int k = 0;
	std::string f; 
    hipMalloc(&dX, N*sizeof(float4));
    hipMalloc(&dV, N*sizeof(float3));
    hipMalloc(&dA, N*sizeof(float3));
	hipMemset(dV, 0.0f, N*sizeof(float3)); 
	hipMemset(dA, 0.0f, N*sizeof(float3));
    hipMemcpy(dX, X.data(), N*sizeof(float4), hipMemcpyHostToDevice);

    dim3 dimGrid(N/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE);

    while(t<tend)
    {
        LeapFrog<<<dimGrid,dimBlock>>>(dX,dV, dA, dt, k);
    	hipDeviceSynchronize();
        if(k%tio==0)
        {
            std::ostringstream ss;
            ss << "f" << std::setw(5) << std::setfill('0') << std::to_string(k);
            f = ss.str() +  ".dat";
            hipMemcpy(X.data(), dX, N*sizeof(float4), hipMemcpyDeviceToHost);
            IoFun(f, X, N);
        }
        t+=dt;
        k++;
    }
    if(k%tio!=0.0f)
    {
         std::ostringstream ss;
         ss << "f" << std::setw(5) << std::setfill('0') << std::to_string(k);
         f = ss.str() +  ".dat";
         hipMemcpy(X.data(), dX, N*sizeof(float4), hipMemcpyDeviceToHost);
         IoFun(f, X, N);
    }
    hipFree(dX);
    hipFree(dA);
    hipFree(dV);
}

/*
   Main driver function, simulates the system until tend is satisfied.
*/
int main()
{
	float dt = 0.00005; 
	int tio = 10; 
	float tend = 0.35;

    std::vector<float4> X(N);
/* Randomized Initial Condition */  
	for(int i = 0; i < N; i++)
	{
		if(i == 0)
		{	
			X[i] = {0.0f, 0.0f, 0.0f, 300.0f};
		}
		else if(i == N/2)
		{
			X[i] = {5.0f, 5.0f, 5.0f, 300.0f};
		}
		else{
			X[i].x = ((float)rand() / (float)(RAND_MAX))*4+0.5f; 
			X[i].y = ((float)rand() / (float)(RAND_MAX))*4+0.5f; 
			X[i].z = ((float)rand() / (float)(RAND_MAX))*4+0.5f; 
			X[i].w = ((float)rand() / (float)(RAND_MAX))*2; 
		}
	}
	IoFun("IC.dat",X,N); //write out initial condition! 
	NBody(X, dt, tio, tend); 
}
