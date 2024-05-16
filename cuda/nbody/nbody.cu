#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>

#define EPS2 0.0001
#define BLOCK_SIZE 256
#define N 2048

__device__ void bodyBodyInteraction(float4 bi, float4 bj, float3 &ai) {

  float3 r;
  // r_ij
  r.x = bj.x - bi.x;
  r.y = bj.y - bi.y;
  r.z = bj.z - bi.z;
  // distSqr = dot(r_ij, r_ij) + EPS^2
  float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
  // invDistCube =1/distSqr^(3/2)
  float distSixth = distSqr * distSqr * distSqr;
  float invDistCube = 1.0f / sqrtf(distSixth);
  // s = m_j * invDistCube
  float s = bj.w * invDistCube;
  // a_i =  a_i + s * r_ij
  ai.x += r.x * s;
  ai.y += r.y * s;
  ai.z += r.z * s;
}

__device__ void tile_calculation(float4 myPosition, float4 *shPosition,
                                 float3 &accel) {

  int i;
  for (i = 0; i < blockDim.x; i++) {
    bodyBodyInteraction(myPosition, shPosition[i], accel);
  }
}

__device__ void calculate_forces(float4 *d_X, float3 *d_A) {
  __shared__ float4 shPosition[BLOCK_SIZE];
  float4 myPosition;
  int i, tile;
  float3 acc = {0.0f, 0.0f, 0.0f};
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  myPosition = d_X[gtid];
  for (i = 0, tile = 0; i < N; i += BLOCK_SIZE, tile++) {
    int idx = tile * blockDim.x + threadIdx.x;
    shPosition[threadIdx.x] = d_X[idx];
    __syncthreads();
    tile_calculation(myPosition, shPosition, acc);
    __syncthreads();
  }
  // Save the result in global memory for the integration step.
  d_A[gtid] = acc;
}

__device__ void pos_advance(float4 &X, const float3 V, const float3 A,
                            float dt) {
  // this is called by each thread
  X.x += V.x * dt + 0.5f * A.x * dt * dt;
  X.y += V.y * dt + 0.5f * A.y * dt * dt;
  X.z += V.z * dt + 0.5f * A.z * dt * dt;
  X.w = X.w; // Mass stays the same
}

__device__ void vel_advance(float3 &V, const float3 A1, const float3 A2,
                            float dt) {
  // called by every thread
  V.x += 0.5f * (A1.x + A2.x) * dt;
  V.y += 0.5f * (A1.y + A2.y) * dt;
  V.z += 0.5f * (A1.z + A2.z) * dt;
}

__global__ void leapfrog(float4 *X, float3 *V, float3 *A, float dt, int iter) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (iter == 0) { // Initial acceleration .
    calculate_forces(X, A);
    __syncthreads();
  }
  float3 temp;
  // Store acceleration from x ^ n
  temp = A[gid];
  __syncthreads();
  // Calculate x ^ n +1
  pos_advance(X[gid], V[gid], temp, dt);
  __syncthreads();
  // Calculate acceleration at the n +1 stage
  calculate_forces(X, A);
  __syncthreads();
  // Calculate v ^ n +1
  vel_advance(V[gid], temp, A[gid], dt);
}

void io_fun(std::string file, float4 *X, int M) {
  std::ofstream myfile_tsN;
  myfile_tsN.open(file);
  for (int i = 0; i < M; i++) {
    myfile_tsN << X[i].x << '\t' << X[i].y << '\t' << X[i].z << '\t' << X[i].w
               << std::endl;
  }

  myfile_tsN.close();
}

void nbody(float4 *X, float dt, int tio, float tend)
// X are the positions, dt = time step, tio = io iter, tend = end simulation
// time, N= #of bodies
{
  float4 *d_X;
  float3 *d_A, *d_V;
  float t = 0.0f;
  int k = 0;
  std::string f;
  cudaMalloc((void **)&d_X, N * sizeof(float4));
  cudaMalloc((void **)&d_V, N * sizeof(float3));
  cudaMalloc((void **)&d_A, N * sizeof(float3));
  cudaMemset(d_V, 0.0f, N * sizeof(float3));
  cudaMemset(d_A, 0.0f, N * sizeof(float3));
  cudaMemcpy(d_X, X, N * sizeof(float4), cudaMemcpyHostToDevice);
  dim3 dimGrid(N / BLOCK_SIZE);
  dim3 dimBlock(BLOCK_SIZE);
  while (t < tend) {
    leapfrog<<<dimGrid, dimBlock>>>(d_X, d_V, d_A, dt, k);
    if (k % tio == 0) {
      f = "f" + std::to_string(k) + ".dat";
      cudaMemcpy(X, d_X, N * sizeof(float4), cudaMemcpyDeviceToHost);
      io_fun(f, X, N);
    }
    t += dt;
    k++;
  }
  if (k % tio != 0.0f) {
    f = "f" + std::to_string(k) + ".dat";
    cudaMemcpy(X, d_X, N * sizeof(float4), cudaMemcpyDeviceToHost);
    io_fun(f, X, N);
  }
  cudaFree(d_X);
  cudaFree(d_A);
  cudaFree(d_V);
}

int main() {
  float4 *X;
  float dt = 0.001;
  int tio = 50;
  float tend = 0.5;

  X = (float4 *)malloc(sizeof(float4) * N);
  /* Initial Condition */
  for (int i = 0; i < N; i++) {
    if (i == 0) {
      X[i] = {0.0f, 0.0f, 0.0f, 300.0f};

    } else if (i == N / 2) {
      X[i] = {5.0f, 5.0f, 5.0f, 1.0f};
    } else {
      X[i].x = ((float)rand() / (RAND_MAX)) * 4 + 0.5f;
      X[i].y = ((float)rand() / (RAND_MAX)) * 4 + 0.5f;
      X[i].z = ((float)rand() / (RAND_MAX)) * 4 + 0.5f;
      //		X[i].w = ((double)rand() / (RAND_MAX));
      X[i].w = ((float)rand() / (RAND_MAX)) * 2;
    }
  }
  io_fun("IC.dat", X, N); // write out initial condition!
  nbody(X, dt, tio, tend);

  free(X);
}
