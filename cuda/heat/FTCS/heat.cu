#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#define M 1024

__global__ void ftcs(float f_new[], const float f[], const float dx,
                     const float k, const float dt) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > 0 && tid < M - 1) // Skip boundaries!
  {
    float temp2 =
        f[tid] + k * dt / (dx * dx) * (f[tid + 1] - 2 * f[tid] + f[tid - 1]);
    f_new[tid] = temp2;
  }
}

__global__ void initialize(float f[], float x[], const float dx) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float xt = -1.f + tid * dx;
  x[tid] = xt;
  f[tid] = exp(-0.5f * xt * xt);
}

__global__ void equate(float f_old[], const float f_new[]) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  f_old[tid] = f_new[tid];
}

__global__ void bc(float *f) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) // use only one thread for 1D BC
  {
    f[0] = f[1];
    f[M - 1] = f[M - 2];
  }
}

void io_fun(std::string file, float *x, float *f) {
  std::ofstream myfile_tsN;
  myfile_tsN.open(file);
  for (int i = 0; i < M; i++) {
    myfile_tsN << x[i] << '\t';
    myfile_tsN << f[i] << std::endl;
  }

  myfile_tsN.close();
}

int main() {
  // Integration Parameters
  float k = 1.0f; // Thermal Conductivity
  /* Numerical Mesh Configuration */
  float dx = 2.0f / float(M);
  float dt = 0.5f * (dx * dx) / k;

  float tmax = 0.5f;
  float t = 0.0f, tio = 0.125f;

  // Allocate Memory
  size_t sz = M * sizeof(float);
  float *f, *x;
  f = new float[M];
  x = new float[M];

  float *d_f1, *d_f2, *d_x;
  cudaMalloc(&d_f1, sz);
  cudaMalloc(&d_f2, sz);
  cudaMalloc(&d_x, sz);

  // Kernel parameters
  dim3 dimBlock(16, 1, 1);
  dim3 dimGrid(M / dimBlock.x, 1, 1);

  // Apply Initial Condition You could also create a kernel for this
  initialize<<<dimGrid, dimBlock>>>(d_f1, d_x, dx);

  // Copy for IO operation
  cudaMemcpy(x, d_x, sz, cudaMemcpyDeviceToHost);

  // device x is no longer needed
  cudaFree(d_x);
  /*====================== Perform Integration =======================*/

  std::string f2;
  int kk = 0;
  while (t < tmax) {
    // Call the stencil routine
    ftcs<<<dimGrid, dimBlock>>>(d_f2, d_f1, dx, k, dt);
    // Call BC
    bc<<<dimGrid, dimBlock>>>(d_f2);
    equate<<<dimGrid, dimBlock>>>(d_f1, d_f2);
    if (fmod(t, tio) == 0.0f) {
      // IO function
      f2 = "sol" + std::to_string(kk) + ".dat";
      cudaMemcpy(f, d_f2, sz, cudaMemcpyDeviceToHost);
      io_fun(f2, x, f);
      kk++;
    }

    t += dt;
  }

  if (fmod(tmax, tio) != 0.0f) { // IO Function
    f2 = "final_sol.dat";
    cudaMemcpy(f, d_f1, sz, cudaMemcpyDeviceToHost);
    io_fun(f2, x, f);
  }

  // deallocate memory
  delete x, f;
  cudaFree(d_f1);
  cudaFree(d_f2);
}
