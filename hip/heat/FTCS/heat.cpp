#include <cmath>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#define M 1024

/*
    Kernel: Computes one step of the forward time centered space algorithm.
    Input: FP32 array f_new, FP32 array f, FP32 dx, FP32 k, FP32 dt
    Output: FP32 array f_new
*/
__global__ void ForwardTimeCenteredSpace(float fNew[], const float f[],
                                         const float dx, const float k,
                                         const float dt) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > 0 && tid < M - 1) // Skip boundaries!
  {
    float temp2 =
        f[tid] + k * dt / (dx * dx) * (f[tid + 1] - 2 * f[tid] + f[tid - 1]);
    fNew[tid] = temp2;
  }
}

/*
    Kernel: Applies the Initial Conditions
    Inputs: FP32 array f, FP32 array x, FP32 dx
    Output: FP32 array f
*/
__global__ void Initialize(float f[], float x[], const float dx) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float xt = -1.f + tid * dx;
  x[tid] = xt;
  f[tid] = exp(-0.5f * xt * xt);
}

/*
    Kernel: Fills f_new with the contents of f_old
    Inputs: FP32 array f_old, FP32 array f_new
    Output: FP32 array f_new
*/
__global__ void Equate(float fOld[], const float fNew[]) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  fOld[tid] = fNew[tid];
}

/*
    Kernel: Applies the outflow boundary conditions.
    Inputs: FP32 array f
    Output: FP32 array f
*/
__global__ void BoundaryCondition(float f[]) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) // use only one thread for 1D BC
  {
    f[0] = f[1];
    f[M - 1] = f[M - 2];
  }
}

/*
    Helper function to write array to filesystem
*/
void IoFun(std::string file, std::vector<float> x, std::vector<float> f) {
  std::ofstream myFileTsN;
  myFileTsN.open(file);
  for (int i = 0; i < M; i++) {
    myFileTsN << x[i] << '\t';
    myFileTsN << f[i] << std::endl;
  }

  myFileTsN.close();
}

/*
    Driver function to simulate the heat profile in a 1-D bar.
*/
int main() {
  // Integration Parameters
  float k = 1.0f; // Thermal Conductivity
  /* Numerical Mesh Configuration */
  float dx = 2.0f / float(M);
  float dt = 0.5f * (dx * dx) / k;

  float tMax = 0.5f;
  float t = 0.0f, tIo = 0.125f;

  // Allocate Memory
  size_t sz = M * sizeof(float);
  std::vector<float> f(M, 0), x(M, 0);

  float *dF1, *dF2, *dX;
  hipMalloc(&dF1, sz);
  hipMalloc(&dF2, sz);
  hipMalloc(&dX, sz);

  // Kernel parameters
  dim3 dimBlock(16, 1, 1);
  dim3 dimGrid(M / dimBlock.x, 1, 1);

  // Apply Initial Condition You could also create a kernel for this
  Initialize<<<dimGrid, dimBlock>>>(dF1, dX, dx);

  // Copy for IO operation
  hipMemcpy(x.data(), dX, sz, hipMemcpyDeviceToHost);

  // device x is no longer needed
  hipFree(dX);
  /*====================== Perform Integration =======================*/

  std::string f2;
  int kk = 0;
  while (t < tMax) {
    // Call the stencil routine
    ForwardTimeCenteredSpace<<<dimGrid, dimBlock>>>(dF2, dF1, dx, k, dt);
    //		hipDeviceSynchronize();
    // Call BC
    BoundaryCondition<<<dimGrid, dimBlock>>>(dF2);
    //		hipDeviceSynchronize();
    Equate<<<dimGrid, dimBlock>>>(dF1, dF2);
    if (fmod(t, tIo) == 0.0f) {
      // IO function
      f2 = "sol" + std::to_string(kk) + ".dat";
      hipMemcpy(f.data(), dF2, sz, hipMemcpyDeviceToHost);
      IoFun(f2, x, f);
      kk++;
    }

    t += dt;
  }

  if (fmod(tMax, tIo) != 0.0f) { // IO Function
    f2 = "final_sol.dat";
    hipMemcpy(f.data(), dF1, sz, hipMemcpyDeviceToHost);
    IoFun(f2, x, f);
  }

  // deallocate memory
  hipFree(dF1);
  hipFree(dF2);
}
