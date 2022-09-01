#include <cmath>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

__device__ float Pi = 3.1415926;

/* Device function to calculate a Gaussian probability
   Inputs: FP32 x, FP32 mu, FP32 sig
   position, mean, and standard deviation,
   Output: FP32 probability of x given mu and sig.
*/
__device__ float Normal(float x, float mu, float sig) {
  float temp1 = 1.0f / (std::sqrt(2.0f * Pi) * sig);
  float temp2 = 0.5f * (x - mu) * (x - mu) / (sig * sig);
  float val = temp1 * std::exp(-temp2);
  return val;
}

/* Kernel, generated the Guassian PDF given mean and standard deviation
   Inputs: FP32 array data, FP32 array x, FP32 xbeg, FP32 dx, FP32 mu, FP32 sig,
   int N Output: FP32 array data, contains the PDF given the input args
*/
__global__ void DataGen(float *data, float *x, float xBeg, float deltaX,
                        float mu, float sig, int n) {
  // generate x and data
  // use x for plotting later
  int tId = threadIdx.x + blockDim.x * blockIdx.x;
  if (tId < 1 || tId > n) {
    return;
  }
  float xTemp[2];
  xTemp[0] = xBeg + tId * deltaX;
  xTemp[1] = xBeg + (tId - 1) * deltaX;
  data[tId - 1] =
      (Normal(xTemp[0], mu, sig) + Normal(xTemp[1], mu, sig)) * deltaX * 0.5f;
  x[tId - 1] = xTemp[0];
}

/* Kernel, performs the Bleloch sum scan algorithm on idata.
   Inputs: FP32 array odata, FP32 array idata, Int n
   Output: FP32 array odata
   odata contains the scanned output of idata.
*/
__global__ void BlScan(float *oData, const float *iData, int n) {
  extern __shared__ float temp[]; // allocated on invocation
  int tId = threadIdx.x;
  int offset = 1;
  temp[2 * tId] = iData[2 * tId]; // load input into shared memory
  temp[2 * tId + 1] = iData[2 * tId + 1];
  for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (tId < d) {
      int aI = offset * (2 * tId + 1) - 1;
      int bI = offset * (2 * tId + 2) - 1;
      temp[bI] += temp[aI];
    }
    offset *= 2;
  }
  if (tId == 0) {
    temp[n - 1] = 0;
  }                              // clear the last element
  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (tId < d) {
      int aI = offset * (2 * tId + 1) - 1;
      int bI = offset * (2 * tId + 2) - 1;
      float t = temp[aI];
      temp[aI] = temp[bI];
      temp[bI] += t;
    }
  }
  __syncthreads();
  oData[2 * tId] = temp[2 * tId]; // write results to device memory
  oData[2 * tId + 1] = temp[2 * tId + 1];
}

/* Kernel, translates an exclusive scan to an inclusive scan
   Inputs: FP32 array scan, FP32 array idata, Int n
   Output: FP32 array scan
*/
__global__ void Ex2In(float *scan, float *iData, int n) {
  extern __shared__ float temp[]; // allocated via kernel config
  int tId = threadIdx.x;
  if (tId >= n)
    return;
  temp[tId] = scan[tId]; // load scan data;
  __syncthreads();

  if (tId > 0)
    scan[tId - 1] = temp[tId];

  if (tId == n - 1)
    scan[tId] = temp[tId] + iData[tId]; // last element clean up!
}

/* Host function that drives the computation of the Gaussian CDF given mean and
   standard deviation Inputs: FP32 array cdf, FP32 array pdf, FP32 array x, FP32
   scalar mu, FP32 scalar simga, Int num_data Outputs: FP32 array cdf
*/
void ComputeNormCdf(std::vector<float> &cdf, std::vector<float> &hPdf,
                    std::vector<float> &x, float mu, float sigma,
                    const int numData) {
  // data and parameter set up!
  float *pdf, *dCdf, *dX;
  float xBeg = mu - 5 * sigma; // Five standard deviations from the mean!
  float xEnd = mu + 5 * sigma;
  float deltaX = (xEnd - xBeg) / (float)numData;
  // data allocation!
  size_t size = numData * sizeof(float);
  hipMalloc(&pdf, size);
  hipMalloc(&dX, size);
  hipMalloc(&dCdf, size);
  // data generation kernel!
  DataGen<<<2, numData / 2 + 1>>>(pdf, dX, xBeg, deltaX, mu, sigma, numData);

  // Perform scan for cdf calculation!
  BlScan<<<1, numData / 2, 2 * size>>>(dCdf, pdf, numData);

  // Shift Correct for CDF
  Ex2In<<<1, numData, size>>>(dCdf, pdf, numData);

  // Transfer to Host!
  hipMemcpy(cdf.data(), dCdf, size, hipMemcpyDeviceToHost);
  hipMemcpy(hPdf.data(), pdf, size, hipMemcpyDeviceToHost);
  hipMemcpy(x.data(), dX, size, hipMemcpyDeviceToHost);

  // Free memory!
  hipFree(pdf);
  hipFree(dCdf);
  hipFree(dX);
}

/* Driver function to compute and output cdf data */
int main() {
  int numData = 1024;

  // Memory allocation
  std::vector<float> cdf(numData, 0.f);
  std::vector<float> x(numData, 0.f);
  std::vector<float> hPdf(numData, 0.f);

  // call function
  ComputeNormCdf(cdf, hPdf, x, 0.0f, 1.0f, numData);

  // create file object to output data
  std::ofstream myFile;
  myFile.open("cdf.dat");
  for (int aa = 0; aa < numData; aa++) {
    myFile << x[aa] << '\t' << hPdf[aa] << '\t' << cdf[aa] << "\n";
  }
  myFile << std::endl;

  // destroy file object
  myFile.close();
}
