#include <iostream>
#include <fstream>
#include <cmath>
#include <hip/hip_runtime.h>

__device__ float PI=3.1415926;

/* Device function to calculate a Gaussian probability
   Inputs: FP32 x, FP32 mu, FP32 sig
   position, mean, and standard deviation,
   Output: FP32 probability of x given mu and sig. 
*/
__device__ float normal(float x, float mu, float sig)
{
    float temp1 = 1.0f/(sqrtf(2.0f*PI)*sig);
    float temp2 = 0.5f*(x-mu)*(x-mu)/(sig*sig);
    float val = temp1*expf(-temp2);
    return val;
}

/* Kernel, generated the Guassian PDF given mean and standard deviation 
   Inputs: FP32 array data, FP32 array x, FP32 xbeg, FP32 dx, FP32 mu, FP32 sig, int N
   Output: FP32 array data, contains the PDF given the input args
*/
__global__ void data_gen(float *data, float *x, float xbeg, float dx, float mu, float sig, int n)
{
    //generate x and data
    //use x for plotting later
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid<1 || tid > n)
    {
            return;
    }
    float xtemp[2];
    xtemp[0] = xbeg + tid*dx;
    xtemp[1] = xbeg + (tid-1)*dx;
    data[tid-1] = (normal(xtemp[0], mu, sig) + normal(xtemp[1], mu, sig))*dx*0.5f;
    x[tid-1] = xtemp[0];
}


/* Kernel, performs the Bleloch sum scan algorithm on idata.
   Inputs: FP32 array odata, FP32 array idata, Int n
   Output: FP32 array odata
   odata contains the scanned output of idata.
*/ 
__global__ void bl_scan(float *odata, const float *idata, int n)
{
    extern __shared__ float temp[];// allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    temp[2*thid] = idata[2*thid]; // load input into shared memory
    temp[2*thid+1] = idata[2*thid+1];
    for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d)
        {
        	int ai = offset*(2*thid+1)-1;
       	int bi = offset*(2*thid+2)-1;
       	temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (thid == 0)
    { temp[n - 1] = 0; } // clear the last element
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
       offset >>= 1;
       __syncthreads();
       if (thid < d)
       {
       	int ai = offset*(2*thid+1)-1;
       	int bi = offset*(2*thid+2)-1;
       	float t = temp[ai];
       	temp[ai] = temp[bi];
       	temp[bi] += t;
        }
    }
    __syncthreads();
    odata[2*thid] = temp[2*thid]; // write results to device memory
    odata[2*thid+1] = temp[2*thid+1]; 
}

/* Kernel, translates an exclusive scan to an inclusive scan
   Inputs: FP32 array scan, FP32 array idata, Int n
   Output: FP32 array scan
*/
__global__ void ex2in(float *scan, float *idata, int n)
{
    extern __shared__ float temp[]; //allocated via kernel config
	int thid = threadIdx.x;
    if(thid >= n)
       return;
    temp[thid] = scan[thid]; //load scan data;
    __syncthreads();

    if(thid > 0)
       scan[thid-1] = temp[thid];

    if(thid == n-1)
       scan[thid] = temp[thid] + idata[thid]; //last element clean up!
}

/* Host function that drives the computation of the Gaussian CDF given mean and standard deviation
   Inputs: FP32 array cdf, FP32 array pdf, FP32 array x, FP32 scalar mu, FP32 scalar simga, Int num_data
   Outputs: FP32 array cdf
*/
void compute_norm_cdf(float *cdf, float *h_pdf, float *x, float mu, float sigma, const int num_data)
{
    //data and parameter set up!
    float *pdf, *d_cdf, *d_x;
    float xbeg = mu - 5*sigma; //Five standard deviations from the mean!
    float xend = mu + 5*sigma;
    float dx = (xend - xbeg)/(float)num_data;
    
    //data allocation!
    size_t size = num_data*sizeof(float);
    hipMalloc((void**)&pdf, size);
    hipMalloc((void**)&d_x, size);
    hipMalloc((void**)&d_cdf, size);
    
    //data generation kernel!
    data_gen<<<2, num_data/2+1>>>(pdf, d_x, xbeg, dx, mu, sigma, num_data);
    hipDeviceSynchronize();
    
    //Perform scan for cdf calculation!
    bl_scan<<<1, num_data/2, 2*size>>>(d_cdf, pdf, num_data); 
    hipDeviceSynchronize();

	//Shift Correct for CDF 
	ex2in<<<1,num_data, size>>>(d_cdf, pdf, num_data);
	hipDeviceSynchronize();
    //Transfer to Host!
    hipMemcpy(cdf, d_cdf, size, hipMemcpyDeviceToHost);
    hipMemcpy(x, d_x, size, hipMemcpyDeviceToHost);
	hipMemcpy(h_pdf, pdf, size, hipMemcpyDeviceToHost);

    //Free memory!
    hipFree(pdf);
    hipFree(cdf);
    hipFree(d_x);
}


/* Driver function to compute and output cdf data */
int main()
{
    int num_data = 1024;
    size_t size = num_data*sizeof(float);

    //Memory allocation
    float *cdf, *x, *h_pdf;
    cdf = new float[size]; 
    x = new float[size];
	h_pdf = new float[size];

    //call function
    compute_norm_cdf(cdf, h_pdf, x, 0.0f, 1.0f, num_data);

    //create file object to output data
    std::ofstream myfile_tsN;
    myfile_tsN.open("cdf.dat");
    for(int aa = 0; aa<num_data; aa++){
       myfile_tsN << x[aa] << '\t' << h_pdf[aa] << '\t' << cdf[aa] << "\n";
    }
    myfile_tsN << std::endl;

    //destroy file object
    myfile_tsN.close();

    //free memory!
    delete x, cdf, h_pdf;
}

