#include <iostream>
#include <fstream>
#include <string>
#include <hip/hip_runtime.h>
#include <vector>
#include "CImg.h"

#define NUM_BINS 256


using namespace std;
using namespace cimg_library;

/* Kernel, Shared Memory based histogram
   Inputs: unsigned char array in, int width, int height, unsigned int array out
   Output: unsigned int array out
   out contains the color histograms from the image in.
*/
__global__ void HistogramSmemAtomics(const unsigned char *in, int width, int height, unsigned int *out)
{
    // pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // linear thread index within 2D block
    int t = threadIdx.x + threadIdx.y * blockDim.x; 
    
    // initialize temporary accumulation array in shared memory
    __shared__ uint3 smem[NUM_BINS];
    smem[t].x = 0;
    smem[t].y = 0;
    smem[t].z = 0;
    __syncthreads();
    
    
    // process pixels
    // updates our block's partial histogram in shared memory
    uint3 rgb;
    rgb.x = (unsigned int)(in[y * width + x]); //Numbers between 0 and 255
    rgb.y = (unsigned int)(in[(y + height) * width + x]);
    rgb.z = (unsigned int)(in[(y + height*2) * width + x]);
    atomicAdd(&smem[rgb.x].x, 1);
    atomicAdd(&smem[rgb.y].y, 1);
    atomicAdd(&smem[rgb.z].z, 1);
    __syncthreads();
    
    // write partial histogram into the global memory
    //  out += g * NUM_BINS;
    atomicAdd(&out[t], smem[t].x);
    atomicAdd(&out[t + NUM_BINS * 1], smem[t].y);
    atomicAdd(&out[t + NUM_BINS * 2], smem[t].z);
}


/* Helper function to write out histogram to filesystem. */
void io_fun(std::string file, std::vector<unsigned int> histo)
{
    std::ofstream myfile_tsN;
    myfile_tsN.open(file);
    for(int i = 0; i < NUM_BINS; i++)
    {
        myfile_tsN << histo[i]  << '\t'<< histo[i + NUM_BINS] << '\t'<< histo[i + 2*NUM_BINS] << std::endl; // R G B 
    }
    myfile_tsN.close();
}


/* 
    Driver function that loads in the image "window.bmp" and launches HIP kernels to compute the color histograms
    derived from that image. Uses the CImg library to data handling. 
*/
int main()
{
    using uint = unsigned int;
    //load image
    CImg<unsigned char> src("window.bmp");
    int width = src.width();
    int height = src.height();
    size_t size = src.size()*sizeof(unsigned char); //note sizeof(uchar) = 1, but src.size() != numBytes in general
    std::vector<uint> hHisto(NUM_BINS*3, 0);

    unsigned char *dSrc;
    unsigned int  *dHisto;

    hipMalloc((void**)&dSrc, size);
    hipMalloc((void**)&dHisto, NUM_BINS*3*sizeof(unsigned int));

    hipMemcpy(dSrc, src.data(), size, hipMemcpyHostToDevice);

    //launch the kernel
    dim3 blkDim (16, 16, 1);
    dim3 grdDim ((width + 15)/16, (height + 15)/16, 1);
    HistogramSmemAtomics<<<grdDim, blkDim>>>(dSrc, width, height, dHisto);
    //copy back the result to CPU
    hipMemcpy(hHisto.data(), dHisto, NUM_BINS*3*sizeof(unsigned int), hipMemcpyDeviceToHost);

    io_fun("histo.dat", hHisto);

    hipFree(dSrc);
    hipFree(dHisto);

    return 0;
}

