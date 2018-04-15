#include <cuda_runtime.h>
#include "CImg.h"

using namespace std;
using namespace cimg_library;

__global__ void rgb2gray(unsigned char * d_src, unsigned char * d_dst, int width, int height)
{
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos_x >= width || pos_y >= height)
        return;
	
    uchar3 rgb;
    rgb.x = d_src[pos_y * width + pos_x];
    rgb.y = d_src[(height + pos_y ) * width + pos_x];
    rgb.z = d_src[(height * 2 + pos_y) * width + pos_x];

    unsigned int _gray = (unsigned int)(0.299f*rgb.x + 0.587f*rgb.y + 0.114*rgb.z);
    unsigned char gray = _gray > 255 ? 255 : _gray;

    d_dst[pos_y * width + pos_x] = gray;
}


int main()
{
    //load image
    CImg<unsigned char> src("SAGAN.bmp");
    int width = src.width();
    int height = src.height();
    unsigned long size = src.size();

    //create pointer to image
    unsigned char *h_src = src.data();

    CImg<unsigned char> gs(width, height, 1, 1);
    unsigned char *h_gs = gs.data();

    unsigned char *d_src;
    unsigned char *d_gs;

    cudaMalloc((void**)&d_src, size);
    cudaMalloc((void**)&d_gs, width*height*sizeof(unsigned char));

    cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 blkDim (16, 16, 1);
    dim3 grdDim ((width + 15)/16, (height + 15)/16, 1);
    rgb2gray<<<grdDim, blkDim>>>(d_src, d_gs, width, height);

    //wait until kernel finishes
    cudaDeviceSynchronize();

    //copy back the result to CPU
    cudaMemcpy(h_gs, d_gs, width*height, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_gs);

    CImg<unsigned char> out(h_gs,width,height);
	out.save("GSSAGAN.bmp");
    return 0;
}
