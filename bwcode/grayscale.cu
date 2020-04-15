//#include <cuda_runtime.h>
#include "CImg.h"

using namespace cimg_library;


/* Kernel to generate gray-scale image from RGB image 
   :inputs: Pointer to RGB values 'd_src', width and height of image
   :output: Grayscale image 'd_dst'
*/ 
__global__ void rgb2gray(const unsigned char d_src[],
                         unsigned char d_dst[],
                         int width, int height)
{
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos_x >= width || pos_y >= height)
        return;
	
    /* get local version for easier typing */
    /* memory is also strided in CImg on channel */  
    uchar3 rgb; /* uchar3 is a struct with 3 unsigned characters */ 
    rgb.x = d_src[pos_y * width + pos_x];
    rgb.y = d_src[(height + pos_y ) * width + pos_x];
    rgb.z = d_src[(height * 2 + pos_y) * width + pos_x];

    /* cast as int */ 
    unsigned int gray = (unsigned int)(0.299f*rgb.x + 0.587f*rgb.y + 0.114*rgb.z);
    /* convert to unsinged int -- if intesity is greater than 255 set to 255 */ 
    d_dst[pos_y * width + pos_x] = gray > 255 ? 255 : gray;
}


int main()
{
    //load image using the CImg class constructor
    //This constructor requires a string to find an image stored on disc
    CImg<unsigned char> src("SAGAN.bmp");
    
    //retrieve width and height from the CImg class 
    int width = src.width();
    int height = src.height();

    //also get size = width*height*3*sizeof(uchar)
    size_t size = src.size();
    size_t gsize = width*height*sizeof(unsigned char);

    //Generate gray-scale image object
    //This constructor requires width, height, number of channels, and number of images
    CImg<unsigned char> gs(width, height, 1, 1);


    /* Generate device memory */ 
    unsigned char *d_src;
    unsigned char *d_gs;
    cudaMalloc((void**)&d_src, size);
    cudaMalloc((void**)&d_gs, gsize);

    /* copy RGB onto device */ 
    cudaMemcpy(d_src, src.data(), size, cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 blkDim (16, 16, 1);
    dim3 grdDim ((width + 15)/16, (height + 15)/16, 1);
    rgb2gray<<<grdDim, blkDim>>>(d_src, d_gs, width, height);

    //copy back Gray-scale to CPU
    cudaMemcpy(gs.data(), d_gs, gsize, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_gs);

	gs.save("GSSAGAN.bmp");
    return 0;
}
