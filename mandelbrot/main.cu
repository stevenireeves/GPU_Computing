// Note: Most of the code comes from the MacResearch OpenCL podcast

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include <cuda.h>

extern "C" {
  #include "bmp.h"
}


/* Mandlebrot rendering function
   :inputs: width and height of domain, max_iterations
   :ouputs: 32bit character array containing mandlebrot image
*/
__global__ void render(char out[], const int width, const int height, const int max_iter) {

  // indexing for mandlebrot set, span domain for escape algo 
  int x_dim = blockIdx.x*blockDim.x + threadIdx.x;
  int y_dim = blockIdx.y*blockDim.y + threadIdx.y;
  // global index since we're creating a 3 channel image 
  int index = 3*(width*y_dim + x_dim);

  if(index >= 3*width*height) return; 

  float x_origin = ((float) x_dim/width)*3.25 - 2; // "Real(C)"  C_x
  float y_origin = ((float) y_dim/width)*2.5 - 1.25; // "Imaginary(C)" C_y  

  float x = 0.0;
  float y = 0.0;

  int iteration = 0;
  //escape algorithm
  // Everythread will loop in this at most max_iter 
  while(x*x + y*y <= 4 && iteration < max_iter) {
    float xtemp = x*x - y*y + x_origin;
    y = 2*x*y + y_origin;
    x = xtemp;
    iteration++;
  }

  if(iteration == max_iter) {
    out[index] = 0;
    out[index + 1] = 0;
    out[index + 2] = 0;
  } else {
    out[index] = iteration;
    out[index + 1] = iteration;
    out[index + 2] = iteration;
  }
}
/*
	Conditional loop in the middle -> threads will finish at different times
	Thread divergence the program as fast as the slowest thread.   
*/




/* Host function for generating the mandlebrot image
   :inputs: width and height of domain, and max_iterations for escape
   :outputs: none
   writes a bmp image to disc
*/
void mandelbrot(const int width, const int height, const int max_iter)
{
	// Multiply by 3 here, since we need red, green and blue for each pixel
  size_t buffer_size = sizeof(char) * width * height * 3;

  char *image; 
  cudaMalloc((void **) &image, buffer_size);

  char *host_image; 
  host_image = new char[width*height*3]; 

  dim3 block_Dim(16, 16, 1); // 16*16 threads 
  dim3 grid_Dim(width / block_Dim.x, height / block_Dim.y, 1); //Rest of the image

  /* dim3  int x, int y, int z */

  /*kernel<< grid_Dim, block_Dim, #bytes_shared_mem>>> */ 
  render<<< grid_Dim, block_Dim >>>(image, width, height, max_iter);

  /*after render is done */ 
  cudaMemcpy(host_image, image, buffer_size, cudaMemcpyDeviceToHost);

  // Now write the file
  write_bmp("output_2.bmp", width, height, host_image);

  cudaFree(image);
  delete host_image;
}


/*main function */ 
int main() 
{
  mandelbrot(7680, 7680, 256);
  return 0;
}
