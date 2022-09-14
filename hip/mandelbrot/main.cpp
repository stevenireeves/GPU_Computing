// Note: Most of the code comes from the MacResearch OpenCL podcast

#include <assert.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "CImg.h"

/* Mandlebrot rendering function
   :inputs: width and height of domain, max_iterations
   :ouputs: 8biti unsigned character array containing mandlebrot image
*/
__global__ void Render(unsigned char out[], const int width, const int height,
                       const int maxIter) {

  // indexing for mandlebrot set, span domain for escape algo
  int xDim = blockIdx.x * blockDim.x + threadIdx.x;
  int yDim = blockIdx.y * blockDim.y + threadIdx.y;
  // flatten the index.
  int index = width * yDim + xDim;

  if (index >= width * height)
    return;

  float xOrigin = ((float)xDim / width) * 3.25 - 2;   // "Real(C)"  C_x
  float yOrigin = ((float)yDim / width) * 2.5 - 1.25; // "Imaginary(C)" C_y

  float x = 0.0;
  float y = 0.0;

  int iteration = 0;
  // escape algorithm
  // Every thread will loop in this at most maxIter
  while (x * x + y * y <= 4 && iteration < maxIter) {
    float xtemp = x * x - y * y + xOrigin;
    y = 2 * x * y + yOrigin;
    x = xtemp;
    iteration++;
  }

  if (iteration == maxIter) {
    out[index] = 0;
  } else {
    out[index] = iteration;
  }
}
/*
        Conditional loop in the middle -> threads will finish at different times
        Thread divergence the program as fast as the slowest thread.
*/

/* Host function for generating the mandlebrot image
   :inputs: width and height of domain, and maxIterations for escape
   :outputs: none
   writes a bmp image to disc
*/
void Mandelbrot(const int width, const int height, const int maxIter) {
  using uchar = unsigned char;

  size_t bufferSize = sizeof(uchar) * width * height;

  uchar *image;
  hipMalloc(&image, bufferSize);

  std::vector<uchar> hostImage(width * height, 0);

  dim3 blockDim(16, 16, 1);                                 // 16*16 threads
  dim3 gridDim(width / blockDim.x, height / blockDim.y, 1); // Rest of the image

  /* dim3  int x, int y, int z */

  /*kernel<< gridDim, blockDim, #bytessharedmem>>> */
  Render<<<gridDim, blockDim>>>(image, width, height, maxIter);

  /*after render is done */
  hipMemcpy(hostImage.data(), image, bufferSize, hipMemcpyDeviceToHost);

  // Now write the file
  cimg_library::CImg<unsigned char> img2(hostImage.data(), width, height);
  img2.save("output.bmp");

  hipFree(image);
}

/*main function */
int main() {
  Mandelbrot(7680, 7680, 256);
  return 0;
}
