#include <stdio.h>
#include <hip/hip_runtime.h>

 //Query code, mostly borrowed from the internets. 
// Print device properties
void printDevProp(hipDeviceProp_t devProp)
{
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %zu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %zu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %zu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %zu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %zu\n",  devProp.textureAlignment);
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}
 
int main()
{
    int devCount;
    printf("HIP Device Query...\n");
    // Get device properties
    printf("\nHIP Device #%d\n", 0);
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    printDevProp(devProp);
 
    printf("\nPress any key to exit...");
    char c;
    scanf("%c", &c);
 
    return 0;
}
