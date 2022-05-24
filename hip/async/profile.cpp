#include <stdlib.h>
#include <hip/hip_runtime.h>

int main()
{
        const unsigned int N = 1048576; 
        const unsigned int bytes = N*sizeof(int);
        int *h_a = (int*)malloc(bytes);
        int *d_a; 
        hipMalloc((int**)&d_a, bytes);

        memset(h_a, 0, bytes); 
        hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
        hipMemcpy(h_a, d_a, bytes, hipMemcpyDeviceToHost);

        return 0;
}

