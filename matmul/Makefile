C = nvcc
NVCCFLAGS =  -arch=sm_70
CFLAGS = -std=c++11

all: matmul 

matmul: matmul.cu mat.o 
	$(C) $(NVCCFLAGS) $(CFLAGS) -rdc=true -o matmul.exe matmul.cu mat.o
 
mat.o: mat.cu
	$(C) $(NVCCFLAGS) $(CFLAGS) -dc mat.cu

clean:
	rm -f matmul.exe *.dat *.o


