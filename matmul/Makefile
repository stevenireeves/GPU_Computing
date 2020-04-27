C = nvcc
NVCCFLAGS =  -arch=sm_60
CFLAGS = -std=c++11

all: matmul 

matmul: matmul.cu submat.o mat.o 
	$(C) $(NVCCFLAGS) $(CFLAGS) -rdc=true -o matmul.exe matmul.cu mat.o submat.o 
 
mat.o: mat.cu
	$(C) $(NVCCFLAGS) $(CFLAGS) -dc mat.cu

submat.o: submat.cu
	$(C) $(NVCCFLAGS) $(CFLAGS) -dc submat.cu

clean:
	rm -f matmul.exe *.dat *.o


