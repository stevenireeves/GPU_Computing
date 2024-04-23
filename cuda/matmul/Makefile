C = nvcc
CFLAGS = -std=c++11 -Xcompiler -fopenmp

all: matmul 

matmul: matmul.o mat.o 
	$(C) $(CFLAGS) -o matmul.exe matmul.o mat.o 
matmul.o: matmul.cu
	$(C) $(CFLAGS) -c matmul.cu -o matmul.o
mat.o: mat.cu
	$(C) $(CFLAGS) -c mat.cu -o mat.o

clean:
	rm -f matmul.exe *.dat *.o


