C = nvcc
NVCCFLAGS =  -arch=sm_60
CFLAGS = -std=c++11

all: nbody 

nbody: nbody.cu
	$(C) $(NVCCFLAGS) $(CFLAGS) -o nbody.exe nbody.cu 
clean:
	rm -f nbody.exe *.dat *.o


