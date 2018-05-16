C = nvcc
CFLAGS = -arch=sm_60 -std=c++11

all: spmat 
spmat: matcompare.cu
	$(C) $(CFLAGS) -o mat.exe matcompare.cu

clean:
	rm -f mat.exe *.o *.err *.out
