C = nvcc
CPP = g++
NVCCFLAGS =  -arch=sm_60 
CFLAGS = -lX11 -lpthread

all: grayscale

grayscale: main.cu bmp.o 
	$(C) $(NVCCFLAGS) -o grayscale.exe main.cu bmp.o

bmp.o: bmp.cpp bmp.hpp
	$(CPP) $(CFLAGS) -c bmp.cpp 

clean:
	rm -f grayscale.exe GSSAGAN.bmp *.o

