C = nvcc
CPP = g++
NVCCFLAGS =  -arch=sm_60 
CFLAGS = -lX11 -lpthread

all: grayscale

grayscale: grayscale.cu  
	$(C) $(NVCCFLAGS) $(CFLAGS) -o grayscale.exe grayscale.cu 
clean:
	rm -f grayscale.exe GSSAGAN.bmp *.o

