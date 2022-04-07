C = nvcc
NVCCFLAGS =  -arch=sm_60 
CFLAGS = -lX11 -lpthread -std=c++11

all: color 

color: color_hist.cu  
	$(C) $(NVCCFLAGS) $(CFLAGS) -o color.exe color_hist.cu 
clean:
	rm -f color.exe histo.dat *.o

