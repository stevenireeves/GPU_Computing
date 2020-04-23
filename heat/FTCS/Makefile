C = nvcc
NVCCFLAGS = -arch=sm_60 
CFLAGS = -std=c++11

all: heat 

heat: heat.cu  
	$(C) $(NVCCFLAGS) $(CFLAGS) -o heat.exe heat.cu 
clean:
	rm -f heat.exe *.dat *.o

