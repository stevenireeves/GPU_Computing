C = hipcc
CFLAGS = -std=c++11

all: matmul 

matmul: matmul.o mat.o 
	$(C) $(CFLAGS) -o matmul.exe matmul.o mat.o 
matmul.o: matmul.cpp
	$(C) $(CFLAGS) -dc matmul.cpp
mat.o: mat.cpp
	$(C) $(CFLAGS) -dc mat.cpp
clean:
	rm -f matmul.exe *.dat *.o


