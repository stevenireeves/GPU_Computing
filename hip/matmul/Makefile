C = hipcc
CFLAGS = -std=c++11

all: matmul 

matmul: matmul.o mat.o 
	$(C) $(CFLAGS) -o matmul.exe matmul.o mat.o 
matmul.o: matmul.cpp
	$(C) $(CFLAGS) -c matmul.cpp -o matmul.o
mat.o: mat.cpp
	$(C) $(CFLAGS) -c mat.cpp -o mat.o

clean:
	rm -f matmul.exe *.dat *.o


