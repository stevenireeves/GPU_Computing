C = hipcc
CFLAGS = -std=c++14 -lhipblas
LDFLAGS = -L/opt/rocm-5.1.0/hipblas/lib

all: saxpy

saxpy: saxpy.cpp
	$(C) $(LDFLAGS) $(CFLAGS) -o saxpy.exe saxpy.cpp

clean:
	rm -f saxpy.exe
