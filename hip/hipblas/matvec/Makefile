C = hipcc
CFLAGS = -std=c++14 -lhipblas
LDFLAGS = -L/opt/rocm-5.1.0/hipblas/lib

all: matvec

matvec: matvec.cpp
	$(C) $(LDFLAGS) $(CFLAGS) -o matvec.exe matvec.cpp

clean:
	rm -f matvec.exe
