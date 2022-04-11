C = hipcc
CFLAGS = -std=c++11

all: nbody 

nbody: nbody.cpp
	$(C) $(CFLAGS) -o nbody.exe nbody.cpp 
clean:
	rm -f nbody.exe *.dat *.o


