C = hipcc
CFLAGS = -std=c++11

all: spmat 
spmat: matcompare.cpp
	$(C) $(CFLAGS) -o mat.exe matcompare.cpp

clean:
	rm -f mat.exe *.o *.err *.out
