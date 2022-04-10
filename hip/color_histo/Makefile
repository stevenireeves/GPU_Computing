C = hipcc
CFLAGS = -lX11 -lpthread -std=c++11

all: color 

color: color_hist.cpp 
	$(C) $(CFLAGS) -o color.exe color_hist.cpp 
clean:
	rm -f color.exe histo.dat *.o

