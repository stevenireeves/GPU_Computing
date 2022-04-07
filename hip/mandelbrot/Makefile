CPP = hipcc
CFLAGS = -lX11 -lpthread -std=c++11
all: mandelbrot

mandelbrot: main.cpp 
	$(CPP) $(CFLAGS) -o mandelbrot main.cpp 

clean:
	rm -f mandelbrot output.bmp *.o
