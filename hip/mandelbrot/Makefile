CPP = hipcc
CFLAGS = -lX11 -lpthread -std=c++11 -I /home/steven/CImg/
all: mandelbrot

# Change /home/steven/CImg to your PathToCImg
mandelbrot: main.cpp 
	$(CPP) $(CFLAGS) -o mandelbrot main.cpp 

clean:
	rm -f mandelbrot output.bmp *.o
