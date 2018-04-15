C = nvcc
# CFLAGS = -deviceemu

all: mandelbrot

mandelbrot: main.cu bmp.o
	$(C) $(CFLAGS) -o mandelbrot main.cu bmp.o

bmp.o: bmp.c bmp.h
	$(C) $(CFLAGS) -c bmp.c

clean:
	rm -f mandelbrot output.bmp *.o
