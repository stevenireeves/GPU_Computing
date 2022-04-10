C = hipcc
CFLAGS = -lX11 -lpthread

all: grayscale

grayscale: grayscale.cpp  
	$(C) $(CFLAGS) -o grayscale.exe grayscale.cpp 
clean:
	rm -f grayscale.exe GSSAGAN.bmp *.o

