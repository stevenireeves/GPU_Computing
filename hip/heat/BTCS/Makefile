C = hipcc
CFLAGS = -std=c++11

all: heat 

heat: heat.cpp  
	$(C) $(CFLAGS) -o heat.exe heat.cpp 
clean:
	rm -f heat.exe *.dat *.o

