# the compiler: gcc for C program, define as g++ for C++

# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings

# The build target
all: tsp tsp.o element.o

tsp.o: tsp.cpp tsp.h
	g++ -c -O3 tsp.cpp

element.o: element.cpp element.h
	g++ -c -O3 element.cpp

tsp: tsp.o element.o
	g++ -g -Wall -O3 -o tsp -fopenmp tsp.o element.o

clean:
	-rm -f tsp *.o