# the compiler: gcc for C program, define as g++ for C++

# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings

# The build target
all: tsp-mpi tsp-mpi.o

tsp-mpi.o: tsp-mpi.cpp tsp-mpi.h
	g++ -c -O3 -fopenmp tsp-mpi.cpp

tsp-mpi: tsp-mpi.o
	g++ -g -Wall -O3 -o tsp-mpi -fopenmp tsp-mpi.o

clean:
	-rm -f tsp-mpi *.o
