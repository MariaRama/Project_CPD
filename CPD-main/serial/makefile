# the compiler: gcc for C program, define as g++ for C++

# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings

# The build target
all: queue.o element.o tsp

queue.o: queue.c queue.h
	gcc -c queue.c

element.o: element.c element.h
	gcc -c element.c element.h

tsp: queue.o element.o tsp.c tsp.h
	gcc -g -Wall -o tsp queue.o element.o tsp.c tsp.h

clean:
	-rm -f tsp *.o