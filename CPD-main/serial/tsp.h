#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "queue.h"
#include "element.h"

void parse_inputs(int argc, char *argv[]);
int initialLB();
int calculateLB(int f, int t, int LB);
char compare_values(void *a, void *b);
void tsp();
void print_result();

// Global variables
int maxValue;
int numCities, numRoads;
int **distances;
