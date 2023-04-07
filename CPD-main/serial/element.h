#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
	int *tour;
    int cost;
    int bound;
    int lenght;
    int node;
} queue_element;

queue_element *create_element();

void print_element(FILE* fp, void* element);