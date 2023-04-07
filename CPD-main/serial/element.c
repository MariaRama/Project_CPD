#include "element.h"

queue_element *create_element(int numRoads, int LB) {
    queue_element *element;

    element = (queue_element*)malloc(sizeof(queue_element));

    element->tour = (int*)malloc(numRoads*sizeof(int));
    element->tour[0] = 0;
    element->cost = 0;
    element->bound = LB;
    element->lenght = 1;
    element->node = 0;

    return element;
}

void print_element(FILE* fp, void* el) {
    queue_element *element = (queue_element *) el;

    printf("Cost: %d\nTour:", element->cost);
    for(int i=0; i<element->lenght; i++) {
        printf(" %d", element->tour[i]);
    }
    printf("\n");
}