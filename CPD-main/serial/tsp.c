#include "tsp.h"

int main(int argc, char *argv[]) {
    // double exec_time;

    parse_inputs(argc, argv);

    // exec_time = -omp_get_wtime();

    tsp();

    // exec_time += omp_get_wtime();

    // fprintf(stderr, "%.1fs\n", exec_time);
    print_result(); // to the stdout!
}

void parse_inputs(int argc, char *argv[]) {
    FILE* fp;
    size_t  line_buffer_size=0;
    char *line_buffer = NULL;
    int row, col, val;

    // check number of arguments
    if(argc-1 != 2) {
        printf("There must be 2 inputs!\n");
        exit(-1);
    }

    // save maximum accepted value
    maxValue = atoi(argv[2]);
    // printf("Maximum accepted value: %d\n", maxValue);

    // open input file
    fp = fopen(argv[1], "r");
    if(fp == NULL) {
      printf("Error: unable to open input file");
      exit(1);
    }

    // read first line of input file
    getline(&line_buffer, &line_buffer_size, fp);
    sscanf(line_buffer, "%d %d", &numCities, &numRoads);
    // printf("Number of cities: %d\n", numCities);
    // printf("Number of roads: %d\n", numRoads);

    // allocate space for the distances
    distances = (int**)malloc(numCities*sizeof(int*));
    for(int i=0; i<numCities; i++) {
        distances[i] = (int*)malloc(numCities*sizeof(int));
    }

    // initialize distances matrix
    for(int i=0; i<numCities; i++) {
        for(int j=0; j<numCities; j++) {
            distances[i][j] = -1;
        }
    }

    // fill distances matrix
    for(int i=0; i<numRoads; i++) {
        getline(&line_buffer, &line_buffer_size, fp);
        sscanf(line_buffer, "%d %d %d", &row, &col, &val);
        // printf("%d %d %d\n", row, col, val);
        distances[row][col] = val;
        distances[col][row] = val;
    }

    // free allocated space for line buffer
    free(line_buffer);

    // for(int i=0; i<numCities; i++) {
    //     for(int j=0; j<numCities; j++) {
    //         printf("%d ", distances[i][j]);
    //     }
    //     printf("\n");
    // }

    // close input file
    fclose(fp);
}

int initialLB() {
    int min1, min2;
    int LB=0;

    for(int i=0; i<numCities; i++) {
        min1 = maxValue;
        min2 = maxValue;
        for (int j=0; j<numCities; j++) {
            if(distances[i][j] > 0) {
                if(distances[i][j] <= min1) {
                    min2 = min1;
                    min1 = distances[i][j];
                }else if(distances[i][j] <= min2) {
                    min2 = distances[i][j];
                }
            }
        }
        LB += (min1+min2)/2;
    }
    return LB;
}

int calculateLB(int f, int t, int LB) {
    int newLB;
    int cf, ct;
    int min1, min2;

    min1 = maxValue;
    min2 = maxValue;
    for (int j=0; j<numCities; j++) {
        if(distances[f][j] > 0) {
            if(distances[f][j] <= min1) {
                min2 = min1;
                min1 = distances[f][j];
            }else if(distances[f][j] <= min2) {
                min2 = distances[f][j];
            }
        }
    }
    if(distances[f][t] >= min2) {
        cf = min2;
    }else {
        cf = min1;
    }

    min1 = maxValue;
    min2 = maxValue;
    for (int j=0; j<numCities; j++) {
        if(distances[t][j] > 0) {
            if(distances[t][j] <= min1) {
                min2 = min1;
                min1 = distances[t][j];
            }else if(distances[t][j] <= min2) {
                min2 = distances[t][j];
            }
        }
    }
    if(distances[f][t] >= min2) {
        ct = min2;
    }else {
        ct = min1;
    }

    newLB = LB + distances[f][t] - (cf+ct)/2;

    return newLB;
}

char compare_values(void *a, void *b) {
    double num_a = *(double*)a;
    double num_b = *(double*)b;
    return num_a > num_b;
}

void tsp() {
    priority_queue_t *queue;
    queue_element *element;

    // initialize the queue elemente starting in city 0
    element = create_element(numRoads, initialLB());

    // initialize queue
    queue = queue_create(&compare_values);

    // push tour to queue
    queue_push(queue, &element);
    
    FILE* fp = stdout;
    queue_print(queue, fp, &print_element);
    fclose(fp);

    // void *node = queue_pop(queue);

    // printf("buffer: %s", node);

    // while(queue->size) {
        
    // }
    
    // free allocated space for distances
    for(int i=0; i<numCities; i++) {
        free(distances[i]);
    }
    free(distances);
}

void print_result() {

}