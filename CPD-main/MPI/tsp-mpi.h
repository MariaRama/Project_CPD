#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <omp.h>

using namespace std;

#include "queue.hpp"

void parse_inputs(int argc, char *argv[]);
void print_result(vector <int> BestTour, double BestTourCost);
pair<vector <int>, double> tsp();

// global variables
double BestTourCost;
int numCities, numRoads;
vector <vector <double>> distances;