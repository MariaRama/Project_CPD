#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <omp.h>
#include <mpi.h>
#include <limits>

using namespace std;

#include "queue.hpp"
#include "element.h"

void parse_inputs(int argc, char *argv[]);
void print_result(vector <int> BestTour, double BestTourCost);
vector<pair<double,double>> get_mins();
double initialLB(vector<pair<double,double>> &mins);
double calculateLB(vector<pair<double,double>> &mins, int f, int t, double LB);
void create_children(QueueElem &myElem, PriorityQueue<QueueElem> &myQueue, vector<pair<double,double>> &mins);
void TSPBB(PriorityQueue<QueueElem> &myQueue, vector<int> &BestTour, vector<pair<double,double>> &mins);
pair<vector <int>, double> tsp(int start, int end, int rank);

// global variables
double BestTourCost;
int numCities, numRoads;
vector <vector <double>> distances;
