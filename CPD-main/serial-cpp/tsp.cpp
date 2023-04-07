#include "tsp.h"
#include <omp.h>

int main(int argc, char *argv[]) {
    double exec_time;

    parse_inputs(argc, argv);

    exec_time = -omp_get_wtime();

    pair<vector <int>, double> results = tsp();

    exec_time += omp_get_wtime();

    cout << "Serial execution time: " << exec_time << endl;

    print_result(results.first, results. second); // to the stdout!

    return 0;
}

void parse_inputs(int argc, char *argv[]) {
    string line;
    ifstream myfile (argv[1]);

    int row, col;
    double val;

    if(argc-1 != 2)
        // cout << "There must be 2 inputs!\n";
        exit(-1);

    if (myfile.is_open()){
        getline(myfile, line);
        sscanf(line.c_str(), "%d %d", &numCities, &numRoads);
    }else
        // cout << "Unable to open file"; 
        exit(-1);

    for(int i=0; i<numCities; i++) {
        vector <double> ones(numCities, -1.0);
        distances.push_back(ones);
    }
    
    if (myfile.is_open()){
        while (getline(myfile, line)) {
            sscanf(line.c_str(), "%d %d %lf", &row, &col, &val);
            distances[row][col] = val;
            distances[col][row] = val;
        }
        myfile.close();
    }else 
        // cout << "Unable to open file";
        exit(-1);

    BestTourCost = atof(argv[2]);
}

vector<pair<double,double>> get_mins() {
    vector<pair<double,double>> mins;
    mins.reserve(numCities);

    for (int i=0; i<numCities; i++) {
        double min1 = BestTourCost, min2 = BestTourCost;
        for (int j=0; j<numCities; j++) {
            double dist1 = distances[i][j];
            if(dist1 > 0) {
                if(dist1 <= min1) {
                    min2 = min1;
                    min1 = dist1;
                }else if(dist1 <= min2) {
                    min2 = dist1;
                }
            }
        }
        mins.push_back(make_pair(min1, min2));
    }
    return mins;
}

double initialLB(vector<pair<double,double>> &mins) {
    double LB=0;

    for(int i=0; i<numCities; i++) {
        LB += (mins[i].first + mins[i].second)/2;
    }
    return LB;
}

double calculateLB(vector<pair<double,double>> &mins, int f, int t, double LB) {
    double cf, ct;
    double directCost = distances[f][t];

    if(distances[f][t] <= 0)
        // cout << "ERROR";
        exit(-1);

    if(directCost >= mins[f].second) {
        cf = mins[f].second;
    }else
        cf = mins[f].first;

    if(directCost >= mins[t].second) {
        ct = mins[t].second;
    }else
        ct = mins[t].first;

    return LB + directCost - (cf+ct)/2;
}

void create_children(QueueElem &myElem, PriorityQueue<QueueElem> &myQueue, vector<pair<double,double>> &mins) {
    bool visitedCities[numCities] = {false};

    for (int city : myElem.tour) {
        visitedCities[city] = true;
    }

    for(int v=0; v<numCities; v++) {
        double dist = distances[myElem.node][v];
        if(dist>0 && !visitedCities[v]) {
            double newBound = calculateLB(mins, myElem.node, v, myElem.bound);                       
            if(newBound <= BestTourCost) {
                vector <int> newTour = myElem.tour;
                newTour.push_back(v);
                myQueue.push({newTour, myElem.cost + dist, newBound, myElem.length+1, v});
            }
        }
    }
}

pair<vector <int>, double> tsp() {
    PriorityQueue<QueueElem> myQueue;

    vector <int> BestTour = {0};
    BestTour.reserve(numCities+1);

    vector<pair<double,double>> mins = get_mins();

    myQueue.push({{0}, 0.0, initialLB(mins), 1, 0});
    
    while(myQueue.size() > 0){
        QueueElem myElem = myQueue.pop();

        if(myElem.bound >= BestTourCost)
            return make_pair(BestTour, BestTourCost);

        if(myElem.length == numCities) {
            double dist = distances[myElem.node][0];
            if(dist > 0) {
                if(myElem.cost + dist <= BestTourCost) {
                    BestTour = myElem.tour;
                    BestTour.push_back(0);
                    BestTourCost = myElem.cost + dist;
                }
            }
        }else 
            create_children(myElem, myQueue, mins);
    }
    return make_pair(BestTour, BestTourCost);
}

void print_result(vector <int> BestTour, double BestTourCost) {
    if(BestTour.size() != numCities+1) {
        cout << "NO SOLUTION" << endl;
    } else {
        cout.precision(1);
        cout << fixed << BestTourCost << endl;
        for(int i=0; i<numCities+1; i++) {
            cout << BestTour[i] << " ";
        }
        cout << endl;
    }
}