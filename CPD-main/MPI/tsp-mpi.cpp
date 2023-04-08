#include "tsp-mpi.h"

int main(int argc, char *argv[]) {
    double exec_time;

    int num_processes, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0) {
        parse_inputs(argc, argv);
    }

    //MPI_Bcast sends the message from the root process to all other processes
    MPI_Bcast(&numCities, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numRoads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&BestTourCost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // divide work among processes
    int start = rank * numCities / num_processes;
    int end = (rank + 1) * numCities / num_processes;

    // calculate tsp
    double start_time = MPI_Wtime();
    pair<vector<int>, double> results = tsp(start, end);
    double end_time = MPI_Wtime();

    // gather results
    //MPI_Gather collects data from all processes in the communicator comm, and sends it to the root process
    vector<pair<vector<int>, double>> all_results(num_processes);
    MPI_Gather(&results, sizeof(pair<vector<int>, double>), MPI_BYTE,
               &all_results[0], sizeof(pair<vector<int>, double>), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if(rank == 0) {
        exec_time = end_time - start_time;
        cout << "Execution time: " << exec_time << endl;

        // find best result
        pair<vector<int>, double> best_result = all_results[0];
        for(int i=1; i<num_processes; i++) {
            if(all_results[i].second < best_result.second) {
                best_result = all_results[i];
            }
        }

        print_result(best_result.first, best_result.second);
    }

    MPI_Finalize();
    return 0;
}

void parse_inputs(int argc, char *argv[]) {
    string line;
    ifstream myfile (argv[1]);

    int row, col;
    double val;

    if(argc-1 != 2)
        exit(-1);

    if (myfile.is_open()){
        getline(myfile, line);
        sscanf(line.c_str(), "%d %d", &numCities, &numRoads);
    }else
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
        exit(-1);

    BestTourCost = atof(argv[2]);
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

pair<vector <int>, double> tsp() {
    vector <int> BestTour = {0};
    
    return make_pair(BestTour, BestTourCost);
}
