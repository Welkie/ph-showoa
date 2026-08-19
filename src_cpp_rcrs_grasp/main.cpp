#include <iostream>
#include <limits>
#include "data.h"
#include "solution.h"
#include "search_framework.h"

int main(int argc, char* argv[]) {
    std::cout << "PH-SHOWOA C++ VRPSDPTW Solver Initializing..." << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem_filepath> [options]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../data/Wang_Chen/explicit_rcdp1001.vrpsdptw --compute_backend cuda" << std::endl;
        return 1;
    }

    std::string problem_file = argv[1];
    Data data;
    if (!data.load_problem(problem_file)) {
        std::cerr << "Failed to load problem file." << std::endl;
        return 1;
    }

    data.parse_args(argc, argv);

    // Run the PH-SHOWOA metaheuristic framework
    Solution best_s;
    best_s.cost = std::numeric_limits<double>::infinity();

    search_framework(data, best_s);

    std::cout << "PH-SHOWOA C++ solver run completed." << std::endl;
    return 0;
}
