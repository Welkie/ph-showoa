#include <iostream>
#include <limits>
#include "data.h"
#include "solution.h"
#include "search_framework.h"
#include "full_gpu_solver.h"

int main(int argc, char* argv[]) {
    std::cout << "PH-SHOWOA C++ VRPSDPTW Solver Initializing..." << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem_filepath> [options]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../data/Wang_Chen/explicit_rcdp1001.vrpsdptw --compute_backend cuda" << std::endl;
        return 1;
    }

    std::string problem_file = argv[1];
    std::cout << "Loading problem file: " << problem_file << std::endl;
    Data data;
    if (!data.load_problem(problem_file)) {
        std::cerr << "Failed to load problem file." << std::endl;
        return 1;
    }

    std::cout << "Done load_problem" << std::endl;
    data.parse_args(argc, argv);
    std::cout << "Done parse_args" << std::endl;

    // Run the PH-SHOWOA metaheuristic framework
    Solution best_s;
    best_s.cost = std::numeric_limits<double>::infinity();

    if (data.architecture == "full_gpu") {
#ifdef USE_CUDA
        std::string error_message;
        if (!run_full_gpu_solver(data, best_s, error_message)) {
            std::cerr << "Full-GPU solver failed: " << error_message << std::endl;
            return 2;
        }
#else
        std::cerr << "Architecture 'full_gpu' requires a CUDA-enabled build." << std::endl;
        return 2;
#endif
    } else {
        search_framework(data, best_s);
    }

    std::cout << "PH-SHOWOA C++ solver run completed." << std::endl;
    return 0;
}
