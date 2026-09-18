#pragma once

#include <string>

#include "data.h"
#include "solution.h"

#ifdef USE_CUDA
bool run_full_gpu_solver(Data& data, Solution& best_solution, std::string& error_message);
#endif
