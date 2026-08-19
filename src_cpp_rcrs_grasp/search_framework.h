#pragma once

#include <vector>
#include <random>
#include "data.h"
#include "solution.h"
#include "compute_backend.h"

struct AgentUpdateTask {
    int index;
    Solution current;
    Solution peer;
    Solution best;
    double current_fit;
    double p_hybrid;
    double a;
    int iteration;
    int max_iter;
    int seed;
};

void search_framework(Data& data, Solution& best_s);
