#pragma once

#include <string>
#include <vector>
#include "solution.h"

enum SolutionViolationFlags {
    SOLUTION_OK = 0,
    SOLUTION_BAD_DEPOT = 1 << 0,
    SOLUTION_BAD_NODE = 1 << 1,
    SOLUTION_CAPACITY = 1 << 2,
    SOLUTION_TIME_WINDOW = 1 << 3,
    SOLUTION_DUPLICATE_CUSTOMER = 1 << 4,
    SOLUTION_MISSING_CUSTOMER = 1 << 5,
};

struct SolutionEval {
    bool feasible = false;
    double fitness = 0.0;
    double distance = 0.0;
    int vehicle_count = 0;
    int violation_flags = SOLUTION_OK;
};

struct EncodedPopulation {
    int solution_count = 0;
    int route_count = 0;
    int customer_count = 0;
    int depot = 0;
    int max_route_len = 0;
    std::vector<int> routes_flat;
    std::vector<int> route_offsets;
    std::vector<int> route_lengths;
    std::vector<int> solution_offsets;
    std::vector<int> solution_route_counts;
    std::vector<bool> dirty_flags;
};

EncodedPopulation encode_population_for_gpu(const std::vector<Solution>& population, const Data& data, const std::vector<bool>& pop_dirty = std::vector<bool>());
EncodedPopulation encode_solution_for_gpu(const Solution& solution, const Data& data);
Solution decode_solution_from_gpu(const EncodedPopulation& encoded, int solution_id, const Data& data);

SolutionEval evaluate_encoded_solution_cpu(const EncodedPopulation& encoded, int solution_id, const Data& data);
std::vector<SolutionEval> evaluate_encoded_population_cpu(const EncodedPopulation& encoded, const Data& data);

bool validate_encoded_solution(
    const Solution& solution,
    const EncodedPopulation& encoded,
    int solution_id,
    const Data& data,
    std::string* error_message = nullptr
);

