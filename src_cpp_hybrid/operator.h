#pragma once

#include <vector>
#include <random>
#include "data.h"
#include "solution.h"
#include "move.h"
#include "compute_backend.h"
#include "deterministic_rng.h"

struct ConstructionConfig {
    int ksize = 0;
    std::string insertion_mode = "rcrs";
    double lambda = 0.0;
    double gamma = 0.0;
};

// Helper definitions
bool _chk_route_list(const std::vector<int>& nl, const Data& data, BaseComputeBackend* backend);
std::vector<RouteEval> evaluate_route_batch(const std::vector<std::vector<int>>& routes, const Data& data, BaseComputeBackend* backend);

bool check_capacity(const Attr& a, const Attr& b, const Data& data);
bool check_tw(const Attr& a, const Attr& b, const Data& data);

bool eval_route(const Solution& s, const Seq* seq_list, int seq_list_len, Attr& tmp_attr, const Data& data);
bool eval_move(const Solution& s, Move& m, const Data& data, BaseComputeBackend* backend);

void apply_move(Solution& s, const Move& m, const Data& data);
void find_local_optima(Solution& s, const Data& data, BaseComputeBackend* backend);
void do_local_search(Solution& s, const Data& data, BaseComputeBackend* backend);

void new_route_insertion(Solution& s, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng, int initial_node = -1);
void new_route_insertion(
    Solution& s,
    const Data& data,
    const ConstructionConfig& config,
    BaseComputeBackend* backend,
    LegacyMt19937& rng,
    int initial_node = -1
);
void new_route_insertion_batched(
    std::vector<Solution>& solutions,
    const Data& data,
    const std::vector<ConstructionConfig>& configs,
    BaseComputeBackend* backend,
    std::vector<LegacyMt19937>& rngs
);

void related_removal(Solution& s, const Data& data, LegacyMt19937& rng);
void random_removal(Solution& s, const Data& data, LegacyMt19937& rng);
void greedy_insertion(Solution& s, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng);
void regret_insertion(Solution& s, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng);

void perturb(std::vector<Solution>& s_vector, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng);

// Paper Heuristics Helpers
Solution _sa_initialization(const Solution& s_0, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng);
Solution feasible_or_repair_algorithm_10(const Solution& s, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng);

// RCRS-GRASP hybrid population initialisation (replaces SA in Algorithm 2).
// `alpha` in [0,1]: 0 = pure greedy RCRS, 1 = full random GRASP.
Solution rcrs_grasp_initialization(const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng, double alpha);
