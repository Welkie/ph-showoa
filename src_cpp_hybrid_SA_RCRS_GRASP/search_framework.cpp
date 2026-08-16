#include <numeric>
#include "search_framework.h"
#include "operator.h"
#include "objective.h"
#include "profile.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <exception>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline int randint(int low, int high, LegacyMt19937& rng) {
    return legacy_randint(low, high, rng);
}

static inline double randdouble(double low, double high, LegacyMt19937& rng) {
    return legacy_randdouble(low, high, rng);
}

static inline void argsort(const std::vector<double>& fit, std::vector<int>& argrank) {
    argrank.resize(fit.size());
    for (size_t i = 0; i < fit.size(); ++i) argrank[i] = i;
    std::sort(argrank.begin(), argrank.end(), [&](int a, int b) {
        return fit[a] < fit[b];
    });
}

static inline double mean(const std::vector<double>& fit) {
    double sum = 0.0;
    for (double val : fit) sum += val;
    return fit.empty() ? 0.0 : sum / fit.size();
}

static ConstructionConfig construction_config_for_index(
    const Data& data,
    int index,
    LegacyMt19937& rng
) {
    ConstructionConfig config;
    config.ksize = data.k_init;
    config.insertion_mode = data.init == "td" ? "td" : "rcrs";
    if (data.init == "rcrs_random" || data.init == "sa_random") {
        config.lambda = randdouble(0.0, 1.0, rng);
        config.gamma = randdouble(0.0, 1.0, rng);
    } else if (index < static_cast<int>(data.latin.size())) {
        config.lambda = data.latin[index].first;
        config.gamma = data.latin[index].second;
    } else if (!data.latin.empty()) {
        // The legacy sequential loop retained its final Latin slot for indices
        // beyond the square. Make that behavior explicit and thread-safe.
        config.lambda = data.latin.back().first;
        config.gamma = data.latin.back().second;
    } else {
        config.lambda = data.lambda_gamma.first;
        config.gamma = data.lambda_gamma.second;
    }
    return config;
}

static bool use_batched_population_construction(
    const Data& data,
    BaseComputeBackend* backend,
    size_t population_size
) {
    if (backend == nullptr || !backend->is_gpu_backend() ||
        !data.gpu_batched_insertion || data.execution_policy == "legacy" ||
        population_size == 0) {
        return false;
    }
    if (data.execution_policy == "cuda_force") {
        return true;
    }

    const size_t customers = static_cast<size_t>(std::max(1, data.customer_num));
    WorkShape shape;
    shape.context = EVALUATION_CONTEXT_INITIALIZATION;
    shape.item_count = population_size * customers * customers;
    shape.total_nodes = population_size * (customers + 2);
    shape.max_length = customers + 2;
    shape.transfer_bytes = shape.total_nodes * sizeof(int) +
        shape.item_count * 4 * sizeof(int);
    return backend->choose_target(shape) == EvalTarget::CudaBatch;
}

// ---------------------------------------------------------------------------
// Objective comparison helpers
// ---------------------------------------------------------------------------
// Weighted cost remains available for legacy behavior and reporting. In
// lexicographic mode population ranking, tournament selection, acceptance,
// migration, and best tracking all prioritize (vehicle count, distance).

using ObjectiveKey = ObjectiveValue;

static inline ObjectiveKey obj_key_full(const Solution& s, const Data& data) {
    double dist = 0.0;
    for (int i = 0; i < s.len(); ++i) {
        dist += s.get(i).self.dist;
    }
    return ObjectiveKey{true, s.len(), dist, s.cost};
}

// Returns true if candidate is strictly better than incumbent under the
// chosen objective mode.
static inline bool is_better_key(const ObjectiveKey& candidate,
                                  const ObjectiveKey& incumbent,
                                  const std::string& objective) {
    return objective_better(candidate, incumbent, objective);
}

// Sentinel "worst possible" ObjectiveKey for initialization
static inline ObjectiveKey worst_key() {
    return worst_objective_value();
}

static inline void argsort_population(
    const std::vector<Solution>& pop,
    const std::vector<double>& fit,
    std::vector<int>& argrank,
    const Data& data
) {
    if (data.objective != "lexicographic") {
        argsort(fit, argrank);
        return;
    }

    std::vector<ObjectiveKey> keys;
    keys.reserve(pop.size());
    for (size_t i = 0; i < pop.size(); ++i) {
        ObjectiveKey key = obj_key_full(pop[i], data);
        key.feasible = i < fit.size() && std::isfinite(fit[i]);
        keys.push_back(key);
    }
    argrank.resize(pop.size());
    for (size_t i = 0; i < pop.size(); ++i) {
        argrank[i] = static_cast<int>(i);
    }
    std::stable_sort(argrank.begin(), argrank.end(), [&](int a, int b) {
        if (is_better_key(keys[a], keys[b], data.objective)) return true;
        if (is_better_key(keys[b], keys[a], data.objective)) return false;
        return a < b;
    });
}

static inline int _configure_omp_threads(const Data& data) {
#ifdef _OPENMP
    int threads = data.parallel_workers > 0
        ? data.parallel_workers
        : static_cast<int>(std::thread::hardware_concurrency());
    if (threads <= 0) {
        threads = 1;
    }
    omp_set_num_threads(threads);
    return threads;
#else
    (void)data;
    return 1;
#endif
}

static inline void _dynamic_parameters(int iteration, int max_iter, double& a, double& p_hybrid) {
    if (max_iter <= 0) {
        a = 0.0;
        p_hybrid = 0.15;
        return;
    }
    double ratio = std::min(std::max(static_cast<double>(iteration) / static_cast<double>(max_iter), 0.0), 1.0);
    a = 2.0 - 2.0 * ratio;
    p_hybrid = std::max(0.15, 0.5 * (1.0 - ratio));
}

static inline double _mode_probability(double p_hybrid, const Data& data) {
    if (data.hybrid_mode == "sho") return 1.0;
    if (data.hybrid_mode == "woa") return 0.0;
    return p_hybrid; // "ph_showoa" fallback
}

static inline int _tournament_peer_index(
    const std::vector<Solution>& pop,
    const std::vector<double>& pop_fit,
    int current_index,
    const Data& data,
    LegacyMt19937& rng,
    int k = 3
) {
    std::vector<int> candidates;
    for (size_t i = 0; i < pop_fit.size(); ++i) {
        if (static_cast<int>(i) != current_index) {
            candidates.push_back(i);
        }
    }
    if (candidates.empty()) return current_index;
    legacy_shuffle(candidates.begin(), candidates.end(), rng);
    
    int best_idx = candidates[0];
    for (int i = 1; i < k && i < candidates.size(); ++i) {
        int idx = candidates[i];
        bool better = false;
        if (data.objective == "lexicographic") {
            ObjectiveKey candidate_key = obj_key_full(pop[idx], data);
            ObjectiveKey best_key = obj_key_full(pop[best_idx], data);
            candidate_key.feasible = std::isfinite(pop_fit[idx]);
            best_key.feasible = std::isfinite(pop_fit[best_idx]);
            better = is_better_key(candidate_key, best_key, data.objective);
        } else {
            better = pop_fit[idx] < pop_fit[best_idx];
        }
        if (better) {
            best_idx = idx;
        }
    }
    return best_idx;
}

static inline std::vector<int> _solution_customers(const Solution& s, const Data& data) {
    std::vector<int> customers;
    for (int i = 0; i < s.len(); ++i) {
        for (int node : s.get(i).node_list) {
            if (node != data.DC) customers.push_back(node);
        }
    }
    return customers;
}

static inline std::vector<int> _route_customers(const Route& route, const Data& data) {
    std::vector<int> customers;
    for (int node : route.node_list) {
        if (node != data.DC) {
            customers.push_back(node);
        }
    }
    return customers;
}

static inline Route _make_route_from_customers(const std::vector<int>& customers, const Data& data) {
    Route route(data);
    route.node_list.clear();
    route.node_list.push_back(data.DC);
    route.node_list.insert(route.node_list.end(), customers.begin(), customers.end());
    route.node_list.push_back(data.DC);
    route.update(data);
    return route;
}

static inline bool _check_route_capacity_only_sf(const std::vector<int>& nl, const Data& data) {
    if (nl.size() <= 2) return true;
    double load = 0.0;
    for (int node : nl) {
        if (node >= 0 && node <= data.customer_num) {
            load += data.node[node].delivery;
        }
    }
    if (load > data.vehicle.capacity) return false;
    for (size_t i = 1; i < nl.size(); ++i) {
        int node = nl[i];
        if (node < 0 || node > data.customer_num) return false;
        load = load - data.node[node].delivery + data.node[node].pickup;
        if (load < 0.0 || load > data.vehicle.capacity) return false;
    }
    return true;
}

static inline void _remove_customers_from_solution(Solution& s, const std::set<int>& customers, const Data& data) {
    if (customers.empty()) return;
    for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
        Route& route = s.get(r_idx);
        std::vector<int> next_nl;
        next_nl.reserve(route.node_list.size());
        for (int node : route.node_list) {
            if (node == data.DC || customers.find(node) == customers.end()) {
                next_nl.push_back(node);
            }
        }
        if (next_nl.empty() || next_nl.front() != data.DC) {
            next_nl.insert(next_nl.begin(), data.DC);
        }
        if (next_nl.back() != data.DC) {
            next_nl.push_back(data.DC);
        }
        route.node_list = next_nl;
        route.update(data);
    }
    s.update(data);
    s.cal_cost(data);
}

static inline bool _append_route_if_clean(Solution& child, const Route& route, std::set<int>& inserted, const Data& data, BaseComputeBackend* backend) {
    std::vector<int> customers = _route_customers(route, data);
    if (customers.empty()) return false;
    for (int node : customers) {
        if (inserted.find(node) != inserted.end()) return false;
    }
    std::vector<int> nl;
    nl.reserve(customers.size() + 2);
    nl.push_back(data.DC);
    nl.insert(nl.end(), customers.begin(), customers.end());
    nl.push_back(data.DC);
    if (!_chk_route_list(nl, data, backend)) return false;
    child.append(_make_route_from_customers(customers, data));
    for (int node : customers) {
        inserted.insert(node);
    }
    return true;
}

static inline bool _append_customer_to_best_position(Solution& s, int node, const Data& data, BaseComputeBackend* backend) {
    std::vector<std::vector<int>> routes_to_check;
    std::vector<std::pair<int, int>> route_meta;
    routes_to_check.push_back({data.DC, node, data.DC});
    route_meta.push_back({-1, 1});

    std::vector<double> original_costs(s.len(), 0.0);
    for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
        const Route& route = s.get(r_idx);
        original_costs[r_idx] = route_total_cost(route, data);
        for (int pos = 1; pos < static_cast<int>(route.node_list.size()); ++pos) {
            std::vector<int> candidate = route.node_list;
            candidate.insert(candidate.begin() + pos, node);
            routes_to_check.push_back(std::move(candidate));
            route_meta.push_back({r_idx, pos});
        }
    }

    std::vector<RouteEval> evals = evaluate_route_batch(routes_to_check, data, backend);
    int best_route = -2;
    int best_pos = 1;
    double best_delta = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < evals.size() && i < route_meta.size(); ++i) {
        if (!evals[i].feasible) continue;
        int r_idx = route_meta[i].first;
        double delta = r_idx == -1 ? evals[i].cost : evals[i].cost - original_costs[r_idx];
        if (delta < best_delta - 1e-9) {
            best_delta = delta;
            best_route = r_idx;
            best_pos = route_meta[i].second;
        }
    }

    if (best_route == -2) return false;
    if (best_route == -1) {
        Route route(data);
        route.node_list.insert(route.node_list.begin() + 1, node);
        route.update(data);
        s.append(route);
    } else {
        Route& route = s.get(best_route);
        route.node_list.insert(route.node_list.begin() + best_pos, node);
        route.update(data);
    }
    s.cal_cost(data);
    return true;
}

static inline Solution _build_solution_from_sequence(const std::vector<int>& sequence, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    Solution s;
    std::vector<int> route_nodes;
    for (int node : sequence) {
        std::vector<int> trial = {data.DC};
        trial.insert(trial.end(), route_nodes.begin(), route_nodes.end());
        trial.push_back(node);
        trial.push_back(data.DC);
        if (_chk_route_list(trial, data, backend)) {
            route_nodes.push_back(node);
            continue;
        }
        if (!route_nodes.empty()) {
            Route r_new(data);
            r_new.node_list = {data.DC};
            r_new.node_list.insert(r_new.node_list.end(), route_nodes.begin(), route_nodes.end());
            r_new.node_list.push_back(data.DC);
            r_new.update(data);
            s.append(r_new);
        }
        route_nodes.clear();
        std::vector<int> single = {data.DC, node, data.DC};
        if (_chk_route_list(single, data, backend)) {
            route_nodes.push_back(node);
        } else {
            Route r_new(data);
            r_new.node_list.insert(r_new.node_list.begin() + 1, node);
            r_new.update(data);
            s.append(r_new);
        }
    }
    if (!route_nodes.empty()) {
        Route r_new(data);
        r_new.node_list = {data.DC};
        r_new.node_list.insert(r_new.node_list.end(), route_nodes.begin(), route_nodes.end());
        r_new.node_list.push_back(data.DC);
        r_new.update(data);
        s.append(r_new);
    }
    Solution s_repaired = feasible_or_repair_algorithm_10(s, data, backend, rng);
    return s_repaired;
}

static inline Solution _li_lim_random_search(const Solution& current, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    std::vector<int> sequence = _solution_customers(current, data);
    if (sequence.size() < 2) return current.clone();

    int move_type = randint(0, 2, rng);
    if (move_type == 0) {
        int i = randint(0, sequence.size() - 1, rng);
        int j = randint(0, sequence.size() - 1, rng);
        std::swap(sequence[i], sequence[j]);
    } else if (move_type == 1) {
        int i = randint(0, sequence.size() - 1, rng);
        int node = sequence[i];
        sequence.erase(sequence.begin() + i);
        int j = randint(0, sequence.size(), rng);
        sequence.insert(sequence.begin() + j, node);
    } else {
        int i = randint(0, sequence.size() - 1, rng);
        int j = randint(0, sequence.size() - 1, rng);
        if (i > j) std::swap(i, j);
        std::reverse(sequence.begin() + i, sequence.begin() + j + 1);
    }
    return _build_solution_from_sequence(sequence, data, backend, rng);
}

static inline void _inject_elite_routes(Solution& child, const Solution& best, double c_value, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    if (best.len() == 0) return;

    int count = std::max(1, std::min(best.len(), static_cast<int>(std::round(1.0 + std::max(0.0, 2.0 - c_value)))));
    std::vector<int> route_indices(best.len());
    for (int i = 0; i < best.len(); ++i) route_indices[i] = i;
    legacy_shuffle(route_indices.begin(), route_indices.end(), rng);

    std::set<int> selected_customers;
    for (int i = 0; i < count; ++i) {
        for (int node : _route_customers(best.get(route_indices[i]), data)) {
            selected_customers.insert(node);
        }
    }

    _remove_customers_from_solution(child, selected_customers, data);
    std::set<int> inserted;
    for (int node : _solution_customers(child, data)) {
        inserted.insert(node);
    }

    for (int i = 0; i < count; ++i) {
        const Route& seed_route = best.get(route_indices[i]);
        if (!_append_route_if_clean(child, seed_route, inserted, data, backend)) {
            for (int node : _route_customers(seed_route, data)) {
                if (inserted.find(node) == inserted.end()) {
                    if (_append_customer_to_best_position(child, node, data, backend)) {
                        inserted.insert(node);
                    }
                }
            }
        }
    }
    child.update(data);
    child.cal_cost(data);
}

static inline void _inject_elite_segments(Solution& child, const Solution& best, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    std::vector<int> candidate_routes;
    for (int i = 0; i < best.len(); ++i) {
        if (!_route_customers(best.get(i), data).empty()) {
            candidate_routes.push_back(i);
        }
    }
    if (candidate_routes.empty()) return;

    legacy_shuffle(candidate_routes.begin(), candidate_routes.end(), rng);
    int segment_count = std::min(static_cast<int>(candidate_routes.size()), randint(1, 3, rng));
    std::vector<std::vector<int>> segments;
    std::set<int> selected_customers;
    for (int i = 0; i < segment_count; ++i) {
        std::vector<int> customers = _route_customers(best.get(candidate_routes[i]), data);
        if (customers.empty()) continue;
        double l_value = randdouble(-1.0, 1.0, rng);
        double spiral_scale = std::abs(std::exp(l_value) * std::cos(2.0 * 3.14159265358979323846 * l_value));
        int seg_len = std::max(1, std::min(static_cast<int>(customers.size()), static_cast<int>(std::round(1.0 + spiral_scale))));
        int start = randint(0, static_cast<int>(customers.size()) - seg_len, rng);
        std::vector<int> segment;
        for (int j = start; j < start + seg_len; ++j) {
            int node = customers[j];
            if (selected_customers.find(node) == selected_customers.end()) {
                segment.push_back(node);
                selected_customers.insert(node);
            }
        }
        if (!segment.empty()) {
            segments.push_back(std::move(segment));
        }
    }

    if (segments.empty()) return;
    _remove_customers_from_solution(child, selected_customers, data);

    std::vector<std::vector<int>> routes_to_check;
    std::vector<int> segment_indices;
    std::vector<int> try_route(segments.size(), 0);
    for (size_t i = 0; i < segments.size(); ++i) {
        if (randdouble(0.0, 1.0, rng) < 0.70) {
            std::vector<int> nl;
            nl.push_back(data.DC);
            nl.insert(nl.end(), segments[i].begin(), segments[i].end());
            nl.push_back(data.DC);
            routes_to_check.push_back(std::move(nl));
            segment_indices.push_back(static_cast<int>(i));
            try_route[i] = 1;
        }
    }

    std::vector<int> feasible_segment_route(segments.size(), 0);
    if (!routes_to_check.empty()) {
        std::vector<RouteEval> evals = evaluate_route_batch(routes_to_check, data, backend);
        for (size_t i = 0; i < evals.size() && i < segment_indices.size(); ++i) {
            feasible_segment_route[segment_indices[i]] = evals[i].feasible ? 1 : 0;
        }
    }

    for (size_t i = 0; i < segments.size(); ++i) {
        if (try_route[i] && feasible_segment_route[i]) {
            child.append(_make_route_from_customers(segments[i], data));
        } else {
            for (int node : segments[i]) {
                _append_customer_to_best_position(child, node, data, backend);
            }
        }
    }
    child.update(data);
    child.cal_cost(data);
}

static inline Solution _woa_intensification(const Solution& current, const Solution& best, double a, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    double r1 = randdouble(0.0, 1.0, rng);
    double r2 = randdouble(0.0, 1.0, rng);
    double a_vector = 2.0 * a * r1 - a;
    
    if (std::abs(a_vector) < 1.0) {
        Solution child = current.clone();
        _inject_elite_routes(child, best, a, data, backend, rng);
        Solution s_repaired = feasible_or_repair_algorithm_10(child, data, backend, rng);
        return s_repaired;
    }

    Solution child = _li_lim_random_search(current, data, backend, rng);
    Solution s_repaired = feasible_or_repair_algorithm_10(child, data, backend, rng);
    return s_repaired;
}

static inline Solution _guided_route_crossover(const Solution& best, const Solution& peer, const Solution& current, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    if (best.len() == 0) return current.clone();
    Solution child;

    std::set<int> kept_customers;
    std::vector<int> route_indices(best.len());
    for (int i = 0; i < best.len(); ++i) route_indices[i] = i;
    legacy_shuffle(route_indices.begin(), route_indices.end(), rng);
    int take = (best.len() == 1 || randdouble(0.0, 1.0, rng) < 0.6) ? 1 : 2;

    for (int i = 0; i < take && i < static_cast<int>(route_indices.size()); ++i) {
        std::vector<int> customers = _route_customers(best.get(route_indices[i]), data);
        if (customers.empty()) continue;
        child.append(_make_route_from_customers(customers, data));
        for (int node : customers) kept_customers.insert(node);
    }

    std::vector<int> remaining;
    auto add_remaining = [&](const Solution& parent) {
        for (int r_idx = 0; r_idx < parent.len(); ++r_idx) {
            for (int node : parent.get(r_idx).node_list) {
                if (node == data.DC || kept_customers.find(node) != kept_customers.end()) {
                    continue;
                }
                if (std::find(remaining.begin(), remaining.end(), node) == remaining.end()) {
                    remaining.push_back(node);
                }
            }
        }
    };
    add_remaining(peer);
    add_remaining(current);

    std::vector<int> current_route{data.DC};
    for (int node : remaining) {
        std::vector<int> singleton{data.DC, node, data.DC};
        if (!_check_route_capacity_only_sf(singleton, data)) {
            if (_append_customer_to_best_position(child, node, data, backend)) {
                kept_customers.insert(node);
            }
            continue;
        }

        std::vector<int> candidate = current_route;
        candidate.push_back(node);
        candidate.push_back(data.DC);
        if (current_route.size() > 1 && !_check_route_capacity_only_sf(candidate, data)) {
            std::vector<int> customers(current_route.begin() + 1, current_route.end());
            child.append(_make_route_from_customers(customers, data));
            current_route = {data.DC};
        }

        current_route.push_back(node);
        kept_customers.insert(node);
    }

    if (current_route.size() > 1) {
        std::vector<int> customers(current_route.begin() + 1, current_route.end());
        child.append(_make_route_from_customers(customers, data));
    }

    if (randdouble(0.0, 1.0, rng) < data.sho_mutation_prob) {
        _inject_elite_segments(child, best, data, backend, rng);
    }

    Solution s_repaired = feasible_or_repair_algorithm_10(child, data, backend, rng);
    return s_repaired;
}

static inline bool _sa_accept(
    const Solution& new_solution,
    const Solution& current_solution,
    double current_fit,
    int iteration,
    int max_iter,
    const Data& data,
    LegacyMt19937& rng
) {
    double delta = new_solution.cost - current_fit;
    double scale = std::abs(current_fit);
    if (data.objective == "lexicographic") {
        ObjectiveKey candidate = obj_key_full(new_solution, data);
        ObjectiveKey current = obj_key_full(current_solution, data);
        if (candidate.nv < current.nv) return true;
        if (candidate.nv > current.nv) return false;
        delta = candidate.dist - current.dist;
        scale = std::abs(current.dist);
    }
    if (delta <= 0.001) return true;
    double temperature = max_iter > 0 ? 1.0 - (static_cast<double>(iteration) / max_iter) : 0.0;
    double denominator = 1e-6 + temperature * scale;
    double probability = std::exp(-delta / denominator);
    return randdouble(0.0, 1.0, rng) < probability;
}

static inline void _update_agent(
    const Solution& current,
    const Solution& peer,
    const Solution& best,
    double current_fit,
    double p_hybrid,
    double a,
    int iteration,
    int max_iter,
    int seed,
    const Data& data,
    BaseComputeBackend* backend,
    Solution& out_sol,
    double& out_cost,
    bool& out_accepted
) {
    LegacyMt19937 rng(seed);
    Solution new_solution;

    if (randdouble(0.0, 1.0, rng) < p_hybrid) {
        ScopedProfileTimer sho_timer(data.profile, profile_registry().sho_update);
        new_solution = _guided_route_crossover(best, peer, current, data, backend, rng);
    } else {
        ScopedProfileTimer woa_timer(data.profile, profile_registry().woa_update);
        new_solution = _woa_intensification(current, best, a, data, backend, rng);
    }

    new_solution.cal_cost(data);
    bool accepted = _sa_accept(new_solution, current, current_fit, iteration, max_iter, data, rng);
    if (accepted) {
        out_sol = new_solution;
        out_cost = new_solution.cost;
        out_accepted = true;
    } else {
        out_sol = current;
        out_cost = current_fit;
        out_accepted = false;
    }
}

static inline void _diversify_pop(std::vector<Solution>& pop, std::vector<double>& pop_fit, std::vector<int>& pop_argrank, std::vector<bool>& pop_dirty, const Solution& best_s, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    argsort_population(pop, pop_fit, pop_argrank, data);
    int elite_index = pop_argrank[0];
    pop[elite_index].copy_from(best_s);
    pop_fit[elite_index] = best_s.cost;
    pop_dirty[elite_index] = true;

    int diversify_count = static_cast<int>(std::round(pop.size() * data.diversify_ratio));
    for (int i = 1; i <= diversify_count && i < pop.size(); ++i) {
        int idx = pop_argrank[i];
        Solution candidate = pop[idx].clone();
        random_removal(candidate, data, rng);
        regret_insertion(candidate, data, backend, rng);
        pop[idx].copy_from(candidate);
        pop_fit[idx] = candidate.cost;
        pop_dirty[idx] = true;
    }
    argsort_population(pop, pop_fit, pop_argrank, data);
}

static inline void _refresh_population_fitness_on_gpu(
    const std::vector<Solution>& pop,
    std::vector<double>& pop_fit,
    std::vector<bool>& pop_dirty,
    const Data& data,
    BaseComputeBackend* backend,
    bool verification_requested = false
) {
    if (backend == nullptr || !backend->is_gpu_backend() || !data.gpu_solution_eval) {
        return;
    }
    if (data.execution_policy == "adaptive" && !verification_requested) {
        return;
    }

    std::vector<bool> effective_dirty = pop_dirty;
    if (verification_requested) {
        effective_dirty.assign(pop.size(), true);
    }
    EncodedPopulation encoded = encode_population_for_gpu(pop, data, effective_dirty);
    std::vector<SolutionEval> evals = backend->evaluate_solutions(encoded);
    if (evals.size() != pop.size()) {
        std::printf("WARNING: GPU solution evaluation returned %zu results for %zu solutions. Keeping CPU fitness values.\n",
                    evals.size(), pop.size());
        return;
    }

    for (size_t i = 0; i < evals.size(); ++i) {
        if (!evals[i].feasible) {
            std::printf("WARNING: GPU solution evaluation marked population index %zu infeasible, flags=%d. Assigning infinity fitness.\n",
                        i, evals[i].violation_flags);
            pop_fit[i] = std::numeric_limits<double>::infinity();
        } else {
            pop_fit[i] = evals[i].fitness;
        }
        pop_dirty[i] = false;
    }
}


struct Island {
    std::vector<Solution> pop;
    std::vector<double> pop_fit;
    std::vector<int> pop_argrank;
    std::vector<bool> pop_dirty;
    LegacyMt19937 rng;
    int last_improvement_gen = 0;
    Solution local_best;
};

static inline void _inject_global_best(
    std::vector<Solution>& pop,
    std::vector<double>& pop_fit,
    std::vector<int>& pop_argrank,
    std::vector<bool>& pop_dirty,
    const Solution& best_s,
    const Data& data
) {
    argsort_population(pop, pop_fit, pop_argrank, data);
    int worst_index = pop_argrank.back();
    pop[worst_index].copy_from(best_s);
    pop_fit[worst_index] = best_s.cost;
    pop_dirty[worst_index] = true;
    argsort_population(pop, pop_fit, pop_argrank, data);
}

static inline void _refresh_multi_island_population_fitness_on_gpu(
    std::vector<Island>& islands,
    const Data& data,
    BaseComputeBackend* backend,
    bool verification_requested = false
) {
    if (backend == nullptr || !backend->is_gpu_backend() || !data.gpu_solution_eval) {
        return;
    }
    if (data.execution_policy == "adaptive" && !verification_requested) {
        return;
    }
    size_t total_solutions = 0;
    for (const auto& island : islands) {
        total_solutions += island.pop.size();
    }
    std::vector<Solution> flat_pop;
    flat_pop.reserve(total_solutions);
    for (const auto& island : islands) {
        flat_pop.insert(flat_pop.end(), island.pop.begin(), island.pop.end());
    }
    std::vector<double> flat_fit(total_solutions);
    std::vector<bool> flat_dirty;
    flat_dirty.reserve(total_solutions);
    for (const auto& island : islands) {
        flat_dirty.insert(flat_dirty.end(), island.pop_dirty.begin(), island.pop_dirty.end());
    }
    _refresh_population_fitness_on_gpu(flat_pop, flat_fit, flat_dirty, data, backend, verification_requested);

    size_t offset = 0;
    for (auto& island : islands) {
        for (size_t i = 0; i < island.pop.size(); ++i) {
            island.pop_fit[i] = flat_fit[offset + i];
            island.pop_dirty[i] = false;
        }
        offset += island.pop.size();
    }
}



void search_framework(Data& data, Solution& best_s) {
    std::cout << "Entering search_framework" << std::endl;
    profile_reset();
    std::cout << "Profile reset" << std::endl;
    int p_size = data.p_size;
    auto stime = std::chrono::high_resolution_clock::now();
    int used = 0;
    int run = 1;
    int completed_runs = 0;

    BaseComputeBackend* backend = create_backend(data, data.compute_backend);
    LegacyMt19937 global_rng(data.seed);
    int omp_threads = _configure_omp_threads(data);
    if (backend != nullptr && backend->is_gpu_backend() && data.gpu_serialize_workers) {
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        omp_threads = 1;
        std::printf("CUDA backend active; serializing OpenMP worker updates around the GPU backend.\n");
    } else if (backend != nullptr && backend->is_gpu_backend()) {
        if (data.gpu_request_broker) {
            std::printf("CUDA backend active; OpenMP worker updates and request broker enabled.\n");
        } else {
            std::printf("CUDA backend active; OpenMP worker updates enabled. Large GPU calls are mutex-serialized inside the backend.\n");
        }
    }
    std::printf("OpenMP threads: %d\n", omp_threads);

    while (run <= data.runs) {
        std::printf("---------------------------------Run %d---------------------------\n", run);
        
        if (data.num_islands > 1) {
            // =========================================================================
            // Island Model Branch
            // =========================================================================
            int K = data.num_islands;
            int P = data.p_size;
            std::vector<Island> islands(K);

            std::printf("Island Model Initialization - %d islands of size %d\n", K, P);
            try {
                ScopedProfileTimer init_timer(data.profile, profile_registry().initialization, K * P);
                Solution initial_solution;
                for (int k = 0; k < K; ++k) {
                    islands[k].pop.assign(P, initial_solution);
                    islands[k].pop_fit.assign(P, std::numeric_limits<double>::infinity());
                    islands[k].pop_argrank.resize(P);
                    islands[k].pop_dirty.assign(P, true);
                    islands[k].rng.seed(data.seed + k * 1000 + run * 100000);
                    std::iota(islands[k].pop_argrank.begin(), islands[k].pop_argrank.end(), 0);
                    islands[k].last_improvement_gen = 0;
                }

                std::vector<std::exception_ptr> initialization_errors(K * P);
                if (data.init == "rcrs_grasp" || data.init == "rcg") {
                    #pragma omp parallel for
                    for (int ip = 0; ip < K * P; ++ip) {
                        const int k = ip / P;
                        const int i = ip % P;
                        try {
                            islands[k].pop[i].clear(data);
                            LegacyMt19937 seed_rng(
                                data.seed + k * 1000 + run * 100000 + 1000000 + i
                            );
                            double ind_alpha = randdouble(data.grasp_alpha_lo, data.grasp_alpha_hi, seed_rng);
                            islands[k].pop[i] = rcrs_grasp_initialization(data, backend, seed_rng, ind_alpha);
                            // Post-refinement: SA 25 iterations to smooth routes and empty sparse routes (Pd-Shift/Pd-Exchange)
                            islands[k].pop[i] = _sa_initialization(islands[k].pop[i], data, backend, seed_rng, 25);
                            islands[k].pop_fit[i] = islands[k].pop[i].cost;
                        } catch (...) {
                            initialization_errors[ip] = std::current_exception();
                        }
                    }
                } else if (use_batched_population_construction(data, backend, K * P)) {
                    std::printf("Generation-level CUDA construction batching enabled for %d solutions.\n", K * P);
                    std::vector<Solution> flat_population(K * P);
                    std::vector<ConstructionConfig> configs(K * P);
                    std::vector<LegacyMt19937> initialization_rngs;
                    initialization_rngs.reserve(K * P);
                    for (int ip = 0; ip < K * P; ++ip) {
                        const int k = ip / P;
                        const int i = ip % P;
                        initialization_rngs.emplace_back(
                            data.seed + k * 1000 + run * 100000 + 1000000 + i
                        );
                        configs[ip] = construction_config_for_index(
                            data, i, initialization_rngs.back()
                        );
                    }
                    new_route_insertion_batched(
                        flat_population, data, configs, backend, initialization_rngs
                    );
                    #pragma omp parallel for
                    for (int ip = 0; ip < K * P; ++ip) {
                        const int k = ip / P;
                        const int i = ip % P;
                        try {
                            if (data.init == "sa") {
                                flat_population[ip] = _sa_initialization(
                                    flat_population[ip], data, backend, initialization_rngs[ip]
                                );
                            }
                            islands[k].pop[i].copy_from(flat_population[ip]);
                            islands[k].pop_fit[i] = islands[k].pop[i].cost;
                        } catch (...) {
                            initialization_errors[ip] = std::current_exception();
                        }
                    }
                } else {
                    #pragma omp parallel for
                    for (int ip = 0; ip < K * P; ++ip) {
                        const int k = ip / P;
                        const int i = ip % P;
                        try {
                            islands[k].pop[i].clear(data);
                            LegacyMt19937 seed_rng(
                                data.seed + k * 1000 + run * 100000 + 1000000 + i
                            );
                            const ConstructionConfig config = construction_config_for_index(data, i, seed_rng);
                            new_route_insertion(islands[k].pop[i], data, config, backend, seed_rng);
                            if (data.init == "sa") {
                                islands[k].pop[i] = _sa_initialization(
                                    islands[k].pop[i], data, backend, seed_rng
                                );
                            }
                            islands[k].pop_fit[i] = islands[k].pop[i].cost;
                        } catch (...) {
                            initialization_errors[ip] = std::current_exception();
                        }
                    }
                }
                for (const std::exception_ptr& error : initialization_errors) {
                    if (error) std::rethrow_exception(error);
                }

                _refresh_multi_island_population_fitness_on_gpu(islands, data, backend);

                for (int k = 0; k < K; ++k) {
                    argsort_population(islands[k].pop, islands[k].pop_fit, islands[k].pop_argrank, data);
                    // Set initial local best
                    islands[k].local_best = islands[k].pop[islands[k].pop_argrank[0]];
                    std::printf("Island %d local best initialized: %.4f\n", k, islands[k].local_best.cost);
                }
            } catch (const std::exception& e) {
                std::printf("FATAL ERROR: Exception in initialization: %s\n", e.what()); std::fflush(stdout);
                return;
            }

            // Initialize global best_s
            ObjectiveKey best_key = worst_key();
            for (int k = 0; k < K; ++k) {
                ObjectiveKey key = obj_key_full(islands[k].local_best, data);
                if (is_better_key(key, best_key, data.objective)) {
                    best_key = key;
                    best_s.copy_from(islands[k].local_best);
                }
            }
            std::printf("Global best initialized: NV=%d dist=%.4f cost=%.4f\n",
                        best_s.len(), best_key.dist, best_s.cost);

            auto elapsed = std::chrono::high_resolution_clock::now() - stime;
            used = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            std::printf("already consumed %d sec\n", used);

            for (int gen = 1; gen <= data.max_iter; ++gen) {
                double a, p_hybrid;
                _dynamic_parameters(gen - 1, data.max_iter, a, p_hybrid);
                double p_mode = _mode_probability(p_hybrid, data);
                // Pre-calculate tournament indices and peer solutions for each island
                std::vector<std::vector<int>> peer_indices(K, std::vector<int>(P, 0));
                std::vector<std::vector<int>> update_seeds(K, std::vector<int>(P, 0));
                std::vector<Solution> local_best_snapshots(K);
                
                for (int k = 0; k < K; ++k) {
                    local_best_snapshots[k] = islands[k].local_best;
                    for (int i = 0; i < P; ++i) {
                        peer_indices[k][i] = _tournament_peer_index(
                            islands[k].pop, islands[k].pop_fit, i, data, islands[k].rng, 3
                        );
                        update_seeds[k][i] = static_cast<int>(islands[k].rng());
                    }
                }

                // Create temporary buffers for the next population
                std::vector<std::vector<Solution>> next_pop(K, std::vector<Solution>(P));
                std::vector<std::vector<double>> next_fit(K, std::vector<double>(P, 0.0));
                std::vector<std::vector<bool>> next_dirty(K, std::vector<bool>(P, false));
                std::vector<int> next_changed(K * P, 0);

                int accepted_count_total = 0;

                // OMP loop over all islands and populations
                {
                    ScopedProfileTimer update_timer(data.profile, profile_registry().population_update, K * P);
                    #pragma omp parallel for reduction(+:accepted_count_total)
                    for (int ip = 0; ip < K * P; ++ip) {
                        int k = ip / P;
                        int index = ip % P;

                        Solution out_sol;
                        double out_cost;
                        bool accepted;

                        _update_agent(
                            islands[k].pop[index],
                            islands[k].pop[peer_indices[k][index]],
                            local_best_snapshots[k], // each island guides search using its local best!
                            islands[k].pop_fit[index],
                            p_mode,
                            a,
                            gen - 1,
                            data.max_iter,
                            update_seeds[k][index],
                            data,
                            backend,
                            out_sol,
                            out_cost,
                            accepted
                        );

                        next_pop[k][index] = out_sol;
                        next_fit[k][index] = out_cost;
                        next_changed[ip] = accepted ? 1 : 0;
                        if (accepted) {
                            accepted_count_total++;
                        }
                    }
                }

                // Swap and evaluate
                for (int k = 0; k < K; ++k) {
                    for (int index = 0; index < P; ++index) {
                        next_dirty[k][index] = islands[k].pop_dirty[index] ||
                            next_changed[k * P + index] != 0;
                    }
                    islands[k].pop.swap(next_pop[k]);
                    islands[k].pop_fit.swap(next_fit[k]);
                    islands[k].pop_dirty.swap(next_dirty[k]);
                }
                const bool verify_generation = data.gpu_solution_verify_interval > 0 &&
                    gen % data.gpu_solution_verify_interval == 0;
                _refresh_multi_island_population_fitness_on_gpu(
                    islands, data, backend, verify_generation
                );
                for (int k = 0; k < K; ++k) {
                    argsort_population(islands[k].pop, islands[k].pop_fit, islands[k].pop_argrank, data);
                }

                // Update local and global bests
                bool global_improved = false;
                for (int k = 0; k < K; ++k) {
                    ObjectiveKey prev_local_best_key = obj_key_full(islands[k].local_best, data);

                    // Check best of population
                    ObjectiveKey cand_key = obj_key_full(islands[k].pop[islands[k].pop_argrank[0]], data);
                    if (is_better_key(cand_key, prev_local_best_key, data.objective)) {
                        islands[k].local_best.copy_from(islands[k].pop[islands[k].pop_argrank[0]]);
                        islands[k].last_improvement_gen = gen;
                    }

                    // Local Search on local best periodically
                    if (gen % data.local_search_interval == 0) {
                        Solution elite = islands[k].local_best.clone();
                        do_local_search(elite, data, backend);
                        ObjectiveKey elite_key = obj_key_full(elite, data);
                        ObjectiveKey current_local_best_key = obj_key_full(islands[k].local_best, data);
                        if (is_better_key(elite_key, current_local_best_key, data.objective)) {
                            islands[k].local_best.copy_from(elite);
                            islands[k].last_improvement_gen = gen;
                        }
                        _inject_global_best(islands[k].pop, islands[k].pop_fit, islands[k].pop_argrank, islands[k].pop_dirty, islands[k].local_best, data);
                    }

                    // Diversification on local population if stagnated
                    if (gen % data.stagnation_interval == 0 && gen - islands[k].last_improvement_gen >= data.stagnation_interval) {
                        _diversify_pop(islands[k].pop, islands[k].pop_fit, islands[k].pop_argrank, islands[k].pop_dirty, islands[k].local_best, data, backend, islands[k].rng);
                        islands[k].last_improvement_gen = gen;
                    }

                    // Check global best improvement
                    ObjectiveKey global_best_key = obj_key_full(best_s, data);
                    ObjectiveKey local_best_key = obj_key_full(islands[k].local_best, data);
                    if (is_better_key(local_best_key, global_best_key, data.objective)) {
                        best_s.copy_from(islands[k].local_best);
                        global_improved = true;
                    }
                }

                if (global_improved) {
                    elapsed = std::chrono::high_resolution_clock::now() - stime;
                    used = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                    ObjectiveKey new_global_key = obj_key_full(best_s, data);
                    std::printf("Global best update: NV=%d dist=%.4f cost=%.4f at gen %d, consumed %d sec\n",
                                best_s.len(), new_global_key.dist, best_s.cost, gen, used);
                }

                // MIGRATION PERIOD
                if (gen % data.migration_interval == 0) {
                    const int migrant_count = std::min(data.migration_size, P);
                    std::printf(
                        "Gen %d: Migration between islands started using %s mode (%d migrants)...\n",
                        gen, data.migration_mode.c_str(), migrant_count
                    );

                    std::vector<std::vector<Solution>> migration_pool(K);
                    for (int k = 0; k < K; ++k) {
                        argsort_population(islands[k].pop, islands[k].pop_fit, islands[k].pop_argrank, data);
                        migration_pool[k].reserve(migrant_count);
                        for (int m = 0; m < migrant_count; ++m) {
                            migration_pool[k].push_back(islands[k].pop[islands[k].pop_argrank[m]].clone());
                        }
                    }

                    auto replace_worst = [&](int destination, const std::vector<Solution>& migrants) {
                        const int replace_count = std::min(
                            static_cast<int>(migrants.size()),
                            static_cast<int>(islands[destination].pop_argrank.size())
                        );
                        for (int m = 0; m < replace_count; ++m) {
                            int worst_rank = static_cast<int>(islands[destination].pop_argrank.size()) - 1 - m;
                            int worst_idx = islands[destination].pop_argrank[worst_rank];
                            islands[destination].pop[worst_idx].copy_from(migrants[m]);
                            islands[destination].pop_fit[worst_idx] = migrants[m].cost;
                            islands[destination].pop_dirty[worst_idx] = true;
                        }
                        argsort_population(
                            islands[destination].pop,
                            islands[destination].pop_fit,
                            islands[destination].pop_argrank,
                            data
                        );
                    };

                    if (data.migration_mode == "ring") {
                        for (int k = 0; k < K; ++k) {
                            int dest = (k + 1) % K;
                            replace_worst(dest, migration_pool[k]);
                        }
                    } else if (data.migration_mode == "broadcast") {
                        int best_k = 0;
                        ObjectiveKey overall_best_key = obj_key_full(islands[0].local_best, data);
                        for (int k = 1; k < K; ++k) {
                            ObjectiveKey k_key = obj_key_full(islands[k].local_best, data);
                            if (is_better_key(k_key, overall_best_key, data.objective)) {
                                overall_best_key = k_key;
                                best_k = k;
                            }
                        }
                        for (int k = 0; k < K; ++k) {
                            if (k == best_k) continue;
                            replace_worst(k, migration_pool[best_k]);
                        }
                    } else { // "tournament" (default)
                        std::vector<int> island_indices(K);
                        for (int k = 0; k < K; ++k) island_indices[k] = k;
                        legacy_shuffle(island_indices.begin(), island_indices.end(), global_rng);

                        for (int i = 0; i < K - 1; i += 2) {
                            int k1 = island_indices[i];
                            int k2 = island_indices[i+1];
                            ObjectiveKey key1 = obj_key_full(islands[k1].local_best, data);
                            ObjectiveKey key2 = obj_key_full(islands[k2].local_best, data);

                            int winner = k1;
                            int loser = k2;
                            if (is_better_key(key2, key1, data.objective)) {
                                winner = k2;
                                loser = k1;
                            }

                            replace_worst(loser, migration_pool[winner]);
                        }
                    }
                }

                if (gen % 25 == 0) {
                    elapsed = std::chrono::high_resolution_clock::now() - stime;
                    used = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                    // Compute average cost over all individuals across all islands
                    double total_sum = 0.0;
                    for (int k = 0; k < K; ++k) {
                        for (double val : islands[k].pop_fit) total_sum += val;
                    }
                    double avg_cost = total_sum / (K * P);
                    std::printf("Gen: %d. a %.4f, p_hybrid %.4f, total accepted %d. Avg %.4f, Best %.4f, Best vehicles %d\n",
                        gen, a, p_mode, accepted_count_total, avg_cost, best_s.cost, best_s.len());
                    std::printf("Gen %d done, already consumed %d sec\n", gen, used);
                }
            }

        } else {
            // =========================================================================
            // Original Single Population Branch
            // =========================================================================
            std::vector<Solution> pop(p_size);
            std::vector<double> pop_fit(p_size, 0.0);
            std::vector<int> pop_argrank(p_size, 0);
            std::vector<bool> pop_dirty(p_size, true);

            // Initialization using RCRS & Simulated Annealing
            std::printf("Initialization, using %s method\n", data.init.c_str());
            try {
                ScopedProfileTimer init_timer(data.profile, profile_registry().initialization, p_size);
                std::vector<std::exception_ptr> initialization_errors(p_size);
                if (data.init == "rcrs_grasp" || data.init == "rcg") {
                    #pragma omp parallel for
                    for (int i = 0; i < p_size; ++i) {
                        try {
                            pop[i].clear(data);
                            LegacyMt19937 seed_rng(data.seed + run * 100000 + 100000 + i);
                            double ind_alpha = randdouble(data.grasp_alpha_lo, data.grasp_alpha_hi, seed_rng);
                            pop[i] = rcrs_grasp_initialization(data, backend, seed_rng, ind_alpha);
                            // Post-refinement: SA 25 iterations to smooth routes and empty sparse routes (Pd-Shift/Pd-Exchange)
                            pop[i] = _sa_initialization(pop[i], data, backend, seed_rng, 25);
                            pop_fit[i] = pop[i].cost;
                        } catch (...) {
                            initialization_errors[i] = std::current_exception();
                        }
                    }
                } else if (use_batched_population_construction(data, backend, p_size)) {
                    std::printf("Generation-level CUDA construction batching enabled for %d solutions.\n", p_size);
                    std::vector<ConstructionConfig> configs(p_size);
                    std::vector<LegacyMt19937> initialization_rngs;
                    initialization_rngs.reserve(p_size);
                    for (int i = 0; i < p_size; ++i) {
                        initialization_rngs.emplace_back(data.seed + run * 100000 + 100000 + i);
                        configs[i] = construction_config_for_index(
                            data, i, initialization_rngs.back()
                        );
                    }
                    new_route_insertion_batched(
                        pop, data, configs, backend, initialization_rngs
                    );
                    #pragma omp parallel for
                    for (int i = 0; i < p_size; ++i) {
                        try {
                            if (data.init == "sa") {
                                pop[i] = _sa_initialization(
                                    pop[i], data, backend, initialization_rngs[i]
                                );
                            }
                            pop_fit[i] = pop[i].cost;
                        } catch (...) {
                            initialization_errors[i] = std::current_exception();
                        }
                    }
                } else {
                    #pragma omp parallel for
                    for (int i = 0; i < p_size; ++i) {
                        try {
                            pop[i].clear(data);
                            LegacyMt19937 seed_rng(data.seed + run * 100000 + 100000 + i);
                            const ConstructionConfig config = construction_config_for_index(data, i, seed_rng);
                            new_route_insertion(pop[i], data, config, backend, seed_rng);
                            if (data.init == "sa") {
                                pop[i] = _sa_initialization(pop[i], data, backend, seed_rng);
                            }
                            pop_fit[i] = pop[i].cost;
                        } catch (...) {
                            initialization_errors[i] = std::current_exception();
                        }
                    }
                }
                for (int i = 0; i < p_size; ++i) {
                    if (initialization_errors[i]) {
                        std::rethrow_exception(initialization_errors[i]);
                    }
                    std::printf("Solution %d, cost %.4f\n", i, pop_fit[i]);
                }
            } catch (const std::exception& e) {
                std::printf("FATAL ERROR: Exception in initialization: %s\n", e.what()); std::fflush(stdout);
                return;
            }
            _refresh_population_fitness_on_gpu(pop, pop_fit, pop_dirty, data, backend);
            argsort_population(pop, pop_fit, pop_argrank, data);
            std::printf("Initialization done.\n");

            auto elapsed = std::chrono::high_resolution_clock::now() - stime;
            used = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            std::printf("already consumed %d sec\n", used);

            if (data.objective == "lexicographic") {
                // In lexicographic mode: initialize best_key from initial population
                ObjectiveKey best_key = worst_key();
                for (int i = 0; i < p_size; ++i) {
                    if (!std::isfinite(pop_fit[i])) continue;
                    ObjectiveKey k = obj_key_full(pop[i], data);
                    if (is_better_key(k, best_key, data.objective)) {
                        best_key = k;
                        best_s.copy_from(pop[i]);
                        std::printf("Best solution update (lex): NV=%d dist=%.4f cost=%.4f\n",
                                    best_s.len(), best_key.dist, best_s.cost);
                    }
                }
            } else {
                if (pop[pop_argrank[0]].cost < best_s.cost) {
                    best_s.copy_from(pop[pop_argrank[0]]);
                    std::printf("Best solution update: %.4f\n", best_s.cost);
                }
            }

            int last_improvement_gen = 0;

            for (int gen = 1; gen <= data.max_iter; ++gen) {
                ObjectiveKey best_before_generation = obj_key_full(best_s, data);
                double a, p_hybrid;
                _dynamic_parameters(gen - 1, data.max_iter, a, p_hybrid);
                double p_mode = _mode_probability(p_hybrid, data);
                const Solution best_snapshot = best_s;
                std::vector<Solution> old_pop = pop;
                std::vector<Solution> next_pop(p_size);
                std::vector<double> next_fit(p_size, 0.0);
                std::vector<bool> next_dirty(p_size, false);
                std::vector<int> next_changed(p_size, 0);
                std::vector<int> peer_indices(p_size, 0);
                std::vector<int> update_seeds(p_size, 0);

                for (int index = 0; index < p_size; ++index) {
                    peer_indices[index] = _tournament_peer_index(old_pop, pop_fit, index, data, global_rng, 3);
                    update_seeds[index] = static_cast<int>(global_rng());
                }

                int accepted_count = 0;
                // Native OpenMP parallel updates
                {
                    ScopedProfileTimer update_timer(data.profile, profile_registry().population_update, p_size);
                    #pragma omp parallel for reduction(+:accepted_count)
                    for (int index = 0; index < p_size; ++index) {
                        Solution out_sol;
                        double out_cost;
                        bool accepted;
                        _update_agent(
                            old_pop[index],
                            old_pop[peer_indices[index]],
                            best_snapshot,
                            pop_fit[index],
                            p_mode,
                            a,
                            gen - 1,
                            data.max_iter,
                            update_seeds[index],
                            data,
                            backend,
                            out_sol,
                            out_cost,
                            accepted
                        );
                        next_pop[index] = out_sol;
                        next_fit[index] = out_cost;
                        next_changed[index] = accepted ? 1 : 0;
                        if (accepted) {
                            accepted_count++;
                        }
                    }
                }

                for (int index = 0; index < p_size; ++index) {
                    next_dirty[index] = pop_dirty[index] || next_changed[index] != 0;
                }
                pop.swap(next_pop);
                pop_fit.swap(next_fit);
                pop_dirty.swap(next_dirty);
                const bool verify_generation = data.gpu_solution_verify_interval > 0 &&
                    gen % data.gpu_solution_verify_interval == 0;
                _refresh_population_fitness_on_gpu(
                    pop, pop_fit, pop_dirty, data, backend, verify_generation
                );

                argsort_population(pop, pop_fit, pop_argrank, data);
                if (data.objective == "lexicographic") {
                    // In lexicographic mode, track best by (NV, distance) not cost
                    ObjectiveKey best_key = obj_key_full(best_s, data);
                    for (int i = 0; i < p_size; ++i) {
                        if (!std::isfinite(pop_fit[i])) continue;
                        ObjectiveKey k = obj_key_full(pop[i], data);
                        if (is_better_key(k, best_key, data.objective)) {
                            best_key = k;
                            best_s.copy_from(pop[i]);
                            elapsed = std::chrono::high_resolution_clock::now() - stime;
                            used = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                            std::printf("Best solution update (lex): NV=%d dist=%.4f cost=%.4f\n",
                                        best_s.len(), best_key.dist, best_s.cost);
                        }
                    }
                } else {
                    if (pop[pop_argrank[0]].cost < best_s.cost) {
                        best_s.copy_from(pop[pop_argrank[0]]);
                        elapsed = std::chrono::high_resolution_clock::now() - stime;
                        used = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                        std::printf("Best solution update: %.4f\n", best_s.cost);
                    }
                }

                // Periodic deep local search
                if (gen % data.local_search_interval == 0) {
                    std::printf("Periodic deep local search on global best.\n");
                    Solution elite = best_s.clone();
                    // Perform local searches
                    do_local_search(elite, data, backend);
                    ObjectiveKey elite_key = obj_key_full(elite, data);
                    ObjectiveKey best_key  = obj_key_full(best_s, data);
                    if (is_better_key(elite_key, best_key, data.objective)) {
                        best_s.copy_from(elite);
                        if (data.objective == "lexicographic") {
                            std::printf("Best solution update (lex): NV=%d dist=%.4f cost=%.4f\n",
                                        best_s.len(), elite_key.dist, best_s.cost);
                        } else {
                            std::printf("Best solution update: %.4f\n", best_s.cost);
                        }
                    }
                    _inject_global_best(pop, pop_fit, pop_argrank, pop_dirty, best_s, data);
                }

                if (is_better_key(obj_key_full(best_s, data), best_before_generation, data.objective)) {
                    last_improvement_gen = gen;
                }

                // Stagnation diversification
                if (gen % data.stagnation_interval == 0 && gen - last_improvement_gen >= data.stagnation_interval) {
                    std::printf("Stagnation detected. Diversifying 40%% of non-elite population.\n");
                    _diversify_pop(pop, pop_fit, pop_argrank, pop_dirty, best_s, data, backend, global_rng);
                    last_improvement_gen = gen;
                }

                if (gen % 25 == 0) {
                    elapsed = std::chrono::high_resolution_clock::now() - stime;
                    used = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                    double avg_cost = mean(pop_fit);
                    std::printf("Gen: %d. a %.4f, p_hybrid %.4f, accepted %d. Avg %.4f, Best %.4f, Worst %.4f, Best vehicles %d\n",
                        gen, a, p_mode, accepted_count, avg_cost, best_s.cost, pop_fit[pop_argrank.back()], best_s.len());
                    std::printf("Gen %d done, no improvement for %d gens, already consumed %d sec\n",
                        gen, gen - last_improvement_gen, used);
                }
            }
        }

        std::printf("Run %d finishes\n", run);
        completed_runs++;
        run++;
    }

    std::printf("------------Summary-----------\n");
    auto elapsed = std::chrono::high_resolution_clock::now() - stime;
    used = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    std::printf("Total %d runs, total consumed %d sec\n", completed_runs, used);
    best_s.output(data);
    best_s.check(data);
    if (data.profile) {
        profile_print();
    }

    delete backend;
}
