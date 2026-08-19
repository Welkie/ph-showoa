#include "search_framework.h"
#include "operator.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline int randint(int low, int high, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(low, high);
    return dist(rng);
}

static inline double randdouble(double low, double high, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(low, high);
    return dist(rng);
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

static inline int _tournament_peer_index(const std::vector<double>& pop_fit, int current_index, std::mt19937& rng, int k = 3) {
    std::vector<int> candidates;
    for (size_t i = 0; i < pop_fit.size(); ++i) {
        if (static_cast<int>(i) != current_index) {
            candidates.push_back(i);
        }
    }
    if (candidates.empty()) return current_index;
    std::shuffle(candidates.begin(), candidates.end(), rng);
    
    int best_idx = candidates[0];
    double best_val = pop_fit[best_idx];
    for (int i = 1; i < k && i < candidates.size(); ++i) {
        int idx = candidates[i];
        if (pop_fit[idx] < best_val) {
            best_val = pop_fit[idx];
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

static inline Solution _build_solution_from_sequence(const std::vector<int>& sequence, const Data& data, BaseComputeBackend* backend, std::mt19937& rng) {
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

static inline Solution _li_lim_random_search(const Solution& current, const Data& data, BaseComputeBackend* backend, std::mt19937& rng) {
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

static inline void _inject_elite_routes(Solution& child, const Solution& best, const Data& data, std::mt19937& rng) {
    if (best.len() == 0) return;
    int r_idx = randint(0, best.len() - 1, rng);
    const Route& r_best = best.get(r_idx);
    
    std::set<int> best_customers;
    for (int node : r_best.node_list) {
        if (node != data.DC) best_customers.insert(node);
    }

    // Remove duplicates
    for (int i = 0; i < child.len(); ++i) {
        Route& route = child.get(i);
        std::vector<int> next_nl;
        for (int node : route.node_list) {
            if (node == data.DC || best_customers.find(node) == best_customers.end()) {
                next_nl.push_back(node);
            }
        }
        route.node_list = next_nl;
    }
    child.update(data);
    child.append(r_best);
    child.update(data);
}

static inline void _inject_elite_segments(Solution& child, const Solution& best, const Data& data, std::mt19937& rng) {
    _inject_elite_routes(child, best, data, rng); // Fallback to route injection
}

static inline Solution _woa_intensification(const Solution& current, const Solution& best, double a, const Data& data, BaseComputeBackend* backend, std::mt19937& rng) {
    double r1 = randdouble(0.0, 1.0, rng);
    double r2 = randdouble(0.0, 1.0, rng);
    double a_vector = 2.0 * a * r1 - a;
    
    if (std::abs(a_vector) < 1.0) {
        Solution child = current.clone();
        _inject_elite_routes(child, best, data, rng);
        Solution s_repaired = feasible_or_repair_algorithm_10(child, data, backend, rng);
        return s_repaired;
    }

    Solution child = _li_lim_random_search(current, data, backend, rng);
    Solution s_repaired = feasible_or_repair_algorithm_10(child, data, backend, rng);
    return s_repaired;
}

static inline Solution _guided_route_crossover(const Solution& best, const Solution& peer, const Solution& current, const Data& data, BaseComputeBackend* backend, std::mt19937& rng) {
    if (best.len() == 0) return current.clone();
    Solution child;
    child.append(best.get(randint(0, best.len() - 1, rng)));
    
    std::set<int> kept_customers;
    for (int node : child.get(0).node_list) {
        if (node != data.DC) kept_customers.insert(node);
    }

    // Insert rest from current
    for (int i = 0; i < current.len(); ++i) {
        for (int node : current.get(i).node_list) {
            if (node != data.DC && kept_customers.find(node) == kept_customers.end()) {
                Route r_new(data);
                r_new.node_list.insert(r_new.node_list.begin() + 1, node);
                r_new.update(data);
                child.append(r_new);
                kept_customers.insert(node);
            }
        }
    }
    Solution s_repaired = feasible_or_repair_algorithm_10(child, data, backend, rng);
    return s_repaired;
}

static inline bool _sa_accept(const Solution& new_solution, const Solution& current, double current_fit, int iteration, int max_iter, std::mt19937& rng) {
    double delta = new_solution.cost - current_fit;
    if (delta <= 0.001) return true;
    double temperature = max_iter > 0 ? 1.0 - (static_cast<double>(iteration) / max_iter) : 0.0;
    double denominator = 1e-6 + temperature * std::abs(current_fit);
    double probability = std::exp(-delta / denominator);
    return randdouble(0.0, 1.0, rng) < probability;
}

static inline void _update_agent(AgentUpdateTask& task, const Data& data, BaseComputeBackend* backend, Solution& out_sol, double& out_cost, bool& out_accepted) {
    std::mt19937 rng(task.seed);
    Solution new_solution;

    if (randdouble(0.0, 1.0, rng) < task.p_hybrid) {
        new_solution = _guided_route_crossover(task.best, task.peer, task.current, data, backend, rng);
    } else {
        new_solution = _woa_intensification(task.current, task.best, task.a, data, backend, rng);
    }

    new_solution.cal_cost(data);
    bool accepted = _sa_accept(new_solution, task.current, task.current_fit, task.iteration, task.max_iter, rng);
    if (accepted) {
        out_sol = new_solution;
        out_cost = new_solution.cost;
        out_accepted = true;
    } else {
        out_sol = task.current;
        out_cost = task.current_fit;
        out_accepted = false;
    }
}

static inline void _diversify_pop(std::vector<Solution>& pop, std::vector<double>& pop_fit, std::vector<int>& pop_argrank, const Solution& best_s, const Data& data, BaseComputeBackend* backend, std::mt19937& rng) {
    argsort(pop_fit, pop_argrank);
    int elite_index = pop_argrank[0];
    pop[elite_index].copy_from(best_s);
    pop_fit[elite_index] = best_s.cost;

    int diversify_count = static_cast<int>(std::round(pop.size() * data.diversify_ratio));
    for (int i = 1; i <= diversify_count && i < pop.size(); ++i) {
        int idx = pop_argrank[i];
        Solution candidate = pop[idx].clone();
        random_removal(candidate, data, rng);
        regret_insertion(candidate, data, backend, rng);
        pop[idx].copy_from(candidate);
        pop_fit[idx] = candidate.cost;
    }
    argsort(pop_fit, pop_argrank);
}

static inline void _inject_global_best(std::vector<Solution>& pop, std::vector<double>& pop_fit, std::vector<int>& pop_argrank, const Solution& best_s) {
    argsort(pop_fit, pop_argrank);
    int worst_index = pop_argrank.back();
    pop[worst_index].copy_from(best_s);
    pop_fit[worst_index] = best_s.cost;
    argsort(pop_fit, pop_argrank);
}

void search_framework(Data& data, Solution& best_s) {
    int p_size = data.p_size;
    std::vector<Solution> pop(p_size);
    std::vector<double> pop_fit(p_size, 0.0);
    std::vector<int> pop_argrank(p_size, 0);

    auto stime = std::chrono::high_resolution_clock::now();
    int used = 0;
    int run = 1;
    int completed_runs = 0;

    BaseComputeBackend* backend = create_backend(data, data.compute_backend);
    std::mt19937 global_rng(data.seed);
    int omp_threads = _configure_omp_threads(data);
    std::printf("OpenMP threads: %d\n", omp_threads);

    while (run <= data.runs) {
        std::printf("---------------------------------Run %d---------------------------\n", run);
        
        // ── Algorithm 2: Population Initialisation ────────────────────────────
        // Three modes controlled by data.init:
        //   "rcrs_grasp" / "rcg" → RCRS-GRASP hybrid (RCG framework, new default)
        //   "sa"                 → legacy Simulated Annealing post-processing
        //   anything else        → pure RCRS greedy (new_route_insertion only)
        std::printf("Initialization, using %s method\n", data.init.c_str());

        // Pre-compute α range for GRASP: each individual draws its own α from
        // [alpha_lo, alpha_hi] to ensure population-level diversity.
        double alpha_lo = 0.10;
        double alpha_hi = 0.40;

        for (int i = 0; i < p_size; ++i) {
            pop[i].clear(data);
            std::mt19937 seed_rng(data.seed + 100000 + run * 1000 + i);

            if (data.init == "rcrs_grasp" || data.init == "rcg") {
                // Per-individual α drawn from [alpha_lo, alpha_hi]
                double ind_alpha = randdouble(alpha_lo, alpha_hi, seed_rng);
                pop[i] = rcrs_grasp_initialization(data, backend, seed_rng, ind_alpha);
            } else {
                // RCRS greedy base construction
                new_route_insertion(pop[i], data, backend, seed_rng);
                if (data.init == "sa") {
                    pop[i] = _sa_initialization(pop[i], data, backend, seed_rng);
                }
            }

            pop_fit[i] = pop[i].cost;
        }
        argsort(pop_fit, pop_argrank);
        std::printf("Initialization done.\n");

        auto elapsed = std::chrono::high_resolution_clock::now() - stime;
        used = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        std::printf("already consumed %d sec\n", used);

        if (pop[pop_argrank[0]].cost < best_s.cost) {
            best_s.copy_from(pop[pop_argrank[0]]);
            std::printf("Best solution update: %.4f\n", best_s.cost);
        }

        int last_improvement_gen = 0;

        for (int gen = 1; gen <= data.max_iter; ++gen) {
            double best_before_generation = best_s.cost;
            double a, p_hybrid;
            _dynamic_parameters(gen - 1, data.max_iter, a, p_hybrid);
            double p_mode = _mode_probability(p_hybrid, data);
            
            Solution best_snapshot = best_s.clone();
            std::vector<AgentUpdateTask> tasks(p_size);

            for (int index = 0; index < p_size; ++index) {
                int peer_index = _tournament_peer_index(pop_fit, index, global_rng, 3);
                tasks[index] = AgentUpdateTask{
                    index,
                    pop[index].clone(),
                    pop[peer_index].clone(),
                    best_snapshot.clone(),
                    pop_fit[index],
                    p_mode,
                    a,
                    gen - 1,
                    data.max_iter,
                    static_cast<int>(global_rng())
                };
            }

            int accepted_count = 0;
            // Native OpenMP parallel updates
            #pragma omp parallel for reduction(+:accepted_count)
            for (int index = 0; index < p_size; ++index) {
                Solution out_sol;
                double out_cost;
                bool accepted;
                _update_agent(tasks[index], data, backend, out_sol, out_cost, accepted);
                pop[index] = out_sol;
                pop_fit[index] = out_cost;
                if (accepted) {
                    accepted_count++;
                }
            }

            argsort(pop_fit, pop_argrank);
            if (pop[pop_argrank[0]].cost < best_s.cost) {
                best_s.copy_from(pop[pop_argrank[0]]);
                elapsed = std::chrono::high_resolution_clock::now() - stime;
                used = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                std::printf("Best solution update: %.4f\n", best_s.cost);
            }

            // Periodic deep local search
            if (gen % data.local_search_interval == 0) {
                Solution elite = best_s.clone();
                
                // Perform local searches
                do_local_search(elite, data, backend);
                if (elite.cost < best_s.cost) {
                    best_s.copy_from(elite);
                    std::printf("Best solution update: %.4f\n", best_s.cost);
                }
                _inject_global_best(pop, pop_fit, pop_argrank, best_s);
            }

            if (best_s.cost < best_before_generation) {
                last_improvement_gen = gen;
            }

            // Stagnation diversification
            if (gen % data.stagnation_interval == 0 && gen - last_improvement_gen >= data.stagnation_interval) {
                _diversify_pop(pop, pop_fit, pop_argrank, best_s, data, backend, global_rng);
                last_improvement_gen = gen;
            }

            if (gen % 25 == 0) {
                elapsed = std::chrono::high_resolution_clock::now() - stime;
                used = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                double avg_cost = mean(pop_fit);
                std::printf("[Gen %4d] a=%.3f ph=%.3f acc=%d | avg=%.1f best=%.4f(%dv) worst=%.1f | no_imp=%d t=%ds\n",
                    gen, a, p_mode, accepted_count, avg_cost,
                    best_s.cost, best_s.len(), pop_fit[pop_argrank.back()],
                    gen - last_improvement_gen, used);
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

    delete backend;
}
