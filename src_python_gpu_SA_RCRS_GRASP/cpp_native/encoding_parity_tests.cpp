#include "compute_backend.h"
#include "deterministic_rng.h"
#include "objective.h"
#include "operator.h"
#include "profile.h"
#include "solution_encoding.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
constexpr double kTol = 1e-6;

bool nearly_equal(double a, double b) {
    return std::fabs(a - b) <= kTol;
}

Data make_data(int customers) {
    Data data;
    // Parity fixtures must instantiate the requested backend explicitly; the
    // product default may otherwise route small synthetic workloads to CPU.
    data.execution_policy = "legacy";
    data.problem_name = "encoding_test";
    data.customer_num = customers;
    data.DC = 0;
    data.vehicle.capacity = 10.0;
    data.vehicle.d_cost = 2000.0;
    data.vehicle.unit_cost = 1.0;
    data.vehicle.max_num = customers + 2;
    data.start_time = 0.0;
    data.end_time = 1000.0;
    data.node.assign(customers + 1, Point{0, 0.0, 0.0, 0.0, 0.0, 1000.0});
    for (int i = 0; i <= customers; ++i) {
        data.node[i].id = i;
        data.node[i].delivery = (i == 0) ? 0.0 : 1.0;
        data.node[i].pickup = 0.0;
        data.node[i].s_time = 0.0;
        data.node[i].start = 0.0;
        data.node[i].end = 1000.0;
    }
    data.dist.assign(customers + 1, std::vector<double>(customers + 1, 0.0));
    data.time.assign(customers + 1, std::vector<double>(customers + 1, 0.0));
    for (int i = 0; i <= customers; ++i) {
        for (int j = 0; j <= customers; ++j) {
            if (i == j) {
                data.dist[i][j] = 0.0;
                data.time[i][j] = 0.0;
            } else {
                data.dist[i][j] = static_cast<double>(std::abs(i - j) + 1);
                data.time[i][j] = data.dist[i][j];
            }
        }
    }
    return data;
}

Route make_route(const Data& data, const std::vector<int>& nodes) {
    Route route(data);
    route.node_list = nodes;
    route.update(data);
    route.cal_cost(data);
    return route;
}

Solution make_solution(const Data& data, const std::vector<std::vector<int>>& routes) {
    Solution solution;
    for (const auto& nodes : routes) {
        solution.append(make_route(data, nodes));
    }
    solution.cal_cost(data);
    return solution;
}

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        return false;
    }
    return true;
}

bool compare_solution_eval(const SolutionEval& a, const SolutionEval& b, const std::string& label) {
    bool ok = true;
    ok &= expect(a.feasible == b.feasible, label + ": feasible mismatch");
    ok &= expect(a.vehicle_count == b.vehicle_count, label + ": vehicle count mismatch");
    ok &= expect(a.violation_flags == b.violation_flags, label + ": violation flags mismatch");
    ok &= expect(nearly_equal(a.distance, b.distance), label + ": distance mismatch");
    ok &= expect(nearly_equal(a.fitness, b.fitness), label + ": fitness mismatch");
    return ok;
}

bool compare_insertion_score(const InsertionScore& a, const InsertionScore& b, const std::string& label) {
    bool ok = true;
    ok &= expect(a.request_id == b.request_id, label + ": request_id mismatch");
    ok &= expect(a.feasible == b.feasible, label + ": feasible mismatch");
    ok &= expect(a.violation_flags == b.violation_flags, label + ": violation flags mismatch");
    ok &= expect(nearly_equal(a.delta_distance, b.delta_distance), label + ": delta distance mismatch");
    ok &= expect(nearly_equal(a.total_distance_after, b.total_distance_after), label + ": total distance mismatch");
    ok &= expect(nearly_equal(a.fitness_after, b.fitness_after), label + ": fitness mismatch");
    return ok;
}

bool run_objective_comparator_tests() {
    bool ok = true;
    ObjectiveValue fewer_vehicles{true, 2, 500.0, 4500.0, 0};
    ObjectiveValue shorter_distance{true, 3, 100.0, 100.0, 0};
    ObjectiveValue infeasible{false, 1, 1.0, 1.0, 1};

    ok &= expect(
        objective_better(fewer_vehicles, shorter_distance, "lexicographic"),
        "lexicographic objective did not prioritize vehicle count"
    );
    ok &= expect(
        objective_better(shorter_distance, fewer_vehicles, "weighted"),
        "weighted objective did not prioritize weighted cost"
    );
    ok &= expect(
        objective_better(fewer_vehicles, infeasible, "lexicographic"),
        "feasible objective did not dominate infeasible objective"
    );
    return ok;
}

bool run_deterministic_rng_compatibility_tests() {
    bool ok = true;
    for (std::uint32_t seed : {0u, 1u, 42u, 5489u, 0xffffffffu}) {
        std::mt19937 reference(seed);
        LegacyMt19937 compatible(seed);
        for (int i = 0; i < 10000; ++i) {
            ok &= expect(reference() == compatible(), "MT19937 engine output mismatch");
            if (!ok) return false;
        }
    }

#ifndef _MSC_VER
    const std::vector<std::pair<int, int>> ranges{
        {0, 1}, {0, 2}, {1, 5}, {-7, 9}, {0, 1000}, {0, 1000000}
    };
    for (std::uint32_t seed : {1u, 42u, 123456789u}) {
        for (const auto& bounds : ranges) {
            std::mt19937 reference(seed);
            LegacyMt19937 compatible(seed);
            std::uniform_int_distribution<int> distribution(bounds.first, bounds.second);
            for (int i = 0; i < 10000; ++i) {
                ok &= expect(
                    distribution(reference) == legacy_randint(bounds.first, bounds.second, compatible),
                    "libstdc++ uniform_int_distribution mismatch"
                );
                if (!ok) return false;
            }
        }
    }

    for (std::uint32_t seed : {1u, 42u, 987654321u}) {
        std::mt19937 reference(seed);
        LegacyMt19937 compatible(seed);
        std::uniform_real_distribution<double> distribution(-3.0, 7.0);
        for (int i = 0; i < 10000; ++i) {
            ok &= expect(
                distribution(reference) == legacy_randdouble(-3.0, 7.0, compatible),
                "libstdc++ uniform_real_distribution mismatch"
            );
            if (!ok) return false;
        }
    }

    for (int size = 0; size <= 128; ++size) {
        std::vector<int> reference(size);
        std::iota(reference.begin(), reference.end(), 0);
        std::vector<int> compatible = reference;
        std::mt19937 reference_rng(42u + static_cast<std::uint32_t>(size));
        LegacyMt19937 compatible_rng(42u + static_cast<std::uint32_t>(size));
        std::shuffle(reference.begin(), reference.end(), reference_rng);
        legacy_shuffle(compatible.begin(), compatible.end(), compatible_rng);
        ok &= expect(
            reference == compatible,
            "libstdc++ shuffle mismatch at size " + std::to_string(size)
        );
        if (!ok) return false;
    }
#endif
    return ok;
}

bool run_execution_policy_tests() {
    bool ok = true;
    Data data = make_data(8);
    data.p_size = 30;
    data.num_islands = 1;
    data.execution_policy = "adaptive";

    CpuComputeBackend cpu(data);
    WorkShape local_shape;
    local_shape.context = EVALUATION_CONTEXT_LOCAL_SEARCH;
    local_shape.item_count = 64;
    local_shape.total_nodes = 512;
    ok &= expect(
        cpu.choose_target(local_shape) == EvalTarget::CpuIncremental,
        "CPU policy did not choose incremental local-search evaluation"
    );

    std::unique_ptr<BaseComputeBackend> adaptive(create_backend(data, "cuda"));
    if (adaptive && adaptive->is_gpu_backend()) {
        ok &= expect(
            adaptive->choose_target(local_shape) == EvalTarget::CpuIncremental,
            "adaptive paper-scale policy did not keep local search on CPU"
        );
        WorkShape population_shape;
        population_shape.context = EVALUATION_CONTEXT_POPULATION;
        population_shape.item_count = 30;
        population_shape.total_nodes = 3000;
        ok &= expect(
            adaptive->choose_target(population_shape) == EvalTarget::CpuBatch,
            "adaptive paper-scale policy unexpectedly selected CUDA population evaluation"
        );
    }

    data.execution_policy = "cuda_force";
    std::unique_ptr<BaseComputeBackend> forced(create_backend(data, "cuda"));
    if (forced && forced->is_gpu_backend()) {
        ok &= expect(
            forced->choose_target(local_shape) == EvalTarget::CudaBatch,
            "cuda_force policy did not force CUDA"
        );
    }
    return ok;
}

bool run_dirty_solution_mirror_parity() {
    Data data = make_data(4);
    data.execution_policy = "cuda_force";
    std::unique_ptr<BaseComputeBackend> cuda_backend(create_backend(data, "cuda"));
    if (!cuda_backend || !cuda_backend->is_gpu_backend()) {
        std::cout << "CUDA dirty-solution mirror parity skipped: CUDA backend unavailable." << std::endl;
        return true;
    }

    std::vector<Solution> population{
        make_solution(data, {{0, 1, 2, 0}, {0, 3, 4, 0}}),
        make_solution(data, {{0, 1, 3, 0}, {0, 2, 4, 0}})
    };
    EncodedPopulation initial = encode_population_for_gpu(population, data, {true, true});
    std::vector<SolutionEval> first = cuda_backend->evaluate_solutions(initial);

    // Change both route count and compact offsets for slot 0. Slot 1 remains
    // clean and must not be uploaded or invalidated by the layout change.
    population[0] = make_solution(data, {{0, 2, 1, 3, 4, 0}});
    EncodedPopulation dirty = encode_population_for_gpu(population, data, {true, false});
    std::vector<SolutionEval> gpu = cuda_backend->evaluate_solutions(dirty);
    std::vector<SolutionEval> cpu = evaluate_encoded_population_cpu(dirty, data);

    bool ok = expect(first.size() == 2 && gpu.size() == 2, "dirty mirror result size mismatch");
    for (size_t i = 0; i < gpu.size() && i < cpu.size(); ++i) {
        ok &= compare_solution_eval(cpu[i], gpu[i], "dirty mirror " + std::to_string(i));
    }
    return ok;
}

bool run_encoded_case(
    const std::string& label,
    const Data& data,
    const Solution& solution
) {
    bool ok = true;
    EncodedPopulation encoded = encode_solution_for_gpu(solution, data);

    std::string validation_error;
    bool validation_ok = validate_encoded_solution(solution, encoded, 0, data, &validation_error);
    ok &= expect(validation_ok, label + ": validation failed: " + validation_error);

    Solution decoded = decode_solution_from_gpu(encoded, 0, data);
    ok &= expect(decoded.len() == solution.len(), label + ": decoded route count mismatch");
    for (int r = 0; r < solution.len() && r < decoded.len(); ++r) {
        ok &= expect(decoded.get(r).node_list == solution.get(r).node_list, label + ": decoded route order mismatch");
    }

    CpuComputeBackend cpu_backend(data);
    std::vector<SolutionEval> cpu_eval = cpu_backend.evaluate_solutions(encoded);
    std::vector<SolutionEval> encoded_eval = evaluate_encoded_population_cpu(encoded, data);
    ok &= expect(cpu_eval.size() == 1 && encoded_eval.size() == 1, label + ": encoded result size mismatch");
    if (!cpu_eval.empty() && !encoded_eval.empty()) {
        ok &= compare_solution_eval(cpu_eval[0], encoded_eval[0], label + ": cpu encoded");
    }

    std::unique_ptr<BaseComputeBackend> cuda_backend(create_backend(data, "cuda"));
    if (cuda_backend != nullptr && cuda_backend->is_gpu_backend()) {
        std::vector<SolutionEval> gpu_eval = cuda_backend->evaluate_solutions(encoded);
        ok &= expect(gpu_eval.size() == 1, label + ": gpu result size mismatch");
        if (!gpu_eval.empty() && !cpu_eval.empty()) {
            ok &= compare_solution_eval(cpu_eval[0], gpu_eval[0], label + ": cuda");
        }
    }

    return ok;
}

bool run_route_batch_parity(const Data& data, BaseComputeBackend* cuda_backend) {
    if (cuda_backend == nullptr || !cuda_backend->is_gpu_backend()) {
        std::cout << "CUDA route parity skipped: CUDA backend unavailable." << std::endl;
        return true;
    }

    CpuComputeBackend cpu_backend(data);
    std::vector<std::vector<int>> routes = {
        {0, 0},
        {0, 1, 0},
        {0, 1, 2, 0},
        {1, 2, 0},
    };
    std::vector<RouteEval> cpu = cpu_backend.evaluate_routes(routes);
    std::vector<RouteEval> gpu = cuda_backend->evaluate_routes(routes);
    bool ok = expect(cpu.size() == gpu.size(), "route batch result size mismatch");
    for (size_t i = 0; i < cpu.size() && i < gpu.size(); ++i) {
        ok &= expect(cpu[i].feasible == gpu[i].feasible, "route batch feasible mismatch");
        ok &= expect(nearly_equal(cpu[i].cost, gpu[i].cost), "route batch cost mismatch");
    }
    return ok;
}

bool run_insertion_batch_parity() {
    Data data = make_data(5);
    data.architecture = "hybrid_v2";
    data.profile = true;
    data.gpu_insertion_min_batch = 0;
    data.vehicle.capacity = 3.0;
    data.node[4].pickup = 2.0;
    data.node[5].end = 1.0;
    data.node[3].start = 10.0;
    data.node[3].end = 100.0;
    std::unique_ptr<BaseComputeBackend> cuda_backend(create_backend(data, "cuda"));
    if (cuda_backend == nullptr || !cuda_backend->is_gpu_backend()) {
        std::cout << "CUDA insertion batch parity skipped: CUDA backend unavailable." << std::endl;
        return true;
    }

    CpuComputeBackend cpu_backend(data);

    std::vector<std::vector<int>> routes = {
        {0, 0},
        {0, 1, 2, 0},
        {0, 4, 0}
    };
    std::vector<InsertionRequest> requests = {
        {100, 0, 3, 1, INSERTION_CONTEXT_INITIALIZATION},
        {101, 1, 3, 1, INSERTION_CONTEXT_REPAIR},
        {102, 1, 3, 2, INSERTION_CONTEXT_REPAIR},
        {103, 1, 3, 3, INSERTION_CONTEXT_REPAIR},
        {104, 1, 4, 2, INSERTION_CONTEXT_REPAIR},
        {105, 0, 5, 1, INSERTION_CONTEXT_REPAIR},
        {106, 2, 3, 1, INSERTION_CONTEXT_LOCAL_SEARCH},
        {107, 2, 3, 2, INSERTION_CONTEXT_LOCAL_SEARCH}
    };

    std::vector<InsertionScore> cpu = cpu_backend.evaluate_insertion_batch(routes, requests);
    std::vector<InsertionScore> gpu = cuda_backend->evaluate_insertion_batch(routes, requests);
    bool ok = expect(cpu.size() == gpu.size(), "insertion batch result size mismatch");
    for (size_t i = 0; i < cpu.size() && i < gpu.size(); ++i) {
        ok &= compare_insertion_score(cpu[i], gpu[i], "insertion batch " + std::to_string(i));
        ok &= expect(gpu[i].request_id == requests[i].request_id, "insertion batch ordering mismatch");
    }

    std::vector<int> legacy_feasible;
    std::vector<double> legacy_costs;
    cpu_backend.evaluate_insertions(routes[1], {3}, legacy_feasible, legacy_costs);
    for (int pos = 1; pos < static_cast<int>(routes[1].size()); ++pos) {
        InsertionRequest req{200 + pos, 1, 3, pos, INSERTION_CONTEXT_REPAIR};
        std::vector<InsertionScore> one = cpu_backend.evaluate_insertion_batch(routes, {req});
        int legacy_idx = pos - 1;
        ok &= expect(!one.empty(), "legacy insertion compare empty result");
        if (!one.empty()) {
            ok &= expect((legacy_feasible[legacy_idx] == 1) == one[0].feasible, "legacy feasible mismatch");
            ok &= expect(nearly_equal(legacy_costs[legacy_idx], one[0].fitness_after), "legacy cost mismatch");
        }
    }

    auto best_index = [](const std::vector<InsertionScore>& scores) {
        int best = -1;
        double best_cost = 1e100;
        for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
            if (scores[i].feasible && scores[i].fitness_after < best_cost) {
                best = i;
                best_cost = scores[i].fitness_after;
            }
        }
        return best;
    };
    ok &= expect(best_index(cpu) == best_index(gpu), "best insertion candidate mismatch");
    return ok;
}

bool run_concurrent_cuda_backend_stress() {
    Data data = make_data(8);
    data.vehicle.capacity = 5.0;
    data.architecture = "hybrid_v2";
    data.parallel_workers = 4;
    data.profile = true;
    data.gpu_insertion_min_batch = 256;
    profile_reset();
    std::unique_ptr<BaseComputeBackend> cuda_backend(create_backend(data, "cuda"));
    if (cuda_backend == nullptr || !cuda_backend->is_gpu_backend()) {
        std::cout << "CUDA concurrent backend stress skipped: CUDA backend unavailable." << std::endl;
        return true;
    }

    CpuComputeBackend cpu_backend(data);
    std::vector<std::vector<int>> routes = {
        {0, 1, 2, 0},
        {0, 3, 4, 0},
        {0, 5, 0},
        {0, 6, 7, 0},
        {0, 8, 0}
    };
    std::vector<RouteEval> cpu_routes = cpu_backend.evaluate_routes(routes);

    std::vector<InsertionRequest> requests;
    int request_id = 1000;
    for (int route_id = 0; route_id < static_cast<int>(routes.size()); ++route_id) {
        for (int customer = 1; customer <= data.customer_num; ++customer) {
            for (int pos = 1; pos < static_cast<int>(routes[route_id].size()); ++pos) {
                requests.push_back(InsertionRequest{request_id++, route_id, customer, pos, INSERTION_CONTEXT_REPAIR});
            }
        }
    }
    std::vector<InsertionScore> cpu_insertions = cpu_backend.evaluate_insertion_batch(routes, requests);

    Solution solution_a = make_solution(data, {{0, 1, 2, 0}, {0, 3, 4, 0}, {0, 5, 6, 0}, {0, 7, 8, 0}});
    Solution solution_b = make_solution(data, {{0, 1, 3, 5, 0}, {0, 2, 4, 6, 0}, {0, 7, 8, 0}});
    EncodedPopulation encoded = encode_population_for_gpu(std::vector<Solution>{solution_a, solution_b}, data);
    std::vector<SolutionEval> cpu_solutions = cpu_backend.evaluate_solutions(encoded);

    constexpr int kThreads = 4;
    constexpr int kIterations = 25;
    std::vector<bool> thread_ok(kThreads, true);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            try {
                for (int iter = 0; iter < kIterations; ++iter) {
                    std::vector<RouteEval> gpu_routes = cuda_backend->evaluate_routes(routes);
                    if (gpu_routes.size() != cpu_routes.size()) {
                        thread_ok[t] = false;
                        return;
                    }
                    for (size_t i = 0; i < gpu_routes.size(); ++i) {
                        if (gpu_routes[i].feasible != cpu_routes[i].feasible ||
                            !nearly_equal(gpu_routes[i].cost, cpu_routes[i].cost)) {
                            thread_ok[t] = false;
                            return;
                        }
                    }

                    std::vector<InsertionScore> gpu_insertions = cuda_backend->evaluate_insertion_batch(routes, requests);
                    if (gpu_insertions.size() != cpu_insertions.size()) {
                        thread_ok[t] = false;
                        return;
                    }
                    for (size_t i = 0; i < gpu_insertions.size(); ++i) {
                        if (gpu_insertions[i].request_id != cpu_insertions[i].request_id ||
                            gpu_insertions[i].feasible != cpu_insertions[i].feasible ||
                            gpu_insertions[i].violation_flags != cpu_insertions[i].violation_flags ||
                            !nearly_equal(gpu_insertions[i].delta_distance, cpu_insertions[i].delta_distance) ||
                            !nearly_equal(gpu_insertions[i].total_distance_after, cpu_insertions[i].total_distance_after) ||
                            !nearly_equal(gpu_insertions[i].fitness_after, cpu_insertions[i].fitness_after)) {
                            thread_ok[t] = false;
                            return;
                        }
                    }

                    std::vector<SolutionEval> gpu_solutions = cuda_backend->evaluate_solutions(encoded);
                    if (gpu_solutions.size() != cpu_solutions.size()) {
                        thread_ok[t] = false;
                        return;
                    }
                    for (size_t i = 0; i < gpu_solutions.size(); ++i) {
                        if (gpu_solutions[i].feasible != cpu_solutions[i].feasible ||
                            gpu_solutions[i].vehicle_count != cpu_solutions[i].vehicle_count ||
                            gpu_solutions[i].violation_flags != cpu_solutions[i].violation_flags ||
                            !nearly_equal(gpu_solutions[i].distance, cpu_solutions[i].distance) ||
                            !nearly_equal(gpu_solutions[i].fitness, cpu_solutions[i].fitness)) {
                            thread_ok[t] = false;
                            return;
                        }
                    }
                }
            } catch (...) {
                thread_ok[t] = false;
            }
        });
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    bool ok = true;
    for (int t = 0; t < kThreads; ++t) {
        ok &= expect(thread_ok[t], "CUDA concurrent backend stress thread " + std::to_string(t) + " failed");
    }
#ifdef _OPENMP
    data.gpu_request_broker = true;
    std::unique_ptr<BaseComputeBackend> broker_backend(create_backend(data, "cuda"));
    ok &= expect(broker_backend != nullptr && broker_backend->is_gpu_backend(),
                 "hybrid_v2 insertion broker backend unavailable");
    profile_reset();
    std::vector<bool> broker_ok(kThreads, true);
    #pragma omp parallel for num_threads(kThreads)
    for (int t = 0; t < kThreads; ++t) {
        std::vector<InsertionScore> broker_results = broker_backend->evaluate_insertion_batch(routes, requests);
        if (broker_results.size() != cpu_insertions.size()) {
            broker_ok[t] = false;
            continue;
        }
        for (size_t i = 0; i < broker_results.size(); ++i) {
            if (!compare_insertion_score(cpu_insertions[i], broker_results[i], "broker parity")) {
                broker_ok[t] = false;
                break;
            }
        }
    }
    for (int t = 0; t < kThreads; ++t) {
        ok &= expect(broker_ok[t], "hybrid_v2 insertion broker thread failed");
    }
    ok &= expect(
        profile_registry().insertion_broker_submission.calls.load(std::memory_order_relaxed) == kThreads,
        "hybrid_v2 insertion broker did not receive every OpenMP submission"
    );
#endif
    return ok;
}

bool run_insertion_threshold_behavior() {
    Data data = make_data(6);
    data.vehicle.capacity = 4.0;
    data.profile = true;

    std::vector<std::vector<int>> routes = {
        {0, 1, 2, 0},
        {0, 3, 4, 0}
    };
    std::vector<InsertionRequest> small_requests = {
        {300, 0, 5, 1, INSERTION_CONTEXT_REPAIR},
        {301, 0, 5, 2, INSERTION_CONTEXT_REPAIR},
        {302, 1, 6, 2, INSERTION_CONTEXT_REPAIR}
    };

    data.gpu_insertion_min_batch = 128;
    std::unique_ptr<BaseComputeBackend> fallback_backend(create_backend(data, "cuda"));
    if (fallback_backend == nullptr || !fallback_backend->is_gpu_backend()) {
        std::cout << "CUDA insertion threshold behavior skipped: CUDA backend unavailable." << std::endl;
        return true;
    }

    CpuComputeBackend cpu_backend(data);
    std::vector<InsertionScore> cpu_small = cpu_backend.evaluate_insertion_batch(routes, small_requests);

    profile_reset();
    std::vector<InsertionScore> fallback_scores = fallback_backend->evaluate_insertion_batch(routes, small_requests);
    bool ok = expect(fallback_scores.size() == cpu_small.size(), "threshold fallback result size mismatch");
    for (size_t i = 0; i < fallback_scores.size() && i < cpu_small.size(); ++i) {
        ok &= compare_insertion_score(cpu_small[i], fallback_scores[i], "threshold fallback " + std::to_string(i));
    }
    ok &= expect(profile_registry().cuda_insertion_cpu_fallback.calls.load(std::memory_order_relaxed) == 1,
                 "threshold fallback counter did not increment");
    ok &= expect(profile_registry().cuda_insertion_kernel.calls.load(std::memory_order_relaxed) == 0,
                 "threshold fallback unexpectedly launched CUDA insertion kernel");

    data.gpu_insertion_min_batch = 0;
    std::unique_ptr<BaseComputeBackend> forced_cuda_backend(create_backend(data, "cuda"));
    if (forced_cuda_backend == nullptr || !forced_cuda_backend->is_gpu_backend()) {
        std::cout << "CUDA insertion threshold force-kernel skipped: CUDA backend unavailable." << std::endl;
        return ok;
    }

    profile_reset();
    std::vector<InsertionScore> forced_scores = forced_cuda_backend->evaluate_insertion_batch(routes, small_requests);
    ok &= expect(forced_scores.size() == cpu_small.size(), "threshold forced CUDA result size mismatch");
    for (size_t i = 0; i < forced_scores.size() && i < cpu_small.size(); ++i) {
        ok &= compare_insertion_score(cpu_small[i], forced_scores[i], "threshold forced CUDA " + std::to_string(i));
    }
    ok &= expect(profile_registry().cuda_insertion_cpu_fallback.calls.load(std::memory_order_relaxed) == 0,
                 "threshold forced CUDA unexpectedly used CPU fallback");
    ok &= expect(profile_registry().cuda_insertion_kernel.calls.load(std::memory_order_relaxed) == 1,
                 "threshold forced CUDA did not launch insertion kernel");

    profile_reset();
    return ok;
}

bool run_local_search_route_batch_parity() {
    Data data = make_data(5);
    data.two_opt = false;
    data.two_opt_star = false;
    data.or_opt = true;
    data.or_opt_len = 2;
    data.two_exchange = false;
    data.skip_finding_lo = false;
    data.small_opts = {"oropt_single"};
    data.profile = true;
    data.gpu_route_min_batch = 128;
    data.gpu_route_min_work = 2048;
    data.gpu_insertion_min_batch = 0;

    for (int i = 0; i <= data.customer_num; ++i) {
        for (int j = 0; j <= data.customer_num; ++j) {
            data.dist[i][j] = (i == j) ? 0.0 : 10.0;
            data.time[i][j] = data.dist[i][j];
        }
    }
    auto set_edge = [&](int a, int b, double value) {
        data.dist[a][b] = value;
        data.time[a][b] = value;
    };
    set_edge(0, 1, 1.0);
    set_edge(1, 2, 1.0);
    set_edge(2, 3, 1.0);
    set_edge(3, 4, 1.0);
    set_edge(4, 5, 1.0);
    set_edge(5, 0, 1.0);

    Solution seed = make_solution(data, {{0, 1, 4, 5, 2, 3, 0}});
    CpuComputeBackend cpu_backend(data);
    Solution cpu_solution = seed.clone();
    profile_reset();
    find_local_optima(cpu_solution, data, &cpu_backend);

    std::unique_ptr<BaseComputeBackend> cuda_backend(create_backend(data, "cuda"));
    if (cuda_backend == nullptr || !cuda_backend->is_gpu_backend()) {
        std::cout << "CUDA local-search route-batch parity skipped: CUDA backend unavailable." << std::endl;
        profile_reset();
        return true;
    }

    Solution cuda_solution = seed.clone();
    profile_reset();
    find_local_optima(cuda_solution, data, cuda_backend.get());
    const long long cuda_route_batch_calls = profile_registry().feasibility_batch.calls.load(std::memory_order_relaxed);
    const long long cuda_route_fallback_calls = profile_registry().cuda_route_cpu_fallback.calls.load(std::memory_order_relaxed);
    const long long cuda_kernel_calls = profile_registry().cuda_kernel.calls.load(std::memory_order_relaxed);
    profile_reset();

    bool ok = true;
    ok &= expect(cpu_solution.len() == cuda_solution.len(), "local-search parity route count mismatch");
    ok &= expect(nearly_equal(cpu_solution.cost, cuda_solution.cost), "local-search parity cost mismatch");
    for (int r = 0; r < cpu_solution.len() && r < cuda_solution.len(); ++r) {
        ok &= expect(cpu_solution.get(r).node_list == cuda_solution.get(r).node_list,
                     "local-search parity route node order mismatch");
    }
    ok &= expect(cpu_solution.cost < seed.cost, "CPU local-search route batch did not improve the test solution");
    ok &= expect(cuda_route_batch_calls > 0, "CUDA local-search route batch evaluator was not exercised");
    ok &= expect(cuda_route_fallback_calls > 0, "CUDA local-search route batch did not use CPU fallback below threshold");
    ok &= expect(cuda_kernel_calls == 0, "CUDA local-search route batch unexpectedly launched CUDA kernel below threshold");

    data.gpu_route_min_batch = 0;
    data.gpu_route_min_work = 0;
    std::unique_ptr<BaseComputeBackend> forced_cuda_backend(create_backend(data, "cuda"));
    if (forced_cuda_backend == nullptr || !forced_cuda_backend->is_gpu_backend()) {
        std::cout << "CUDA local-search force-route-kernel parity skipped: CUDA backend unavailable." << std::endl;
        return ok;
    }

    Solution forced_cuda_solution = seed.clone();
    profile_reset();
    find_local_optima(forced_cuda_solution, data, forced_cuda_backend.get());
    const long long forced_cuda_kernel_calls = profile_registry().cuda_kernel.calls.load(std::memory_order_relaxed);
    profile_reset();

    ok &= expect(cpu_solution.len() == forced_cuda_solution.len(), "forced local-search parity route count mismatch");
    ok &= expect(nearly_equal(cpu_solution.cost, forced_cuda_solution.cost), "forced local-search parity cost mismatch");
    for (int r = 0; r < cpu_solution.len() && r < forced_cuda_solution.len(); ++r) {
        ok &= expect(cpu_solution.get(r).node_list == forced_cuda_solution.get(r).node_list,
                     "forced local-search parity route node order mismatch");
    }
    ok &= expect(forced_cuda_kernel_calls > 0, "forced CUDA local-search route batch did not launch CUDA kernel");
    return ok;
}

bool run_hybrid_v2_move_batch_parity() {
    Data data = make_data(5);
    data.architecture = "hybrid_v2";
    data.two_opt = true;
    data.two_opt_star = true;
    data.or_opt = true;
    data.or_opt_len = 2;
    data.two_exchange = true;
    data.ex_len = 2;
    data.small_opts = {"2opt", "2opt*", "oropt_single", "oropt_double", "2exchange"};
    data.profile = true;
    data.gpu_route_min_batch = 0;
    data.gpu_route_min_work = 0;

    Solution seed = make_solution(data, {{0, 1, 4, 0}, {0, 2, 3, 5, 0}});
    CpuComputeBackend cpu_backend(data);
    Solution cpu_solution = seed.clone();
    find_local_optima(cpu_solution, data, &cpu_backend);

    std::unique_ptr<BaseComputeBackend> cuda_backend(create_backend(data, "cuda"));
    if (cuda_backend == nullptr || !cuda_backend->is_gpu_backend()) {
        std::cout << "CUDA hybrid_v2 move-batch parity skipped: CUDA backend unavailable." << std::endl;
        return true;
    }

    profile_reset();
    Solution cuda_solution = seed.clone();
    find_local_optima(cuda_solution, data, cuda_backend.get());
    const long long cuda_batch_calls = profile_registry().feasibility_batch.calls.load(std::memory_order_relaxed);
    const long long cuda_kernel_calls = profile_registry().cuda_kernel.calls.load(std::memory_order_relaxed);
    profile_reset();

    bool ok = true;
    ok &= expect(cpu_solution.len() == cuda_solution.len(), "hybrid_v2 move-batch route count mismatch");
    ok &= expect(nearly_equal(cpu_solution.cost, cuda_solution.cost), "hybrid_v2 move-batch cost mismatch");
    for (int r = 0; r < cpu_solution.len() && r < cuda_solution.len(); ++r) {
        ok &= expect(cpu_solution.get(r).node_list == cuda_solution.get(r).node_list,
                     "hybrid_v2 move-batch route order mismatch");
    }
    ok &= expect(cuda_batch_calls > 0, "hybrid_v2 move-batch evaluator was not exercised");
    ok &= expect(cuda_kernel_calls > 0, "hybrid_v2 move-batch CUDA kernel was not exercised");
    return ok;
}

bool run_batched_construction_state_machine_parity() {
    Data data = make_data(8);
    data.all_delivery = 8.0;
    data.all_pickup = 0.0;
    data.max_dist = 9.0;
    data.min_dist = 1.0;
    data.pruning = false;

    std::vector<ConstructionConfig> configs(3);
    for (size_t i = 0; i < configs.size(); ++i) {
        configs[i].ksize = 3;
        configs[i].insertion_mode = i == 1 ? "rcrs" : "td";
        configs[i].lambda = 0.2 * static_cast<double>(i + 1);
        configs[i].gamma = 0.1 * static_cast<double>(i);
    }

    std::vector<Solution> sequential(configs.size());
    std::vector<Solution> batched(configs.size());
    std::vector<LegacyMt19937> sequential_rngs;
    std::vector<LegacyMt19937> batched_rngs;
    for (size_t i = 0; i < configs.size(); ++i) {
        sequential_rngs.emplace_back(1000 + static_cast<unsigned int>(i));
        batched_rngs.emplace_back(1000 + static_cast<unsigned int>(i));
    }

    CpuComputeBackend cpu(data);
    for (size_t i = 0; i < sequential.size(); ++i) {
        new_route_insertion(
            sequential[i], data, configs[i], &cpu, sequential_rngs[i]
        );
    }
    new_route_insertion_batched(batched, data, configs, &cpu, batched_rngs);

    bool ok = true;
    std::vector<LegacyMt19937::result_type> expected_next_rng(configs.size());
    for (size_t i = 0; i < sequential.size(); ++i) {
        ok &= expect(sequential[i].len() == batched[i].len(),
                     "batched construction route count mismatch");
        ok &= expect(nearly_equal(sequential[i].cost, batched[i].cost),
                     "batched construction cost mismatch");
        for (int route_idx = 0;
             route_idx < sequential[i].len() && route_idx < batched[i].len();
             ++route_idx) {
            ok &= expect(
                sequential[i].get(route_idx).node_list == batched[i].get(route_idx).node_list,
                "batched construction route order mismatch"
            );
        }
        expected_next_rng[i] = sequential_rngs[i]();
        ok &= expect(expected_next_rng[i] == batched_rngs[i](),
                     "batched construction changed RNG consumption");
    }

    data.execution_policy = "cuda_force";
    std::unique_ptr<BaseComputeBackend> cuda_backend(create_backend(data, "cuda"));
    if (cuda_backend != nullptr && cuda_backend->is_gpu_backend()) {
        std::vector<Solution> cuda_batched(configs.size());
        std::vector<LegacyMt19937> cuda_rngs;
        for (size_t i = 0; i < configs.size(); ++i) {
            cuda_rngs.emplace_back(1000 + static_cast<unsigned int>(i));
        }
        new_route_insertion_batched(
            cuda_batched, data, configs, cuda_backend.get(), cuda_rngs
        );
        for (size_t i = 0; i < sequential.size(); ++i) {
            ok &= expect(sequential[i].len() == cuda_batched[i].len(),
                         "CUDA batched construction route count mismatch");
            ok &= expect(nearly_equal(sequential[i].cost, cuda_batched[i].cost),
                         "CUDA batched construction cost mismatch");
            for (int route_idx = 0;
                 route_idx < sequential[i].len() && route_idx < cuda_batched[i].len();
                 ++route_idx) {
                ok &= expect(
                    sequential[i].get(route_idx).node_list == cuda_batched[i].get(route_idx).node_list,
                    "CUDA batched construction route order mismatch"
                );
            }
            ok &= expect(expected_next_rng[i] == cuda_rngs[i](),
                         "CUDA batched construction changed RNG consumption");
        }
    } else {
        std::cout << "CUDA batched construction state-machine parity skipped: CUDA backend unavailable."
                  << std::endl;
    }
    return ok;
}
} // namespace

int main() {
    bool ok = true;
    ok &= run_deterministic_rng_compatibility_tests();
    ok &= run_objective_comparator_tests();
    ok &= run_execution_policy_tests();

    Data route_data = make_data(2);
    std::unique_ptr<BaseComputeBackend> cuda_backend(create_backend(route_data, "cuda"));
    ok &= run_route_batch_parity(route_data, cuda_backend.get());
    ok &= run_insertion_batch_parity();
    ok &= run_concurrent_cuda_backend_stress();
    ok &= run_insertion_threshold_behavior();
    ok &= run_local_search_route_batch_parity();
    ok &= run_hybrid_v2_move_batch_parity();
    ok &= run_dirty_solution_mirror_parity();
    ok &= run_batched_construction_state_machine_parity();

    {
        Data data = make_data(0);
        ok &= run_encoded_case("empty/minimal route", data, make_solution(data, {{0, 0}}));
    }
    {
        Data data = make_data(1);
        ok &= run_encoded_case("single customer route", data, make_solution(data, {{0, 1, 0}}));
    }
    {
        Data data = make_data(4);
        ok &= run_encoded_case("normal multi-route solution", data, make_solution(data, {{0, 1, 2, 0}, {0, 3, 4, 0}}));
    }
    {
        Data data = make_data(1);
        data.node[1].start = 10.0;
        data.node[1].end = 50.0;
        ok &= run_encoded_case("time-window waiting", data, make_solution(data, {{0, 1, 0}}));
    }
    {
        Data data = make_data(1);
        data.node[1].end = 0.5;
        ok &= run_encoded_case("latest-time violation", data, make_solution(data, {{0, 1, 0}}));
    }
    {
        Data data = make_data(2);
        data.vehicle.capacity = 1.0;
        ok &= run_encoded_case("capacity violation", data, make_solution(data, {{0, 1, 2, 0}}));
    }
    {
        Data data = make_data(2);
        ok &= run_encoded_case("duplicate customer", data, make_solution(data, {{0, 1, 1, 2, 0}}));
    }
    {
        Data data = make_data(2);
        ok &= run_encoded_case("missing customer", data, make_solution(data, {{0, 1, 0}}));
    }
    {
        Data data = make_data(1);
        ok &= run_encoded_case("depot error", data, make_solution(data, {{1, 0}}));
    }
    {
        Data data = make_data(6);
        data.vehicle.capacity = 10.0;
        ok &= run_encoded_case("maximum-length route", data, make_solution(data, {{0, 1, 2, 3, 4, 5, 6, 0}}));
    }

    if (!ok) {
        std::cerr << "Encoding parity tests failed." << std::endl;
        return 1;
    }
    std::cout << "Encoding parity tests passed." << std::endl;
    return 0;
}
