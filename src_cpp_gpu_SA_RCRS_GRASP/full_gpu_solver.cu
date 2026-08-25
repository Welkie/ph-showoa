#include "full_gpu_solver.h"

#include "deterministic_rng.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << operation << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { if (pointer_) cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t count) {
        if (pointer_) cudaFree(pointer_);
        pointer_ = nullptr;
        count_ = count;
        if (count > 0) cuda_check(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc(full_gpu)");
    }
    T* get() { return pointer_; }
    const T* get() const { return pointer_; }
    std::size_t size() const { return count_; }

private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

struct DeviceProblem {
    int customer_count;
    int depot;
    int stride;
    double capacity;
    double start_time;
    double dispatch_cost;
    double unit_cost;
    double all_delivery;
    double all_pickup;
    double max_distance;
    double min_distance;
    const double* delivery;
    const double* pickup;
    const double* earliest;
    const double* latest;
    const double* service;
    const double* distance;
    const double* travel_time;
    const unsigned char* pruning;
    const double* related;
    const int* related_rank;
    int pruning_enabled;
};

struct DeviceConstructionConfig {
    int ksize;
    int insertion_mode_td;
    double lambda;
    double gamma;
};

struct DevicePopulation {
    int solution_count;
    int max_routes;
    int node_capacity;
    int* nodes;
    int* route_offsets;
    int* route_lengths;
    int* route_counts;
    int* node_counts;
    double* costs;
    double* distances;
};

struct DeviceConstructionScratch {
    int route_capacity;
    int customer_capacity;
    int* sample_pool;
    int* unrouted;
    int* route_nodes;
    int* candidate_nodes;
    int* candidate_nodes_2;
    int* flags;
    double* load;
    double* cd;
    double* cp;
    double* rd;
    double* rp;
};

DevicePopulation slice_population(DevicePopulation population, int first, int count) {
    DevicePopulation slice = population;
    slice.solution_count = count;
    slice.nodes += first * population.node_capacity;
    slice.route_offsets += first * population.max_routes;
    slice.route_lengths += first * population.max_routes;
    slice.route_counts += first;
    slice.node_counts += first;
    slice.costs += first;
    slice.distances += first;
    return slice;
}

DeviceConstructionScratch slice_scratch(DeviceConstructionScratch scratch, int first) {
    DeviceConstructionScratch slice = scratch;
    slice.sample_pool += first * scratch.customer_capacity;
    slice.unrouted += first * scratch.customer_capacity;
    slice.route_nodes += first * scratch.route_capacity;
    slice.candidate_nodes += first * scratch.route_capacity;
    slice.candidate_nodes_2 += first * scratch.route_capacity;
    slice.flags += first * (scratch.customer_capacity + 1);
    slice.load += first * scratch.route_capacity;
    slice.cd += first * scratch.route_capacity;
    slice.cp += first * scratch.route_capacity;
    slice.rd += first * scratch.route_capacity;
    slice.rp += first * scratch.route_capacity;
    return slice;
}

__device__ inline int* solution_nodes(DevicePopulation population, int solution_id) {
    return population.nodes + solution_id * population.node_capacity;
}

__device__ inline int* solution_offsets(DevicePopulation population, int solution_id) {
    return population.route_offsets + solution_id * population.max_routes;
}

__device__ inline int* solution_lengths(DevicePopulation population, int solution_id) {
    return population.route_lengths + solution_id * population.max_routes;
}

__device__ void clear_solution(DevicePopulation population, int solution_id) {
    population.route_counts[solution_id] = 0;
    population.node_counts[solution_id] = 0;
    population.costs[solution_id] = 0.0;
    population.distances[solution_id] = 0.0;
}

__device__ void copy_solution(
    DevicePopulation destination,
    int destination_id,
    DevicePopulation source,
    int source_id
) {
    const int route_count = source.route_counts[source_id];
    const int node_count = source.node_counts[source_id];
    int* destination_nodes = solution_nodes(destination, destination_id);
    int* source_nodes_ptr = solution_nodes(source, source_id);
    int* destination_offsets = solution_offsets(destination, destination_id);
    int* source_offsets_ptr = solution_offsets(source, source_id);
    int* destination_lengths = solution_lengths(destination, destination_id);
    int* source_lengths_ptr = solution_lengths(source, source_id);
    for (int i = 0; i < node_count; ++i) destination_nodes[i] = source_nodes_ptr[i];
    for (int i = 0; i < route_count; ++i) {
        destination_offsets[i] = source_offsets_ptr[i];
        destination_lengths[i] = source_lengths_ptr[i];
    }
    destination.route_counts[destination_id] = route_count;
    destination.node_counts[destination_id] = node_count;
    destination.costs[destination_id] = source.costs[source_id];
    destination.distances[destination_id] = source.distances[source_id];
}

__device__ inline void warp_reduce_min(double& val, int& idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        double other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val < val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

__device__ bool evaluate_route(
    const int* route,
    int length,
    const DeviceProblem& problem,
    double* out_distance
) {
    // Branchless Mask-based Tensorization for GPU Warp Execution
    double valid_mask = (length >= 2 && route[0] == problem.depot && route[length - 1] == problem.depot) ? 1.0 : 0.0;
    double load = 0.0;
    for (int i = 1; i < length - 1; ++i) {
        load += problem.delivery[route[i]];
    }
    valid_mask *= (load <= problem.capacity ? 1.0 : 0.0);

    double time_value = problem.start_time;
    double distance_value = 0.0;
    int previous = route[0];
    for (int i = 1; i < length; ++i) {
        const int node = route[i];
        load = load - problem.delivery[node] + problem.pickup[node];
        valid_mask *= (load >= 0.0 && load <= problem.capacity ? 1.0 : 0.0);
        time_value += problem.travel_time[previous * problem.stride + node];
        valid_mask *= (time_value <= problem.latest[node] ? 1.0 : 0.0);
        time_value = fmax(time_value, problem.earliest[node]) + problem.service[node];
        distance_value += problem.distance[previous * problem.stride + node];
        previous = node;
    }
    if (out_distance) *out_distance = (valid_mask > 0.5) ? distance_value : INFINITY;
    return valid_mask > 0.5;
}

__device__ bool append_route(
    DevicePopulation population,
    int solution_id,
    const int* route,
    int route_length,
    const DeviceProblem& problem
) {
    const int route_count = population.route_counts[solution_id];
    const int node_count = population.node_counts[solution_id];
    if (route_count >= population.max_routes ||
        node_count + route_length > population.node_capacity) {
        return false;
    }
    double route_distance = 0.0;
    if (!evaluate_route(route, route_length, problem, &route_distance)) return false;
    int* nodes = solution_nodes(population, solution_id);
    for (int i = 0; i < route_length; ++i) nodes[node_count + i] = route[i];
    solution_offsets(population, solution_id)[route_count] = node_count;
    solution_lengths(population, solution_id)[route_count] = route_length;
    population.route_counts[solution_id] = route_count + 1;
    population.node_counts[solution_id] = node_count + route_length;
    population.distances[solution_id] += route_distance;
    population.costs[solution_id] += problem.dispatch_cost + route_distance * problem.unit_cost;
    return true;
}

__device__ bool append_route_unchecked(
    DevicePopulation population,
    int solution_id,
    const int* route,
    int route_length,
    const DeviceProblem& problem
) {
    const int route_count = population.route_counts[solution_id];
    const int node_count = population.node_counts[solution_id];
    if (route_count >= population.max_routes ||
        node_count + route_length > population.node_capacity) {
        return false;
    }
    double route_distance = 0.0;
    for (int i = 1; i < route_length; ++i) {
        route_distance += problem.distance[route[i - 1] * problem.stride + route[i]];
    }
    int* nodes = solution_nodes(population, solution_id);
    for (int i = 0; i < route_length; ++i) nodes[node_count + i] = route[i];
    solution_offsets(population, solution_id)[route_count] = node_count;
    solution_lengths(population, solution_id)[route_count] = route_length;
    population.route_counts[solution_id] = route_count + 1;
    population.node_counts[solution_id] = node_count + route_length;
    population.distances[solution_id] += route_distance;
    if (route_length > 2) {
        population.costs[solution_id] +=
            problem.dispatch_cost + route_distance * problem.unit_cost;
    }
    return true;
}

__device__ bool rebuild_solution_with_routes_unchecked(
    DevicePopulation destination,
    int solution_id,
    DevicePopulation source,
    int first_route_id,
    const int* first_route,
    int first_length,
    int second_route_id,
    const int* second_route,
    int second_length,
    const DeviceProblem& problem,
    bool remove_empty
) {
    clear_solution(destination, solution_id);
    const int route_count = source.route_counts[solution_id];
    const int* source_node_data = solution_nodes(source, solution_id);
    const int* source_offsets = solution_offsets(source, solution_id);
    const int* source_lengths = solution_lengths(source, solution_id);
    for (int route_id = 0; route_id < route_count; ++route_id) {
        const int* route = source_node_data + source_offsets[route_id];
        int route_length = source_lengths[route_id];
        if (route_id == first_route_id) {
            route = first_route;
            route_length = first_length;
        } else if (route_id == second_route_id) {
            route = second_route;
            route_length = second_length;
        }
        if (remove_empty && route_length <= 2) continue;
        if (!append_route_unchecked(destination, solution_id, route, route_length, problem)) return false;
    }
    return true;
}

__device__ bool rebuild_solution_with_routes(
    DevicePopulation destination,
    int solution_id,
    DevicePopulation source,
    int first_route_id,
    const int* first_route,
    int first_length,
    int second_route_id,
    const int* second_route,
    int second_length,
    const DeviceProblem& problem
) {
    clear_solution(destination, solution_id);
    const int route_count = source.route_counts[solution_id];
    const int* source_node_data = solution_nodes(source, solution_id);
    const int* source_offsets = solution_offsets(source, solution_id);
    const int* source_lengths = solution_lengths(source, solution_id);
    for (int route_id = 0; route_id < route_count; ++route_id) {
        const int* route = source_node_data + source_offsets[route_id];
        int route_length = source_lengths[route_id];
        if (route_id == first_route_id) {
            route = first_route;
            route_length = first_length;
        } else if (route_id == second_route_id) {
            route = second_route;
            route_length = second_length;
        }
        // Solution::update removes routes that contain only the two depots.
        if (route_length <= 2) continue;
        if (!append_route(destination, solution_id, route, route_length, problem)) return false;
    }
    return true;
}

__device__ double construction_tc(
    int solution_id,
    const int* route,
    int route_length,
    int inserted_node,
    int insert_position,
    double unrouted_delivery,
    double unrouted_pickup,
    const DeviceProblem& problem,
    DeviceConstructionScratch scratch
) {
    const int base = solution_id * scratch.route_capacity;
    int* nodes = scratch.candidate_nodes + base;
    double* load = scratch.load + base;
    double* cd = scratch.cd + base;
    double* cp = scratch.cp + base;
    double* rd = scratch.rd + base;
    double* rp = scratch.rp + base;
    const int new_length = route_length + 1;
    for (int i = 0; i < new_length; ++i) {
        if (i == insert_position) nodes[i] = inserted_node;
        else if (i < insert_position) nodes[i] = route[i];
        else nodes[i] = route[i - 1];
        load[i] = cd[i] = cp[i] = rd[i] = rp[i] = 0.0;
    }

    double route_delivery = 0.0;
    double route_pickup = 0.0;
    for (int i = 1; i < new_length - 1; ++i) {
        route_delivery += problem.delivery[nodes[i]];
        route_pickup += problem.pickup[nodes[i]];
    }
    load[0] = route_delivery;
    for (int i = 1; i < new_length; ++i) {
        const int node = nodes[i];
        load[i] = load[i - 1] - problem.delivery[node] + problem.pickup[node];
        cd[i] = cd[i - 1] + problem.distance[nodes[i - 1] * problem.stride + node];
    }
    cp[new_length - 1] = 0.0;
    for (int i = new_length - 1; i > 0; --i) {
        cp[i - 1] = cp[i] + problem.distance[nodes[i - 1] * problem.stride + nodes[i]];
    }
    rd[0] = problem.capacity - route_delivery;
    if (new_length >= 2) rp[new_length - 2] = problem.capacity - route_pickup;
    for (int i = 1; i < new_length - 1; ++i) {
        rd[i] = fmin(rd[i - 1], problem.capacity - load[i]);
        rp[new_length - 2 - i] = fmin(
            rp[new_length - 1 - i],
            problem.capacity - load[new_length - 2 - i]
        );
    }
    double rdt_u = 0.0;
    double rdt_d = 0.0;
    double rpt_u = 0.0;
    double rpt_d = 0.0;
    for (int i = 0; i < new_length - 1; ++i) {
        rdt_u += rd[i] * cd[i + 1];
        rdt_d += cd[i + 1];
        rpt_u += rp[i] * cp[i];
        rpt_d += cp[i];
    }
    const double rdt = rdt_d > 0.0 ? rdt_u / rdt_d : 0.0;
    const double rpt = rpt_d > 0.0 ? rpt_u / rpt_d : 0.0;
    const double delivery_term = problem.all_delivery > 0.0
        ? (unrouted_delivery / problem.all_delivery) * (1.0 - rdt / problem.capacity)
        : 0.0;
    const double pickup_term = problem.all_pickup > 0.0
        ? (unrouted_pickup / problem.all_pickup) * (1.0 - rpt / problem.capacity)
        : 0.0;
    return delivery_term + pickup_term;
}

__device__ double construction_metric(
    int solution_id,
    const int* route,
    int route_length,
    int customer,
    int position,
    double unrouted_delivery,
    double unrouted_pickup,
    const DeviceProblem& problem,
    const DeviceConstructionConfig& config,
    DeviceConstructionScratch scratch
) {
    const int previous = route[position - 1];
    const int successor = route[position];
    const double distance_delta =
        problem.distance[previous * problem.stride + customer] +
        problem.distance[customer * problem.stride + successor] -
        problem.distance[previous * problem.stride + successor];
    if (config.insertion_mode_td) return distance_delta;
    const double tc = construction_tc(
        solution_id, route, route_length, customer, position,
        unrouted_delivery, unrouted_pickup, problem, scratch
    );
    const double round_trip =
        problem.distance[problem.depot * problem.stride + customer] +
        problem.distance[customer * problem.stride + problem.depot];
    return distance_delta + config.lambda * tc *
        (2.0 * problem.max_distance - problem.min_distance) -
        config.gamma * round_trip;
}

__device__ bool choose_construction_insertion(
    int solution_id,
    const int* route,
    int route_length,
    const int* unrouted,
    int unrouted_count,
    double unrouted_delivery,
    double unrouted_pickup,
    const DeviceProblem& problem,
    const DeviceConstructionConfig& config,
    DeviceConstructionScratch scratch,
    int* selected_unrouted,
    int* selected_position
) {
    const int candidate_base = solution_id * scratch.route_capacity;
    int* candidate = scratch.candidate_nodes + candidate_base;
    int best_unrouted = -1;
    int best_position = -1;
    double best_metric = INFINITY;
    for (int customer_index = 0; customer_index < unrouted_count; ++customer_index) {
        const int customer = unrouted[customer_index];
        for (int position = 1; position < route_length; ++position) {
            if (problem.pruning_enabled) {
                const int previous = route[position - 1];
                const int successor = route[position];
                if (!problem.pruning[previous * problem.stride + customer] ||
                    !problem.pruning[customer * problem.stride + successor]) {
                    continue;
                }
            }
            for (int i = 0; i < route_length + 1; ++i) {
                if (i == position) candidate[i] = customer;
                else if (i < position) candidate[i] = route[i];
                else candidate[i] = route[i - 1];
            }
            if (!evaluate_route(candidate, route_length + 1, problem, nullptr)) continue;
            const double metric = construction_metric(
                solution_id, route, route_length, customer, position,
                unrouted_delivery, unrouted_pickup, problem, config, scratch
            );
            if (best_unrouted == -1 || metric < best_metric - 1e-9) {
                best_unrouted = customer_index;
                best_position = position;
                best_metric = metric;
            }
        }
    }
    *selected_unrouted = best_unrouted;
    *selected_position = best_position;
    return best_unrouted >= 0;
}

__device__ bool construct_solution(
    int solution_id,
    int initial_customer,
    DevicePopulation branch,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem,
    const DeviceConstructionConfig& config,
    LegacyMt19937& rng
) {
    clear_solution(branch, solution_id);
    int* unrouted = scratch.unrouted + solution_id * scratch.customer_capacity;
    int unrouted_count = 0;
    double unrouted_delivery = problem.all_delivery;
    double unrouted_pickup = problem.all_pickup;
    for (int customer = 1; customer <= problem.customer_count; ++customer) {
        if (customer != problem.depot) unrouted[unrouted_count++] = customer;
    }
    int* route = scratch.route_nodes + solution_id * scratch.route_capacity;
    bool first_route = true;
    while (unrouted_count > 0) {
        route[0] = problem.depot;
        route[1] = problem.depot;
        int route_length = 2;
        int selected = -1;
        if (first_route) {
            for (int i = 0; i < unrouted_count; ++i) {
                if (unrouted[i] == initial_customer) {
                    selected = i;
                    break;
                }
            }
        }
        if (selected < 0) selected = legacy_randint(0, unrouted_count - 1, rng);
        const int first_customer = unrouted[selected];
        route[2] = route[1];
        route[1] = first_customer;
        route_length = 3;
        unrouted_delivery -= problem.delivery[first_customer];
        unrouted_pickup -= problem.pickup[first_customer];
        for (int i = selected; i + 1 < unrouted_count; ++i) unrouted[i] = unrouted[i + 1];
        --unrouted_count;
        first_route = false;

        int customer_index = -1;
        int position = -1;
        while (choose_construction_insertion(
            solution_id, route, route_length, unrouted, unrouted_count,
            unrouted_delivery, unrouted_pickup, problem, config, scratch,
            &customer_index, &position
        )) {
            const int customer = unrouted[customer_index];
            for (int i = route_length; i > position; --i) route[i] = route[i - 1];
            route[position] = customer;
            ++route_length;
            unrouted_delivery -= problem.delivery[customer];
            unrouted_pickup -= problem.pickup[customer];
            for (int i = customer_index; i + 1 < unrouted_count; ++i) {
                unrouted[i] = unrouted[i + 1];
            }
            --unrouted_count;
        }
        if (!append_route(branch, solution_id, route, route_length, problem)) return false;
    }
    return true;
}

__global__ void initialize_population_kernel(
    DevicePopulation population,
    DevicePopulation branch,
    DeviceConstructionScratch scratch,
    DeviceProblem problem,
    const DeviceConstructionConfig* configs,
    LegacyMt19937* rng_states,
    int* status
) {
    const int solution_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (solution_id >= population.solution_count) return;
    LegacyMt19937 rng = rng_states[solution_id];
    const DeviceConstructionConfig config = configs[solution_id];
    int* sample_pool = scratch.sample_pool + solution_id * scratch.customer_capacity;
    for (int i = 0; i < problem.customer_count; ++i) sample_pool[i] = i + 1;
    const int sample_count = min(
        config.ksize > 0 ? config.ksize : problem.customer_count,
        problem.customer_count
    );
    clear_solution(population, solution_id);
    double best_cost = INFINITY;
    for (int sample = 0; sample < sample_count; ++sample) {
        const int remaining = problem.customer_count - sample;
        const int selected = legacy_randint(0, remaining - 1, rng);
        const int initial_customer = sample_pool[selected];
        const int tail = problem.customer_count - 1 - sample;
        const int temporary = sample_pool[selected];
        sample_pool[selected] = sample_pool[tail];
        sample_pool[tail] = temporary;
        if (!construct_solution(
            solution_id, initial_customer, branch, scratch, problem, config, rng
        )) {
            status[solution_id] = 1;
            rng_states[solution_id] = rng;
            return;
        }
        if (branch.costs[solution_id] < best_cost - 0.001) {
            copy_solution(population, solution_id, branch, solution_id);
            best_cost = branch.costs[solution_id];
        }
    }
    rng_states[solution_id] = rng;
    status[solution_id] = 0;
}

__device__ void copy_route_to_scratch(
    DevicePopulation population,
    int solution_id,
    int route_id,
    int* destination,
    int* out_length
) {
    const int offset = solution_offsets(population, solution_id)[route_id];
    const int length = solution_lengths(population, solution_id)[route_id];
    const int* source = solution_nodes(population, solution_id) + offset;
    for (int i = 0; i < length; ++i) destination[i] = source[i];
    *out_length = length;
}

__device__ void erase_route_node(int* route, int* length, int index) {
    for (int i = index; i + 1 < *length; ++i) route[i] = route[i + 1];
    --(*length);
}

__device__ void insert_route_node(int* route, int* length, int index, int node) {
    for (int i = *length; i > index; --i) route[i] = route[i - 1];
    route[index] = node;
    ++(*length);
}

__global__ void simulated_annealing_initialization_kernel(
    DevicePopulation population,
    DevicePopulation candidate,
    DevicePopulation best,
    DeviceConstructionScratch scratch,
    DeviceProblem problem,
    LegacyMt19937* rng_states,
    int* status
) {
    const int solution_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (solution_id >= population.solution_count) return;
    LegacyMt19937 rng = rng_states[solution_id];
    copy_solution(best, solution_id, population, solution_id);
    double best_cost = population.costs[solution_id];
    int* first = scratch.candidate_nodes + solution_id * scratch.route_capacity;
    int* second = scratch.candidate_nodes_2 + solution_id * scratch.route_capacity;

    double temperature = 100.0;
    while (temperature > 0.1) {
        for (int iteration = 0; iteration < 100; ++iteration) {
            const int move_type = legacy_randint(1, 5, rng);
            const int route_count = population.route_counts[solution_id];
            if (move_type <= 3) {
                if (route_count == 0) continue;
                const int route_id = legacy_randint(0, route_count - 1, rng);
                int length = 0;
                copy_route_to_scratch(population, solution_id, route_id, first, &length);
                if (length < 4) continue;
                if (move_type == 1) {
                    const int first_index = legacy_randint(1, length - 2, rng);
                    int second_index = legacy_randint(1, length - 2, rng);
                    while (first_index == second_index) {
                        second_index = legacy_randint(1, length - 2, rng);
                    }
                    const int temporary = first[first_index];
                    first[first_index] = first[second_index];
                    first[second_index] = temporary;
                } else if (move_type == 2) {
                    const int first_index = legacy_randint(1, length - 2, rng);
                    const int node = first[first_index];
                    erase_route_node(first, &length, first_index);
                    const int second_index = legacy_randint(1, length - 1, rng);
                    insert_route_node(first, &length, second_index, node);
                } else {
                    int first_index = legacy_randint(1, length - 2, rng);
                    int second_index = legacy_randint(1, length - 2, rng);
                    if (first_index > second_index) {
                        const int temporary = first_index;
                        first_index = second_index;
                        second_index = temporary;
                    }
                    while (first_index < second_index) {
                        const int temporary = first[first_index];
                        first[first_index++] = first[second_index];
                        first[second_index--] = temporary;
                    }
                }
                if (!evaluate_route(first, length, problem, nullptr)) continue;
                if (!rebuild_solution_with_routes(
                    candidate, solution_id, population, route_id, first, length,
                    -1, nullptr, 0, problem
                )) {
                    status[solution_id] = 2;
                    rng_states[solution_id] = rng;
                    return;
                }
            } else {
                if (route_count < 2) continue;
                const int first_route_id = legacy_randint(0, route_count - 1, rng);
                int second_route_id = legacy_randint(0, route_count - 1, rng);
                while (first_route_id == second_route_id) {
                    second_route_id = legacy_randint(0, route_count - 1, rng);
                }
                int first_length = 0;
                int second_length = 0;
                copy_route_to_scratch(population, solution_id, first_route_id, first, &first_length);
                copy_route_to_scratch(population, solution_id, second_route_id, second, &second_length);
                if (move_type == 4) {
                    if (first_length < 3) continue;
                    const int first_index = legacy_randint(1, first_length - 2, rng);
                    const int node = first[first_index];
                    erase_route_node(first, &first_length, first_index);
                    const int second_index = legacy_randint(1, second_length - 1, rng);
                    insert_route_node(second, &second_length, second_index, node);
                } else {
                    if (first_length < 3 || second_length < 3) continue;
                    const int first_index = legacy_randint(1, first_length - 2, rng);
                    const int second_index = legacy_randint(1, second_length - 2, rng);
                    const int temporary = first[first_index];
                    first[first_index] = second[second_index];
                    second[second_index] = temporary;
                }
                if (!evaluate_route(first, first_length, problem, nullptr) ||
                    !evaluate_route(second, second_length, problem, nullptr)) {
                    continue;
                }
                if (!rebuild_solution_with_routes(
                    candidate, solution_id, population,
                    first_route_id, first, first_length,
                    second_route_id, second, second_length, problem
                )) {
                    status[solution_id] = 2;
                    rng_states[solution_id] = rng;
                    return;
                }
            }

            const double delta = candidate.costs[solution_id] - population.costs[solution_id];
            const bool accept = delta < 0.0 ||
                legacy_randdouble(0.0, 1.0, rng) <
                    exp(-delta / (1e-6 + temperature * fabs(population.costs[solution_id])));
            if (accept) {
                copy_solution(population, solution_id, candidate, solution_id);
                if (population.costs[solution_id] < best_cost) {
                    best_cost = population.costs[solution_id];
                    copy_solution(best, solution_id, population, solution_id);
                }
            }
        }
        temperature *= 0.95;
    }
    copy_solution(population, solution_id, best, solution_id);
    rng_states[solution_id] = rng;
}

__device__ bool route_capacity_only(
    const int* route,
    int length,
    const DeviceProblem& problem
) {
    if (length <= 2) return true;
    double load = 0.0;
    for (int i = 0; i < length; ++i) load += problem.delivery[route[i]];
    if (load > problem.capacity) return false;
    for (int i = 1; i < length; ++i) {
        const int node = route[i];
        load = load - problem.delivery[node] + problem.pickup[node];
        if (load < 0.0 || load > problem.capacity) return false;
    }
    return true;
}

__device__ int first_time_window_violation(
    const int* route,
    int length,
    const DeviceProblem& problem
) {
    double time_value = problem.start_time;
    int previous = route[0];
    for (int i = 1; i < length; ++i) {
        const int node = route[i];
        time_value += problem.travel_time[previous * problem.stride + node];
        if (node != problem.depot && time_value > problem.latest[node]) return node;
        if (time_value < problem.earliest[node]) time_value = problem.earliest[node];
        time_value += problem.service[node];
        previous = node;
    }
    return -1;
}

__device__ double arrival_at_position(
    const int* route,
    int length,
    int position,
    const DeviceProblem& problem
) {
    double time_value = problem.start_time;
    int previous = route[0];
    for (int i = 1; i < length; ++i) {
        const int node = route[i];
        time_value += problem.travel_time[previous * problem.stride + node];
        if (i == position) return time_value;
        if (time_value < problem.earliest[node]) time_value = problem.earliest[node];
        time_value += problem.service[node];
        previous = node;
    }
    return INFINITY;
}

__device__ bool insert_customer_best_position_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    int customer,
    int excluded_route,
    bool minimize_arrival,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    int* candidate_route = scratch.candidate_nodes + solution_id * scratch.route_capacity;
    int best_route = -1;
    int best_position = -1;
    double best_metric = INFINITY;
    const int route_count = solution.route_counts[solution_id];
    const int* nodes = solution_nodes(solution, solution_id);
    const int* offsets = solution_offsets(solution, solution_id);
    const int* lengths = solution_lengths(solution, solution_id);
    for (int route_id = 0; route_id < route_count; ++route_id) {
        if (route_id == excluded_route) continue;
        const int* route = nodes + offsets[route_id];
        const int length = lengths[route_id];
        for (int position = 1; position < length; ++position) {
            for (int i = 0; i < length + 1; ++i) {
                if (i == position) candidate_route[i] = customer;
                else if (i < position) candidate_route[i] = route[i];
                else candidate_route[i] = route[i - 1];
            }
            double route_distance = 0.0;
            if (!evaluate_route(candidate_route, length + 1, problem, &route_distance)) continue;
            const double metric = minimize_arrival
                ? arrival_at_position(candidate_route, length + 1, position, problem)
                : problem.dispatch_cost + route_distance * problem.unit_cost;
            if (best_route == -1 || metric < best_metric) {
                best_route = route_id;
                best_position = position;
                best_metric = metric;
            }
        }
    }
    if (best_route >= 0) {
        int route_length = 0;
        copy_route_to_scratch(solution, solution_id, best_route, candidate_route, &route_length);
        insert_route_node(candidate_route, &route_length, best_position, customer);
        if (!rebuild_solution_with_routes_unchecked(
            temporary, solution_id, solution, best_route, candidate_route, route_length,
            -1, nullptr, 0, problem, false
        )) return false;
        copy_solution(solution, solution_id, temporary, solution_id);
        return true;
    }
    int singleton[3] = {problem.depot, customer, problem.depot};
    return append_route(solution, solution_id, singleton, 3, problem);
}

__device__ bool remove_customer_from_route_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    int route_id,
    int customer,
    bool remove_empty,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    int* route = scratch.candidate_nodes + solution_id * scratch.route_capacity;
    int length = 0;
    copy_route_to_scratch(solution, solution_id, route_id, route, &length);
    int position = -1;
    for (int i = 1; i < length - 1; ++i) {
        if (route[i] == customer) {
            position = i;
            break;
        }
    }
    if (position < 0) return false;
    erase_route_node(route, &length, position);
    if (!rebuild_solution_with_routes_unchecked(
        temporary, solution_id, solution, route_id, route, length,
        -1, nullptr, 0, problem, remove_empty
    )) return false;
    copy_solution(solution, solution_id, temporary, solution_id);
    return true;
}

__device__ bool remove_empty_routes_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    const DeviceProblem& problem
) {
    return rebuild_solution_with_routes_unchecked(
        temporary, solution_id, solution, -1, nullptr, 0, -1, nullptr, 0,
        problem, true
    );
}

__device__ bool repair_solution_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    int* visited = scratch.flags + solution_id * (scratch.customer_capacity + 1);
    for (int i = 0; i <= problem.customer_count; ++i) visited[i] = 0;
    const int* node_data = solution_nodes(solution, solution_id);
    const int node_count = solution.node_counts[solution_id];
    bool already_feasible = true;
    const int route_count = solution.route_counts[solution_id];
    const int* offsets = solution_offsets(solution, solution_id);
    const int* lengths = solution_lengths(solution, solution_id);
    for (int route_id = 0; route_id < route_count; ++route_id) {
        if (!evaluate_route(
            node_data + offsets[route_id], lengths[route_id], problem, nullptr
        )) {
            already_feasible = false;
        }
    }
    for (int i = 0; i < node_count; ++i) {
        const int node = node_data[i];
        if (node > 0 && node <= problem.customer_count) {
            ++visited[node];
            if (visited[node] > 1) already_feasible = false;
        }
    }
    for (int customer = 1; customer <= problem.customer_count; ++customer) {
        if (visited[customer] != 1) already_feasible = false;
    }
    if (already_feasible) return true;
    for (int customer = 1; customer <= problem.customer_count; ++customer) {
        if (!visited[customer] && !insert_customer_best_position_device(
            solution, temporary, solution_id, customer, -1, false, scratch, problem
        )) return false;
    }

    int route_id = 0;
    while (route_id < solution.route_counts[solution_id]) {
        int length = 0;
        int* route = scratch.candidate_nodes + solution_id * scratch.route_capacity;
        copy_route_to_scratch(solution, solution_id, route_id, route, &length);
        if (route_capacity_only(route, length, problem)) {
            ++route_id;
            continue;
        }
        int customer = -1;
        double worst_balance = -1.0;
        for (int i = 1; i < length - 1; ++i) {
            const int node = route[i];
            const double balance = fabs(problem.delivery[node] - problem.pickup[node]);
            if (balance > worst_balance) {
                worst_balance = balance;
                customer = node;
            }
        }
        if (customer < 0 || !remove_customer_from_route_device(
            solution, temporary, solution_id, route_id, customer, false, scratch, problem
        )) return false;
        if (!insert_customer_best_position_device(
            solution, temporary, solution_id, customer, route_id, false, scratch, problem
        )) return false;
        if (!remove_empty_routes_device(solution, temporary, solution_id, problem)) return false;
        copy_solution(solution, solution_id, temporary, solution_id);
    }

    route_id = 0;
    while (route_id < solution.route_counts[solution_id]) {
        int length = 0;
        int* route = scratch.candidate_nodes + solution_id * scratch.route_capacity;
        copy_route_to_scratch(solution, solution_id, route_id, route, &length);
        const int customer = first_time_window_violation(route, length, problem);
        if (customer < 0) {
            ++route_id;
            continue;
        }
        if (!remove_customer_from_route_device(
            solution, temporary, solution_id, route_id, customer, false, scratch, problem
        )) return false;
        if (!insert_customer_best_position_device(
            solution, temporary, solution_id, customer, -1, true, scratch, problem
        )) return false;
        if (!remove_empty_routes_device(solution, temporary, solution_id, problem)) return false;
        copy_solution(solution, solution_id, temporary, solution_id);
    }

    for (route_id = 0; route_id < solution.route_counts[solution_id]; ++route_id) {
        bool improved = true;
        while (improved) {
            improved = false;
            int* original = scratch.candidate_nodes + solution_id * scratch.route_capacity;
            int* candidate = scratch.candidate_nodes_2 + solution_id * scratch.route_capacity;
            int length = 0;
            copy_route_to_scratch(solution, solution_id, route_id, original, &length);
            if (length < 4) break;
            double current_distance = 0.0;
            evaluate_route(original, length, problem, &current_distance);
            double best_distance = current_distance;
            int best_first = -1;
            int best_second = -1;
            for (int first = 1; first < length - 2; ++first) {
                for (int second = first + 1; second < length - 1; ++second) {
                    for (int i = 0; i < length; ++i) candidate[i] = original[i];
                    for (int left = first, right = second; left < right; ++left, --right) {
                        const int temporary_node = candidate[left];
                        candidate[left] = candidate[right];
                        candidate[right] = temporary_node;
                    }
                    double candidate_distance = 0.0;
                    if (evaluate_route(candidate, length, problem, &candidate_distance) &&
                        candidate_distance < best_distance) {
                        best_distance = candidate_distance;
                        best_first = first;
                        best_second = second;
                    }
                }
            }
            if (best_first >= 0) {
                for (int left = best_first, right = best_second; left < right; ++left, --right) {
                    const int temporary_node = original[left];
                    original[left] = original[right];
                    original[right] = temporary_node;
                }
                if (!rebuild_solution_with_routes_unchecked(
                    temporary, solution_id, solution, route_id, original, length,
                    -1, nullptr, 0, problem, false
                )) return false;
                copy_solution(solution, solution_id, temporary, solution_id);
                improved = true;
            }
        }
    }
    return true;
}

__device__ void legacy_shuffle_int(int* values, int count, LegacyMt19937& rng) {
    if (count <= 1) return;
    int current = 1;
    if ((count % 2) == 0) {
        const int position = legacy_randint(0, 1, rng);
        const int temporary = values[current];
        values[current] = values[position];
        values[position] = temporary;
        ++current;
    }
    while (current < count) {
        const unsigned long long swap_range = static_cast<unsigned long long>(current) + 1u;
        const unsigned int combined = legacy_uniform_u32(
            rng, static_cast<unsigned int>(swap_range * (swap_range + 1u))
        );
        const int first_position = static_cast<int>(combined / (swap_range + 1u));
        const int second_position = static_cast<int>(combined % (swap_range + 1u));
        int temporary = values[current];
        values[current] = values[first_position];
        values[first_position] = temporary;
        ++current;
        temporary = values[current];
        values[current] = values[second_position];
        values[second_position] = temporary;
        ++current;
    }
}

__device__ bool objective_better_device(
    DevicePopulation population,
    int candidate,
    DevicePopulation incumbent_population,
    int incumbent
) {
    const int candidate_routes = population.route_counts[candidate];
    const int incumbent_routes = incumbent_population.route_counts[incumbent];
    if (candidate_routes != incumbent_routes) return candidate_routes < incumbent_routes;
    return population.distances[candidate] < incumbent_population.distances[incumbent] - 1e-9;
}

__device__ bool remove_flagged_customers_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    const int* flags,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    clear_solution(temporary, solution_id);
    const int route_count = solution.route_counts[solution_id];
    const int* nodes = solution_nodes(solution, solution_id);
    const int* offsets = solution_offsets(solution, solution_id);
    const int* lengths = solution_lengths(solution, solution_id);
    int* route = scratch.candidate_nodes + solution_id * scratch.route_capacity;
    for (int route_id = 0; route_id < route_count; ++route_id) {
        int length = 0;
        route[length++] = problem.depot;
        const int* source = nodes + offsets[route_id];
        for (int i = 1; i < lengths[route_id] - 1; ++i) {
            if (!flags[source[i]]) route[length++] = source[i];
        }
        route[length++] = problem.depot;
        if (length > 2 && !append_route_unchecked(temporary, solution_id, route, length, problem)) {
            return false;
        }
    }
    copy_solution(solution, solution_id, temporary, solution_id);
    return true;
}

__device__ bool append_customer_min_delta_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    int customer,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    int* candidate = scratch.candidate_nodes + solution_id * scratch.route_capacity;
    int singleton[3] = {problem.depot, customer, problem.depot};
    double singleton_distance = 0.0;
    int best_route = -2;
    int best_position = 1;
    double best_delta = INFINITY;
    if (evaluate_route(singleton, 3, problem, &singleton_distance)) {
        best_route = -1;
        best_delta = problem.dispatch_cost + singleton_distance * problem.unit_cost;
    }
    const int route_count = solution.route_counts[solution_id];
    const int* nodes = solution_nodes(solution, solution_id);
    const int* offsets = solution_offsets(solution, solution_id);
    const int* lengths = solution_lengths(solution, solution_id);
    for (int route_id = 0; route_id < route_count; ++route_id) {
        const int* route = nodes + offsets[route_id];
        const int length = lengths[route_id];
        double old_distance = 0.0;
        for (int i = 1; i < length; ++i) {
            old_distance += problem.distance[route[i - 1] * problem.stride + route[i]];
        }
        for (int position = 1; position < length; ++position) {
            for (int i = 0; i < length + 1; ++i) {
                if (i == position) candidate[i] = customer;
                else if (i < position) candidate[i] = route[i];
                else candidate[i] = route[i - 1];
            }
            double new_distance = 0.0;
            if (!evaluate_route(candidate, length + 1, problem, &new_distance)) continue;
            const double delta = (new_distance - old_distance) * problem.unit_cost;
            if (delta < best_delta - 1e-9) {
                best_delta = delta;
                best_route = route_id;
                best_position = position;
            }
        }
    }
    if (best_route == -2) return false;
    if (best_route == -1) return append_route(solution, solution_id, singleton, 3, problem);
    int length = 0;
    copy_route_to_scratch(solution, solution_id, best_route, candidate, &length);
    insert_route_node(candidate, &length, best_position, customer);
    if (!rebuild_solution_with_routes_unchecked(
        temporary, solution_id, solution, best_route, candidate, length,
        -1, nullptr, 0, problem, false
    )) return false;
    copy_solution(solution, solution_id, temporary, solution_id);
    return true;
}

__device__ bool append_parent_route(
    DevicePopulation child,
    int child_id,
    DevicePopulation parent,
    int parent_id,
    int route_id,
    const DeviceProblem& problem
) {
    const int offset = solution_offsets(parent, parent_id)[route_id];
    const int length = solution_lengths(parent, parent_id)[route_id];
    return append_route_unchecked(
        child, child_id, solution_nodes(parent, parent_id) + offset, length, problem
    );
}

__device__ bool inject_elite_segments_device(
    DevicePopulation child,
    DevicePopulation temporary,
    int solution_id,
    DevicePopulation best,
    LegacyMt19937& rng,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    int* route_indices = scratch.sample_pool + solution_id * scratch.customer_capacity;
    int route_index_count = 0;
    for (int route_id = 0; route_id < best.route_counts[0]; ++route_id) {
        if (solution_lengths(best, 0)[route_id] > 2) route_indices[route_index_count++] = route_id;
    }
    if (route_index_count == 0) return true;
    legacy_shuffle_int(route_indices, route_index_count, rng);
    const int segment_count = min(route_index_count, legacy_randint(1, 3, rng));
    int* segments = scratch.unrouted + solution_id * scratch.customer_capacity;
    int* selected = scratch.flags + solution_id * (scratch.customer_capacity + 1);
    for (int i = 0; i <= problem.customer_count; ++i) selected[i] = 0;
    int segment_offsets[3] = {0, 0, 0};
    int segment_lengths[3] = {0, 0, 0};
    int segment_nodes = 0;
    for (int segment_id = 0; segment_id < segment_count; ++segment_id) {
        const int route_id = route_indices[segment_id];
        const int offset = solution_offsets(best, 0)[route_id];
        const int length = solution_lengths(best, 0)[route_id];
        const int customer_count = length - 2;
        const double l_value = legacy_randdouble(-1.0, 1.0, rng);
        const double spiral_scale = fabs(exp(l_value) * cos(2.0 * 3.14159265358979323846 * l_value));
        const int requested = static_cast<int>(floor(1.0 + spiral_scale + 0.5));
        const int segment_length = max(1, min(customer_count, requested));
        const int start = legacy_randint(0, customer_count - segment_length, rng);
        segment_offsets[segment_id] = segment_nodes;
        const int* route = solution_nodes(best, 0) + offset;
        for (int i = start; i < start + segment_length; ++i) {
            const int node = route[i + 1];
            if (!selected[node]) {
                selected[node] = 1;
                segments[segment_nodes++] = node;
                ++segment_lengths[segment_id];
            }
        }
    }
    if (segment_nodes == 0) return true;
    if (!remove_flagged_customers_device(
        child, temporary, solution_id, selected, scratch, problem
    )) return false;
    int* route = scratch.candidate_nodes + solution_id * scratch.route_capacity;
    for (int segment_id = 0; segment_id < segment_count; ++segment_id) {
        const int length = segment_lengths[segment_id];
        if (length == 0) continue;
        const bool try_route = legacy_randdouble(0.0, 1.0, rng) < 0.70;
        route[0] = problem.depot;
        for (int i = 0; i < length; ++i) {
            route[i + 1] = segments[segment_offsets[segment_id] + i];
        }
        route[length + 1] = problem.depot;
        if (try_route && evaluate_route(route, length + 2, problem, nullptr)) {
            if (!append_route_unchecked(child, solution_id, route, length + 2, problem)) return false;
        } else {
            for (int i = 0; i < length; ++i) {
                if (!append_customer_min_delta_device(
                    child, temporary, solution_id,
                    segments[segment_offsets[segment_id] + i], scratch, problem
                )) return false;
            }
        }
    }
    return true;
}

__device__ bool guided_route_crossover_device(
    DevicePopulation current,
    int current_id,
    int peer_id,
    DevicePopulation best,
    DevicePopulation child,
    DevicePopulation temporary,
    LegacyMt19937& rng,
    double mutation_probability,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    clear_solution(child, current_id);
    int* inserted = scratch.flags + current_id * (scratch.customer_capacity + 1);
    for (int i = 0; i <= problem.customer_count; ++i) inserted[i] = 0;
    int* route_indices = scratch.sample_pool + current_id * scratch.customer_capacity;
    const int best_route_count = best.route_counts[0];
    for (int i = 0; i < best_route_count; ++i) route_indices[i] = i;
    legacy_shuffle_int(route_indices, best_route_count, rng);
    const int take = (best_route_count == 1 || legacy_randdouble(0.0, 1.0, rng) < 0.6) ? 1 : 2;
    for (int i = 0; i < take && i < best_route_count; ++i) {
        const int route_id = route_indices[i];
        const int offset = solution_offsets(best, 0)[route_id];
        const int length = solution_lengths(best, 0)[route_id];
        if (length <= 2) continue;
        if (!append_parent_route(child, current_id, best, 0, route_id, problem)) return false;
        const int* route = solution_nodes(best, 0) + offset;
        for (int j = 1; j < length - 1; ++j) inserted[route[j]] = 1;
    }

    int* remaining = scratch.unrouted + current_id * scratch.customer_capacity;
    int remaining_count = 0;
    DevicePopulation parents[2] = {current, current};
    int parent_ids[2] = {peer_id, current_id};
    for (int parent_index = 0; parent_index < 2; ++parent_index) {
        const int parent_id = parent_ids[parent_index];
        const int node_count = parents[parent_index].node_counts[parent_id];
        const int* nodes = solution_nodes(parents[parent_index], parent_id);
        for (int i = 0; i < node_count; ++i) {
            const int node = nodes[i];
            if (node == problem.depot || inserted[node]) continue;
            inserted[node] = 1;
            remaining[remaining_count++] = node;
        }
    }

    int* route = scratch.route_nodes + current_id * scratch.route_capacity;
    int route_length = 1;
    route[0] = problem.depot;
    for (int i = 0; i < remaining_count; ++i) {
        const int node = remaining[i];
        int singleton[3] = {problem.depot, node, problem.depot};
        if (!route_capacity_only(singleton, 3, problem)) {
            if (!append_customer_min_delta_device(
                child, temporary, current_id, node, scratch, problem
            )) return false;
            continue;
        }
        route[route_length] = node;
        route[route_length + 1] = problem.depot;
        if (route_length > 1 && !route_capacity_only(route, route_length + 2, problem)) {
            route[route_length] = problem.depot;
            if (!append_route_unchecked(child, current_id, route, route_length + 1, problem)) return false;
            route_length = 1;
            route[0] = problem.depot;
        }
        route[route_length++] = node;
    }
    if (route_length > 1) {
        route[route_length] = problem.depot;
        if (!append_route_unchecked(child, current_id, route, route_length + 1, problem)) return false;
    }
    if (legacy_randdouble(0.0, 1.0, rng) < mutation_probability &&
        !inject_elite_segments_device(
            child, temporary, current_id, best, rng, scratch, problem
        )) return false;
    return repair_solution_device(child, temporary, current_id, scratch, problem);
}

__device__ bool woa_intensification_device(
    DevicePopulation current,
    int current_id,
    DevicePopulation best,
    DevicePopulation child,
    DevicePopulation temporary,
    LegacyMt19937& rng,
    double a,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    const double r1 = legacy_randdouble(0.0, 1.0, rng);
    (void)legacy_randdouble(0.0, 1.0, rng);
    const double a_vector = 2.0 * a * r1 - a;
    if (fabs(a_vector) < 1.0) {
        copy_solution(child, current_id, current, current_id);
        const int best_route_count = best.route_counts[0];
        if (best_route_count == 0) return true;
        int count = static_cast<int>(floor(1.0 + fmax(0.0, 2.0 - a) + 0.5));
        count = max(1, min(best_route_count, count));
        int* route_indices = scratch.sample_pool + current_id * scratch.customer_capacity;
        for (int i = 0; i < best_route_count; ++i) route_indices[i] = i;
        legacy_shuffle_int(route_indices, best_route_count, rng);
        int* selected = scratch.flags + current_id * (scratch.customer_capacity + 1);
        for (int i = 0; i <= problem.customer_count; ++i) selected[i] = 0;
        for (int i = 0; i < count; ++i) {
            const int route_id = route_indices[i];
            const int offset = solution_offsets(best, 0)[route_id];
            const int length = solution_lengths(best, 0)[route_id];
            const int* route = solution_nodes(best, 0) + offset;
            for (int j = 1; j < length - 1; ++j) selected[route[j]] = 1;
        }
        if (!remove_flagged_customers_device(
            child, temporary, current_id, selected, scratch, problem
        )) return false;
        for (int i = 0; i < count; ++i) {
            if (!append_parent_route(child, current_id, best, 0, route_indices[i], problem)) return false;
        }
        return repair_solution_device(child, temporary, current_id, scratch, problem);
    }

    int* sequence = scratch.unrouted + current_id * scratch.customer_capacity;
    int sequence_count = 0;
    const int* current_nodes = solution_nodes(current, current_id);
    for (int i = 0; i < current.node_counts[current_id]; ++i) {
        if (current_nodes[i] != problem.depot) sequence[sequence_count++] = current_nodes[i];
    }
    if (sequence_count >= 2) {
        const int move_type = legacy_randint(0, 2, rng);
        if (move_type == 0) {
            const int first = legacy_randint(0, sequence_count - 1, rng);
            const int second = legacy_randint(0, sequence_count - 1, rng);
            const int temporary_node = sequence[first];
            sequence[first] = sequence[second];
            sequence[second] = temporary_node;
        } else if (move_type == 1) {
            const int first = legacy_randint(0, sequence_count - 1, rng);
            const int node = sequence[first];
            for (int i = first; i + 1 < sequence_count; ++i) sequence[i] = sequence[i + 1];
            --sequence_count;
            const int second = legacy_randint(0, sequence_count, rng);
            for (int i = sequence_count; i > second; --i) sequence[i] = sequence[i - 1];
            sequence[second] = node;
            ++sequence_count;
        } else {
            int first = legacy_randint(0, sequence_count - 1, rng);
            int second = legacy_randint(0, sequence_count - 1, rng);
            if (first > second) {
                const int temporary_index = first;
                first = second;
                second = temporary_index;
            }
            while (first < second) {
                const int temporary_node = sequence[first];
                sequence[first++] = sequence[second];
                sequence[second--] = temporary_node;
            }
        }
    }

    clear_solution(child, current_id);
    int* route = scratch.route_nodes + current_id * scratch.route_capacity;
    int route_length = 1;
    route[0] = problem.depot;
    for (int i = 0; i < sequence_count; ++i) {
        const int node = sequence[i];
        route[route_length] = node;
        route[route_length + 1] = problem.depot;
        if (evaluate_route(route, route_length + 2, problem, nullptr)) {
            ++route_length;
            continue;
        }
        if (route_length > 1) {
            route[route_length] = problem.depot;
            if (!append_route_unchecked(child, current_id, route, route_length + 1, problem)) return false;
        }
        route_length = 1;
        route[0] = problem.depot;
        int singleton[3] = {problem.depot, node, problem.depot};
        if (evaluate_route(singleton, 3, problem, nullptr)) {
            route[route_length++] = node;
        } else if (!append_route_unchecked(child, current_id, singleton, 3, problem)) {
            return false;
        }
    }
    if (route_length > 1) {
        route[route_length] = problem.depot;
        if (!append_route_unchecked(child, current_id, route, route_length + 1, problem)) return false;
    }
    return repair_solution_device(child, temporary, current_id, scratch, problem);
}

__global__ void initialize_global_best_kernel(
    DevicePopulation population,
    DevicePopulation best
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int best_id = 0;
    for (int i = 1; i < population.solution_count; ++i) {
        if (objective_better_device(population, i, population, best_id)) best_id = i;
    }
    copy_solution(best, 0, population, best_id);
}

__global__ void initialize_island_bests_kernel(
    DevicePopulation population,
    DevicePopulation island_bests,
    int island_size
) {
    const int island = blockIdx.x * blockDim.x + threadIdx.x;
    if (island >= island_bests.solution_count) return;
    const int begin = island * island_size;
    int best_id = begin;
    for (int i = 1; i < island_size; ++i) {
        const int candidate = begin + i;
        if (objective_better_device(population, candidate, population, best_id)) best_id = candidate;
    }
    copy_solution(island_bests, island, population, best_id);
}

__global__ void update_global_from_islands_kernel(
    DevicePopulation island_bests,
    DevicePopulation global_best
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int best_island = 0;
    for (int i = 1; i < island_bests.solution_count; ++i) {
        if (objective_better_device(island_bests, i, island_bests, best_island)) best_island = i;
    }
    if (objective_better_device(island_bests, best_island, global_best, 0)) {
        copy_solution(global_best, 0, island_bests, best_island);
    }
}

__global__ void prepare_generation_kernel(
    DevicePopulation population,
    LegacyMt19937* global_rng_state,
    int* peer_workspace,
    int* peer_indices,
    int* update_seeds
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    LegacyMt19937 rng = global_rng_state[0];
    for (int current = 0; current < population.solution_count; ++current) {
        int count = 0;
        for (int i = 0; i < population.solution_count; ++i) {
            if (i != current) peer_workspace[count++] = i;
        }
        if (count == 0) {
            peer_indices[current] = current;
        } else {
            legacy_shuffle_int(peer_workspace, count, rng);
            int best_peer = peer_workspace[0];
            for (int i = 1; i < 3 && i < count; ++i) {
                if (objective_better_device(population, peer_workspace[i], population, best_peer)) {
                    best_peer = peer_workspace[i];
                }
            }
            peer_indices[current] = best_peer;
        }
        update_seeds[current] = static_cast<int>(rng());
    }
    global_rng_state[0] = rng;
}

__global__ void update_population_kernel(
    DevicePopulation current,
    DevicePopulation next,
    DevicePopulation temporary,
    DevicePopulation best,
    DeviceConstructionScratch scratch,
    DeviceProblem problem,
    const int* peer_indices,
    const int* update_seeds,
    double p_mode,
    double a,
    int iteration,
    int max_iterations,
    double mutation_probability,
    int* accepted_count,
    int* status
) {
    const int solution_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (solution_id >= current.solution_count) return;
    LegacyMt19937 rng(static_cast<unsigned int>(update_seeds[solution_id]));
    bool ok = false;
    if (legacy_randdouble(0.0, 1.0, rng) < p_mode) {
        ok = guided_route_crossover_device(
            current, solution_id, peer_indices[solution_id], best,
            next, temporary, rng, mutation_probability, scratch, problem
        );
    } else {
        ok = woa_intensification_device(
            current, solution_id, best, next, temporary, rng, a, scratch, problem
        );
    }
    if (!ok) {
        copy_solution(next, solution_id, current, solution_id);
        status[solution_id] = 0;
        return;
    }
    bool accepted = false;
    const int new_routes = next.route_counts[solution_id];
    const int old_routes = current.route_counts[solution_id];
    if (new_routes < old_routes) {
        accepted = true;
    } else if (new_routes == old_routes) {
        const double delta = next.distances[solution_id] - current.distances[solution_id];
        if (delta <= 0.001) {
            accepted = true;
        } else {
            const double temperature = max_iterations > 0
                ? 1.0 - static_cast<double>(iteration) / max_iterations
                : 0.0;
            const double probability = exp(
                -delta / (1e-6 + temperature * fabs(current.distances[solution_id]))
            );
            accepted = legacy_randdouble(0.0, 1.0, rng) < probability;
        }
    }
    if (!accepted) {
        copy_solution(next, solution_id, current, solution_id);
    } else {
        atomicAdd(accepted_count, 1);
    }
    status[solution_id] = 0;
}

__global__ void update_global_best_kernel(
    DevicePopulation population,
    DevicePopulation best,
    int generation,
    int* last_improvement_generation
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int generation_best = 0;
    for (int i = 1; i < population.solution_count; ++i) {
        if (objective_better_device(population, i, population, generation_best)) generation_best = i;
    }
    if (objective_better_device(population, generation_best, best, 0)) {
        copy_solution(best, 0, population, generation_best);
        last_improvement_generation[0] = generation;
    }
}

struct DeviceLocalMove {
    int op;
    int route_1;
    int route_2;
    int first_1;
    int last_1;
    int first_2;
    int last_2;
    int position;
    double delta;
};

__device__ double route_cost_device(
    const int* route,
    int length,
    const DeviceProblem& problem
) {
    if (length <= 2) return 0.0;
    double distance = 0.0;
    for (int i = 1; i < length; ++i) {
        distance += problem.distance[route[i - 1] * problem.stride + route[i]];
    }
    return problem.dispatch_cost + distance * problem.unit_cost;
}

__device__ void append_range(
    int* destination,
    int* destination_length,
    const int* source,
    int first,
    int last
) {
    if (first <= last) {
        for (int i = first; i <= last; ++i) destination[(*destination_length)++] = source[i];
    } else {
        for (int i = first; i >= last; --i) destination[(*destination_length)++] = source[i];
    }
}

__device__ bool materialize_local_move(
    DevicePopulation solution,
    int solution_id,
    const DeviceLocalMove& move,
    int* first,
    int* first_length,
    int* second,
    int* second_length
) {
    const int* nodes = solution_nodes(solution, solution_id);
    const int* offsets = solution_offsets(solution, solution_id);
    const int* lengths = solution_lengths(solution, solution_id);
    const int* route_1 = nodes + offsets[move.route_1];
    const int length_1 = lengths[move.route_1];
    *first_length = 0;
    *second_length = 0;
    if (move.op == 0) {
        for (int i = 0; i < length_1; ++i) first[i] = route_1[i];
        *first_length = length_1;
        const int temporary = first[move.first_1];
        first[move.first_1] = first[move.first_1 + 1];
        first[move.first_1 + 1] = temporary;
        return true;
    }
    if (move.op == 2 || move.op == 4) {
        if (move.op == 4) {
            append_range(first, first_length, route_1, 0, move.first_1 - 1);
            append_range(first, first_length, route_1, move.last_1 + 1, length_1 - 1);
            second[(*second_length)++] = route_1[0];
            append_range(second, second_length, route_1, move.first_1, move.last_1);
            second[(*second_length)++] = route_1[length_1 - 1];
        } else if (move.position < move.first_1) {
            append_range(first, first_length, route_1, 0, move.position - 1);
            append_range(first, first_length, route_1, move.first_1, move.last_1);
            append_range(first, first_length, route_1, move.position, move.first_1 - 1);
            append_range(first, first_length, route_1, move.last_1 + 1, length_1 - 1);
        } else {
            append_range(first, first_length, route_1, 0, move.first_1 - 1);
            append_range(first, first_length, route_1, move.last_1 + 1, move.position - 1);
            append_range(first, first_length, route_1, move.first_1, move.last_1);
            append_range(first, first_length, route_1, move.position, length_1 - 1);
        }
        return true;
    }
    const int* route_2 = nodes + offsets[move.route_2];
    const int length_2 = lengths[move.route_2];
    if (move.op == 1) {
        append_range(first, first_length, route_1, 0, move.first_1 - 1);
        append_range(first, first_length, route_2, move.first_2, length_2 - 1);
        append_range(second, second_length, route_2, 0, move.first_2 - 1);
        append_range(second, second_length, route_1, move.first_1, length_1 - 1);
    } else if (move.op == 3) {
        append_range(first, first_length, route_1, 0, move.first_1 - 1);
        append_range(first, first_length, route_2, move.first_2, move.last_2);
        append_range(first, first_length, route_1, move.last_1 + 1, length_1 - 1);
        append_range(second, second_length, route_2, 0, move.first_2 - 1);
        append_range(second, second_length, route_1, move.first_1, move.last_1);
        append_range(second, second_length, route_2, move.last_2 + 1, length_2 - 1);
    } else {
        return false;
    }
    return true;
}

__device__ bool evaluate_local_move_device(
    DevicePopulation solution,
    int solution_id,
    DeviceLocalMove* move,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    int* first = scratch.candidate_nodes + solution_id * scratch.route_capacity;
    int* second = scratch.candidate_nodes_2 + solution_id * scratch.route_capacity;
    int first_length = 0;
    int second_length = 0;
    if (!materialize_local_move(
        solution, solution_id, *move, first, &first_length, second, &second_length
    )) return false;
    if (!evaluate_route(first, first_length, problem, nullptr)) return false;
    if (move->route_2 != -2 && !evaluate_route(second, second_length, problem, nullptr)) return false;
    const int* nodes = solution_nodes(solution, solution_id);
    const int* offsets = solution_offsets(solution, solution_id);
    const int* lengths = solution_lengths(solution, solution_id);
    double old_cost = route_cost_device(
        nodes + offsets[move->route_1], lengths[move->route_1], problem
    );
    if (move->route_2 >= 0) {
        old_cost += route_cost_device(
            nodes + offsets[move->route_2], lengths[move->route_2], problem
        );
    }
    double new_cost = route_cost_device(first, first_length, problem);
    if (move->route_2 != -2) new_cost += route_cost_device(second, second_length, problem);
    move->delta = new_cost - old_cost;
    return true;
}

__device__ void consider_local_move(
    DevicePopulation solution,
    int solution_id,
    DeviceLocalMove candidate,
    DeviceLocalMove* best,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    if (evaluate_local_move_device(solution, solution_id, &candidate, scratch, problem) &&
        candidate.delta < best->delta - 1e-9) {
        *best = candidate;
    }
}

__device__ bool apply_local_move_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    const DeviceLocalMove& move,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    int* first = scratch.candidate_nodes + solution_id * scratch.route_capacity;
    int* second = scratch.candidate_nodes_2 + solution_id * scratch.route_capacity;
    int first_length = 0;
    int second_length = 0;
    if (!materialize_local_move(
        solution, solution_id, move, first, &first_length, second, &second_length
    )) return false;
    clear_solution(temporary, solution_id);
    const int route_count = solution.route_counts[solution_id];
    const int* nodes = solution_nodes(solution, solution_id);
    const int* offsets = solution_offsets(solution, solution_id);
    const int* lengths = solution_lengths(solution, solution_id);
    for (int route_id = 0; route_id < route_count; ++route_id) {
        const int* route = nodes + offsets[route_id];
        int length = lengths[route_id];
        if (route_id == move.route_1) {
            route = first;
            length = first_length;
        } else if (route_id == move.route_2) {
            route = second;
            length = second_length;
        }
        if (length <= 2) continue;
        if (!append_route(temporary, solution_id, route, length, problem)) return false;
    }
    if (move.route_2 == -1 && second_length > 2 &&
        !append_route(temporary, solution_id, second, second_length, problem)) return false;
    copy_solution(solution, solution_id, temporary, solution_id);
    return true;
}

__device__ bool find_local_optimum_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    int or_opt_length,
    int exchange_length,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    while (true) {
        DeviceLocalMove operator_best[4];
        for (int i = 0; i < 4; ++i) operator_best[i].delta = INFINITY;
        const int route_count = solution.route_counts[solution_id];
        const int* lengths = solution_lengths(solution, solution_id);
        for (int route_id = 0; route_id < route_count; ++route_id) {
            const int length = lengths[route_id];
            for (int start = 1; start < length - 2; ++start) {
                DeviceLocalMove move{0, route_id, -2, start, start + 1, 0, 0, 0, INFINITY};
                consider_local_move(solution, solution_id, move, &operator_best[0], scratch, problem);
            }
        }
        for (int first_route = 0; first_route < route_count; ++first_route) {
            const int first_length = lengths[first_route];
            for (int second_route = first_route + 1; second_route < route_count; ++second_route) {
                const int second_length = lengths[second_route];
                for (int first = 1; first < first_length; ++first) {
                    for (int second = 1; second < second_length; ++second) {
                        if ((first == 1 && second == 1) ||
                            (first == first_length - 1 && second == second_length - 1)) continue;
                        DeviceLocalMove move{1, first_route, second_route, first, 0, second, 0, 0, INFINITY};
                        consider_local_move(solution, solution_id, move, &operator_best[1], scratch, problem);
                    }
                }
            }
        }
        for (int route_id = 0; route_id < route_count; ++route_id) {
            const int length = lengths[route_id];
            for (int start = 1; start < length - 1; ++start) {
                for (int sequence_length = 1; sequence_length <= or_opt_length; ++sequence_length) {
                    const int end = start + sequence_length - 1;
                    if (end >= length - 1) continue;
                    for (int position = 1; position < start; ++position) {
                        DeviceLocalMove move{2, route_id, -2, start, end, 0, 0, position, INFINITY};
                        consider_local_move(solution, solution_id, move, &operator_best[2], scratch, problem);
                    }
                    for (int position = end + 2; position < length; ++position) {
                        DeviceLocalMove move{2, route_id, -2, start, end, 0, 0, position, INFINITY};
                        consider_local_move(solution, solution_id, move, &operator_best[2], scratch, problem);
                    }
                    DeviceLocalMove move{4, route_id, -1, start, end, 0, 0, 0, INFINITY};
                    consider_local_move(solution, solution_id, move, &operator_best[2], scratch, problem);
                }
            }
        }
        for (int first_route = 0; first_route < route_count; ++first_route) {
            const int first_length = lengths[first_route];
            for (int second_route = first_route + 1; second_route < route_count; ++second_route) {
                const int second_length = lengths[second_route];
                for (int first = 1; first < first_length - 1; ++first) {
                    for (int first_size = 1; first_size <= exchange_length; ++first_size) {
                        const int first_end = first + first_size - 1;
                        if (first_end >= first_length - 1) continue;
                        for (int second = 1; second < second_length - 1; ++second) {
                            for (int second_size = 1; second_size <= exchange_length; ++second_size) {
                                const int second_end = second + second_size - 1;
                                if (second_end >= second_length - 1) continue;
                                DeviceLocalMove move{3, first_route, second_route,
                                    first, first_end, second, second_end, 0, INFINITY};
                                consider_local_move(solution, solution_id, move, &operator_best[3], scratch, problem);
                            }
                        }
                    }
                }
            }
        }
        int best_operator = -1;
        double best_delta = INFINITY;
        for (int i = 0; i < 4; ++i) {
            if (operator_best[i].delta - best_delta < -0.001) {
                best_operator = i;
                best_delta = operator_best[i].delta;
            }
        }
        if (best_operator < 0 || best_delta >= -0.001) return true;
        if (!apply_local_move_device(
            solution, temporary, solution_id, operator_best[best_operator], scratch, problem
        )) return false;
    }
}

__device__ bool regret_insertion_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
);

__device__ bool related_removal_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    LegacyMt19937& rng,
    double removal_lower,
    double removal_upper,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
);

__global__ void deep_local_search_kernel(
    DevicePopulation best,
    DevicePopulation perturbation,
    DevicePopulation temporary,
    DeviceConstructionScratch scratch,
    DeviceProblem problem,
    int or_opt_length,
    int exchange_length,
    int elo,
    unsigned int seed,
    double removal_lower,
    double removal_upper,
    int* status
) {
    const int island = blockIdx.x;
    if (island >= best.solution_count) return;
    if (threadIdx.x != 0) return;
    DeviceConstructionScratch island_scratch = slice_scratch(scratch, island * best.max_routes);
    if (!find_local_optimum_device(
        best, temporary, island, or_opt_length, exchange_length, island_scratch, problem
    )) {
        status[island] = 4;
        return;
    }
    if (elo <= 0) return;
    LegacyMt19937 rng(seed + island * 1000);
    int no_improve = 0;
    while (no_improve < elo) {
        copy_solution(perturbation, island, best, island);
        if (!related_removal_device(
            perturbation, temporary, island, rng, removal_lower, removal_upper, island_scratch, problem
        ) || !regret_insertion_device(
            perturbation, temporary, island, island_scratch, problem
        ) || !find_local_optimum_device(
            perturbation, temporary, island, or_opt_length, exchange_length, island_scratch, problem
        )) {
            status[island] = 4;
            return;
        }
        if (perturbation.costs[island] - best.costs[island] < -0.001) {
            copy_solution(best, island, perturbation, island);
            no_improve = 0;
        } else {
            ++no_improve;
        }
    }
}

__global__ void prepare_local_search_kernel(
    DevicePopulation best,
    DevicePopulation candidate
) {
    const int island = blockIdx.x * blockDim.x + threadIdx.x;
    if (island < best.solution_count) {
        copy_solution(candidate, island, best, island);
    }
}

__global__ void commit_local_search_and_inject_kernel(
    DevicePopulation population,
    DevicePopulation best,
    DevicePopulation candidate,
    int generation,
    int* last_improvement_generation,
    int* status
) {
    const int island = blockIdx.x;
    if (island >= best.solution_count || threadIdx.x != 0) return;
    if (status[island] != 0) {
        status[island] = 0;
        return;
    }
    if (objective_better_device(candidate, island, best, island)) {
        copy_solution(best, island, candidate, island);
        last_improvement_generation[island] = generation;
    }
    int worst = island * (population.solution_count / best.solution_count);
    const int end_idx = worst + (population.solution_count / best.solution_count);
    for (int i = worst + 1; i < end_idx; ++i) {
        if (objective_better_device(population, worst, population, i)) worst = i;
    }
    copy_solution(population, worst, best, island);
}

struct DeviceInsertionOption {
    int route;
    int position;
    double delta;
};

__device__ bool insertion_option_better_device(
    const DeviceInsertionOption& candidate,
    const DeviceInsertionOption& incumbent
) {
    if (fabs(candidate.delta - incumbent.delta) > 1e-9) return candidate.delta < incumbent.delta;
    if (candidate.route != incumbent.route) return candidate.route < incumbent.route;
    return candidate.position < incumbent.position;
}

__device__ bool best_two_insertion_options_device(
    DevicePopulation solution,
    int solution_id,
    int customer,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem,
    DeviceInsertionOption* first,
    DeviceInsertionOption* second
) {
    first->delta = INFINITY;
    second->delta = INFINITY;
    int singleton[3] = {problem.depot, customer, problem.depot};
    double singleton_distance = 0.0;
    if (evaluate_route(singleton, 3, problem, &singleton_distance)) {
        DeviceInsertionOption option{-1, 1,
            problem.dispatch_cost + singleton_distance * problem.unit_cost};
        *first = option;
    }
    int* candidate_route = scratch.candidate_nodes + solution_id * scratch.route_capacity;
    const int route_count = solution.route_counts[solution_id];
    const int* nodes = solution_nodes(solution, solution_id);
    const int* offsets = solution_offsets(solution, solution_id);
    const int* lengths = solution_lengths(solution, solution_id);
    for (int route_id = 0; route_id < route_count; ++route_id) {
        const int* route = nodes + offsets[route_id];
        const int length = lengths[route_id];
        const double old_cost = route_cost_device(route, length, problem);
        for (int position = 1; position < length; ++position) {
            for (int i = 0; i < length + 1; ++i) {
                if (i == position) candidate_route[i] = customer;
                else if (i < position) candidate_route[i] = route[i];
                else candidate_route[i] = route[i - 1];
            }
            double distance = 0.0;
            if (!evaluate_route(candidate_route, length + 1, problem, &distance)) continue;
            DeviceInsertionOption option{
                route_id, position,
                problem.dispatch_cost + distance * problem.unit_cost - old_cost
            };
            if (!isfinite(first->delta) || insertion_option_better_device(option, *first)) {
                *second = *first;
                *first = option;
            } else if (!isfinite(second->delta) || insertion_option_better_device(option, *second)) {
                *second = option;
            }
        }
    }
    if (!isfinite(first->delta)) return false;
    if (!isfinite(second->delta)) *second = *first;
    return true;
}

__device__ bool apply_insertion_option_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    int customer,
    const DeviceInsertionOption& option,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    if (option.route < 0) {
        int singleton[3] = {problem.depot, customer, problem.depot};
        return append_route(solution, solution_id, singleton, 3, problem);
    }
    int* route = scratch.candidate_nodes + solution_id * scratch.route_capacity;
    int length = 0;
    copy_route_to_scratch(solution, solution_id, option.route, route, &length);
    insert_route_node(route, &length, option.position, customer);
    if (!rebuild_solution_with_routes_unchecked(
        temporary, solution_id, solution, option.route, route, length,
        -1, nullptr, 0, problem, false
    )) return false;
    copy_solution(solution, solution_id, temporary, solution_id);
    return true;
}

__device__ bool regret_insertion_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    int* visited = scratch.flags + solution_id * (scratch.customer_capacity + 1);
    int* unrouted = scratch.unrouted + solution_id * scratch.customer_capacity;
    while (true) {
        for (int i = 0; i <= problem.customer_count; ++i) visited[i] = 0;
        const int* nodes = solution_nodes(solution, solution_id);
        for (int i = 0; i < solution.node_counts[solution_id]; ++i) {
            if (nodes[i] > 0 && nodes[i] <= problem.customer_count) visited[nodes[i]] = 1;
        }
        int unrouted_count = 0;
        for (int customer = 1; customer <= problem.customer_count; ++customer) {
            if (!visited[customer]) unrouted[unrouted_count++] = customer;
        }
        if (unrouted_count == 0) return true;
        int selected = -1;
        DeviceInsertionOption selected_option{};
        double best_regret = -INFINITY;
        for (int i = 0; i < unrouted_count; ++i) {
            DeviceInsertionOption first{};
            DeviceInsertionOption second{};
            if (!best_two_insertion_options_device(
                solution, solution_id, unrouted[i], scratch, problem, &first, &second
            )) continue;
            const double regret = second.delta - first.delta;
            if (regret > best_regret + 1e-9 ||
                (fabs(regret - best_regret) <= 1e-9 &&
                 (selected < 0 || first.delta < selected_option.delta))) {
                selected = i;
                selected_option = first;
                best_regret = regret;
            }
        }
        if (selected < 0 || !apply_insertion_option_device(
            solution, temporary, solution_id, unrouted[selected], selected_option, scratch, problem
        )) return false;
    }
}

__device__ bool related_removal_device(
    DevicePopulation solution,
    DevicePopulation temporary,
    int solution_id,
    LegacyMt19937& rng,
    double removal_lower,
    double removal_upper,
    DeviceConstructionScratch scratch,
    const DeviceProblem& problem
) {
    int* present = scratch.sample_pool + solution_id * scratch.customer_capacity;
    int* selected = scratch.unrouted + solution_id * scratch.customer_capacity;
    int* state = scratch.flags + solution_id * (scratch.customer_capacity + 1);
    for (int i = 0; i <= problem.customer_count; ++i) state[i] = 0;
    int present_count = 0;
    const int* nodes = solution_nodes(solution, solution_id);
    for (int i = 0; i < solution.node_counts[solution_id]; ++i) {
        const int node = nodes[i];
        if (node > 0 && node <= problem.customer_count && state[node] == 0) {
            state[node] = 1;
            present[present_count++] = node;
        }
    }
    if (present_count == 0) return true;
    int remove_count = static_cast<int>(floor(
        problem.customer_count * legacy_randdouble(removal_lower, removal_upper, rng) + 0.5
    ));
    remove_count = max(1, min(remove_count, present_count));
    int selected_count = 0;
    const int seed_node = present[legacy_randint(0, present_count - 1, rng)];
    state[seed_node] = 2;
    selected[selected_count++] = seed_node;
    while (selected_count < remove_count) {
        const int reference = selected[legacy_randint(0, selected_count - 1, rng)];
        int first = -1;
        int second = -1;
        const int* rank = problem.related_rank + reference * problem.stride;
        for (int i = 0; i < problem.stride; ++i) {
            const int candidate = rank[i];
            if (candidate <= 0 || candidate > problem.customer_count || state[candidate] != 1) continue;
            if (first < 0) first = candidate;
            else {
                second = candidate;
                break;
            }
        }
        int chosen = first;
        if (first < 0) {
            for (int i = 0; i < present_count; ++i) {
                if (state[present[i]] == 1) {
                    chosen = present[i];
                    break;
                }
            }
        } else if (second >= 0) {
            const double first_value = problem.related[reference * problem.stride + first];
            const double second_value = problem.related[reference * problem.stride + second];
            const double denominator = first_value + second_value;
            const double first_weight = denominator > 0.0 && isfinite(denominator)
                ? second_value / denominator
                : 1.0;
            chosen = legacy_randdouble(0.0, 1.0, rng) < first_weight ? first : second;
        }
        if (chosen < 0) break;
        state[chosen] = 2;
        selected[selected_count++] = chosen;
    }
    for (int i = 0; i <= problem.customer_count; ++i) state[i] = state[i] == 2 ? 1 : 0;
    return remove_flagged_customers_device(
        solution, temporary, solution_id, state, scratch, problem
    );
}

__device__ void stable_rank_population_device(DevicePopulation population, int* rank) {
    for (int i = 0; i < population.solution_count; ++i) rank[i] = i;
    for (int i = 1; i < population.solution_count; ++i) {
        const int value = rank[i];
        int position = i;
        while (position > 0 && objective_better_device(population, value, population, rank[position - 1])) {
            rank[position] = rank[position - 1];
            --position;
        }
        rank[position] = value;
    }
}

__device__ void stable_rank_island_device(
    DevicePopulation population,
    int island,
    int island_size,
    int* rank
) {
    const int begin = island * island_size;
    for (int i = 0; i < island_size; ++i) rank[i] = i;
    for (int i = 1; i < island_size; ++i) {
        const int value = rank[i];
        int position = i;
        while (position > 0 && objective_better_device(
            population, begin + value, population, begin + rank[position - 1]
        )) {
            rank[position] = rank[position - 1];
            --position;
        }
        rank[position] = value;
    }
}

__device__ void replace_island_worst_device(
    DevicePopulation population,
    DevicePopulation migrants,
    int destination,
    int source,
    int island_size,
    int migrant_count,
    int* all_ranks
) {
    int* destination_rank = all_ranks + destination * island_size;
    for (int migrant = 0; migrant < migrant_count; ++migrant) {
        const int destination_id = destination * island_size +
            destination_rank[island_size - 1 - migrant];
        const int migrant_id = source * migrant_count + migrant;
        copy_solution(population, destination_id, migrants, migrant_id);
    }
    stable_rank_island_device(
        population, destination, island_size, destination_rank
    );
}

__global__ void migrate_islands_kernel(
    DevicePopulation population,
    DevicePopulation island_bests,
    DevicePopulation migrants,
    int island_size,
    int migrant_count,
    int migration_mode,
    int* ranks,
    int* island_indices,
    LegacyMt19937* global_rng_state
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int island_count = island_bests.solution_count;
    for (int island = 0; island < island_count; ++island) {
        int* rank = ranks + island * island_size;
        stable_rank_island_device(population, island, island_size, rank);
        for (int migrant = 0; migrant < migrant_count; ++migrant) {
            copy_solution(
                migrants, island * migrant_count + migrant,
                population, island * island_size + rank[migrant]
            );
        }
    }
    if (migration_mode == 0) {
        for (int source = 0; source < island_count; ++source) {
            replace_island_worst_device(
                population, migrants, (source + 1) % island_count, source,
                island_size, migrant_count, ranks
            );
        }
    } else if (migration_mode == 1) {
        int best_island = 0;
        for (int island = 1; island < island_count; ++island) {
            if (objective_better_device(island_bests, island, island_bests, best_island)) {
                best_island = island;
            }
        }
        for (int destination = 0; destination < island_count; ++destination) {
            if (destination != best_island) {
                replace_island_worst_device(
                    population, migrants, destination, best_island,
                    island_size, migrant_count, ranks
                );
            }
        }
    } else {
        LegacyMt19937 rng = global_rng_state[0];
        for (int island = 0; island < island_count; ++island) island_indices[island] = island;
        legacy_shuffle_int(island_indices, island_count, rng);
        for (int i = 0; i < island_count - 1; i += 2) {
            const int first = island_indices[i];
            const int second = island_indices[i + 1];
            int winner = first;
            int loser = second;
            if (objective_better_device(island_bests, second, island_bests, first)) {
                winner = second;
                loser = first;
            }
            replace_island_worst_device(
                population, migrants, loser, winner, island_size, migrant_count, ranks
            );
        }
        global_rng_state[0] = rng;
    }
}

__global__ void diversify_population_kernel(
    DevicePopulation population,
    DevicePopulation temporary,
    DevicePopulation backup,
    DevicePopulation best,
    DeviceConstructionScratch scratch,
    DeviceProblem problem,
    LegacyMt19937* global_rng_state,
    int* rank,
    int generation,
    int* last_improvement_generation,
    double diversify_ratio,
    double removal_lower,
    double removal_upper,
    int stagnation_interval,
    int* status
) {
    if (threadIdx.x != 0) return;
    if (generation - last_improvement_generation[0] < stagnation_interval) return;
    LegacyMt19937 rng = global_rng_state[0];
    stable_rank_population_device(population, rank);
    copy_solution(population, rank[0], best, 0);
    const int diversify_count = static_cast<int>(floor(
        population.solution_count * diversify_ratio + 0.5
    ));
    for (int rank_position = 1;
         rank_position <= diversify_count && rank_position < population.solution_count;
         ++rank_position) {
        const int solution_id = rank[rank_position];
        copy_solution(backup, solution_id, population, solution_id);
        int* customers = scratch.sample_pool + solution_id * scratch.customer_capacity;
        int* flags = scratch.flags + solution_id * (scratch.customer_capacity + 1);
        for (int i = 0; i < problem.customer_count; ++i) customers[i] = i + 1;
        const int remove_count = static_cast<int>(floor(
            problem.customer_count * legacy_randdouble(removal_lower, removal_upper, rng) + 0.5
        ));
        legacy_shuffle_int(customers, problem.customer_count, rng);
        for (int i = 0; i <= problem.customer_count; ++i) flags[i] = 0;
        for (int i = 0; i < remove_count && i < problem.customer_count; ++i) flags[customers[i]] = 1;
        if (!remove_flagged_customers_device(
            population, temporary, solution_id, flags, scratch, problem
        ) || !regret_insertion_device(
            population, temporary, solution_id, scratch, problem
        )) {
            copy_solution(population, solution_id, backup, solution_id);
            status[solution_id] = 0;
        }
    }
    stable_rank_population_device(population, rank);
    global_rng_state[0] = rng;
    last_improvement_generation[0] = generation;
}

struct DevicePopulationOwner {
    DevicePopulation view{};
    DeviceBuffer<int> nodes;
    DeviceBuffer<int> offsets;
    DeviceBuffer<int> lengths;
    DeviceBuffer<int> route_counts;
    DeviceBuffer<int> node_counts;
    DeviceBuffer<double> costs;
    DeviceBuffer<double> distances;

    DevicePopulationOwner(int solution_count, int customer_count) :
        nodes(static_cast<std::size_t>(solution_count) * 3u * customer_count),
        offsets(static_cast<std::size_t>(solution_count) * customer_count),
        lengths(static_cast<std::size_t>(solution_count) * customer_count),
        route_counts(solution_count),
        node_counts(solution_count),
        costs(solution_count),
        distances(solution_count) {
        view.solution_count = solution_count;
        view.max_routes = customer_count;
        view.node_capacity = 3 * customer_count;
        view.nodes = nodes.get();
        view.route_offsets = offsets.get();
        view.route_lengths = lengths.get();
        view.route_counts = route_counts.get();
        view.node_counts = node_counts.get();
        view.costs = costs.get();
        view.distances = distances.get();
    }
};

Solution decode_solution(
    const DevicePopulationOwner& population,
    int solution_id,
    const Data& data
) {
    int route_count = 0;
    int node_count = 0;
    cuda_check(cudaMemcpy(
        &route_count,
        population.route_counts.get() + solution_id,
        sizeof(int), cudaMemcpyDeviceToHost
    ), "cudaMemcpy(full_gpu route_count)");
    cuda_check(cudaMemcpy(
        &node_count,
        population.node_counts.get() + solution_id,
        sizeof(int), cudaMemcpyDeviceToHost
    ), "cudaMemcpy(full_gpu node_count)");
    std::vector<int> nodes(node_count);
    std::vector<int> offsets(route_count);
    std::vector<int> lengths(route_count);
    cuda_check(cudaMemcpy(
        nodes.data(), population.nodes.get() + solution_id * population.view.node_capacity,
        node_count * sizeof(int), cudaMemcpyDeviceToHost
    ), "cudaMemcpy(full_gpu nodes)");
    cuda_check(cudaMemcpy(
        offsets.data(), population.offsets.get() + solution_id * population.view.max_routes,
        route_count * sizeof(int), cudaMemcpyDeviceToHost
    ), "cudaMemcpy(full_gpu offsets)");
    cuda_check(cudaMemcpy(
        lengths.data(), population.lengths.get() + solution_id * population.view.max_routes,
        route_count * sizeof(int), cudaMemcpyDeviceToHost
    ), "cudaMemcpy(full_gpu lengths)");

    Solution result;
    for (int route_id = 0; route_id < route_count; ++route_id) {
        Route route(data);
        route.node_list.assign(
            nodes.begin() + offsets[route_id],
            nodes.begin() + offsets[route_id] + lengths[route_id]
        );
        route.update(data);
        route.cal_cost(data);
        result.append(route);
    }
    result.cal_cost(data);
    return result;
}

} // namespace

bool run_full_gpu_solver(Data& data, Solution& best_solution, std::string& error_message) {
    try {
        if (data.compute_backend != "cuda") {
            throw std::runtime_error("architecture full_gpu requires --compute_backend cuda");
        }
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) throw std::runtime_error("no CUDA device is available");

        const int solution_count = data.p_size * data.num_islands;
        if (solution_count <= 0 || data.customer_num <= 0) {
            throw std::runtime_error("full_gpu requires a non-empty population and customer set");
        }
        const int stride = data.customer_num + 1;
        const std::size_t vector_size = static_cast<std::size_t>(stride);
        const std::size_t matrix_size = vector_size * vector_size;

        std::vector<double> delivery(vector_size);
        std::vector<double> pickup(vector_size);
        std::vector<double> earliest(vector_size);
        std::vector<double> latest(vector_size);
        std::vector<double> service(vector_size);
        std::vector<double> distance(matrix_size);
        std::vector<double> travel_time(matrix_size);
        std::vector<unsigned char> pruning(matrix_size, 1);
        std::vector<double> related(matrix_size, INFINITY);
        std::vector<int> related_rank(matrix_size, 0);
        for (int i = 0; i < stride; ++i) {
            delivery[i] = data.node[i].delivery;
            pickup[i] = data.node[i].pickup;
            earliest[i] = data.node[i].start;
            latest[i] = data.node[i].end;
            service[i] = data.node[i].s_time;
            for (int j = 0; j < stride; ++j) {
                const std::size_t index = static_cast<std::size_t>(i) * stride + j;
                distance[index] = data.dist[i][j];
                travel_time[index] = data.time[i][j];
                if (!data.pm.empty()) pruning[index] = data.pm[i][j] ? 1 : 0;
                if (!data.rm.empty()) related[index] = data.rm[i][j];
                if (!data.rm_argrank.empty() && j < static_cast<int>(data.rm_argrank[i].size())) {
                    related_rank[index] = data.rm_argrank[i][j];
                } else {
                    related_rank[index] = j;
                }
            }
        }

        DeviceBuffer<double> d_delivery(vector_size), d_pickup(vector_size);
        DeviceBuffer<double> d_earliest(vector_size), d_latest(vector_size), d_service(vector_size);
        DeviceBuffer<double> d_distance(matrix_size), d_travel_time(matrix_size);
        DeviceBuffer<unsigned char> d_pruning(matrix_size);
        DeviceBuffer<double> d_related(matrix_size);
        DeviceBuffer<int> d_related_rank(matrix_size);
        cuda_check(cudaMemcpy(d_delivery.get(), delivery.data(), vector_size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(delivery)");
        cuda_check(cudaMemcpy(d_pickup.get(), pickup.data(), vector_size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(pickup)");
        cuda_check(cudaMemcpy(d_earliest.get(), earliest.data(), vector_size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(earliest)");
        cuda_check(cudaMemcpy(d_latest.get(), latest.data(), vector_size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(latest)");
        cuda_check(cudaMemcpy(d_service.get(), service.data(), vector_size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(service)");
        cuda_check(cudaMemcpy(d_distance.get(), distance.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(distance)");
        cuda_check(cudaMemcpy(d_travel_time.get(), travel_time.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(time)");
        cuda_check(cudaMemcpy(d_pruning.get(), pruning.data(), matrix_size * sizeof(unsigned char), cudaMemcpyHostToDevice), "cudaMemcpy(pruning)");
        cuda_check(cudaMemcpy(d_related.get(), related.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(related)");
        cuda_check(cudaMemcpy(d_related_rank.get(), related_rank.data(), matrix_size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(related_rank)");

        DeviceProblem problem{
            data.customer_num, data.DC, stride, data.vehicle.capacity,
            data.start_time, data.vehicle.d_cost, data.vehicle.unit_cost,
            data.all_delivery, data.all_pickup, data.max_dist, data.min_dist,
            d_delivery.get(), d_pickup.get(), d_earliest.get(), d_latest.get(),
            d_service.get(), d_distance.get(), d_travel_time.get(), d_pruning.get(),
            d_related.get(), d_related_rank.get(),
            data.pruning && !data.pm.empty() ? 1 : 0
        };

        best_solution.cost = std::numeric_limits<double>::infinity();
        auto start_wall_time = std::chrono::high_resolution_clock::now();
        float total_gpu_ms = 0.0f;

        DevicePopulationOwner population(solution_count, data.customer_num);
        DevicePopulationOwner branch(solution_count, data.customer_num);
        DevicePopulationOwner best_initialization(solution_count, data.customer_num);
        DevicePopulationOwner global_best(1, data.customer_num);
        DevicePopulationOwner island_bests(data.num_islands, data.customer_num);
        DevicePopulationOwner local_candidate(data.num_islands, data.customer_num);
        DevicePopulationOwner local_perturbation(data.num_islands, data.customer_num);
        DevicePopulationOwner diversification_backup(solution_count, data.customer_num);
        const int migrant_count = std::min(data.migration_size, data.p_size);
        DevicePopulationOwner migrants(data.num_islands * migrant_count, data.customer_num);
        DeviceBuffer<DeviceConstructionConfig> d_configs(solution_count);
        DeviceBuffer<LegacyMt19937> d_rng_states(solution_count);
        DeviceBuffer<int> d_status(solution_count);
        DeviceBuffer<int> d_peer_workspace(solution_count);
        DeviceBuffer<int> d_peer_indices(solution_count);
        DeviceBuffer<int> d_update_seeds(solution_count);
        DeviceBuffer<int> d_accepted_count(1);
        DeviceBuffer<int> d_last_improvement_generation(data.num_islands);
        DeviceBuffer<int> d_population_rank(solution_count);
        DeviceBuffer<int> d_island_indices(data.num_islands);
        DeviceBuffer<LegacyMt19937> d_global_rng(1);
        DeviceBuffer<LegacyMt19937> d_search_rng(data.num_islands);
        const int route_capacity = data.customer_num + 2;
        const int customer_capacity = data.customer_num;
        DeviceBuffer<int> d_sample_pool(static_cast<std::size_t>(solution_count) * customer_capacity);
        DeviceBuffer<int> d_unrouted(static_cast<std::size_t>(solution_count) * customer_capacity);
        DeviceBuffer<int> d_route_nodes(static_cast<std::size_t>(solution_count) * route_capacity);
        DeviceBuffer<int> d_candidate_nodes(static_cast<std::size_t>(solution_count) * route_capacity);
        DeviceBuffer<int> d_candidate_nodes_2(static_cast<std::size_t>(solution_count) * route_capacity);
        DeviceBuffer<int> d_flags(static_cast<std::size_t>(solution_count) * (customer_capacity + 1));
        DeviceBuffer<double> d_load(static_cast<std::size_t>(solution_count) * route_capacity);
        DeviceBuffer<double> d_cd(static_cast<std::size_t>(solution_count) * route_capacity);
        DeviceBuffer<double> d_cp(static_cast<std::size_t>(solution_count) * route_capacity);
        DeviceBuffer<double> d_rd(static_cast<std::size_t>(solution_count) * route_capacity);
        DeviceBuffer<double> d_rp(static_cast<std::size_t>(solution_count) * route_capacity);
        DeviceConstructionScratch scratch{
            route_capacity, customer_capacity, d_sample_pool.get(), d_unrouted.get(),
            d_route_nodes.get(), d_candidate_nodes.get(), d_candidate_nodes_2.get(),
            d_flags.get(), d_load.get(), d_cd.get(),
            d_cp.get(), d_rd.get(), d_rp.get()
        };

        for (int run = 1; run <= data.runs; ++run) {
            std::printf("---------------------------------Run %d---------------------------\n", run);
            std::vector<DeviceConstructionConfig> configs(solution_count);
            std::vector<LegacyMt19937> rng_states;
            rng_states.reserve(solution_count);
            for (int solution_id = 0; solution_id < solution_count; ++solution_id) {
                const int island_id = solution_id / data.p_size;
                const int agent_id = solution_id % data.p_size;
                const std::uint32_t seed = data.num_islands > 1
                    ? static_cast<std::uint32_t>(data.seed + run * 100000 + island_id * 1000 + 100000 + 1000000 + agent_id)
                    : static_cast<std::uint32_t>(data.seed + run * 100000 + 100000 + agent_id);
                rng_states.emplace_back(seed);
                DeviceConstructionConfig config{};
                config.ksize = data.k_init;
                config.insertion_mode_td = data.init == "td" ? 1 : 0;
                if (data.init == "rcrs_random" || data.init == "sa_random") {
                    config.lambda = legacy_randdouble(0.0, 1.0, rng_states.back());
                    config.gamma = legacy_randdouble(0.0, 1.0, rng_states.back());
                } else if (agent_id < static_cast<int>(data.latin.size())) {
                    config.lambda = data.latin[agent_id].first;
                    config.gamma = data.latin[agent_id].second;
                } else if (!data.latin.empty()) {
                    config.lambda = data.latin.back().first;
                    config.gamma = data.latin.back().second;
                } else {
                    config.lambda = data.lambda_gamma.first;
                    config.gamma = data.lambda_gamma.second;
                }
                configs[solution_id] = config;
            }

            cuda_check(cudaMemcpy(d_configs.get(), configs.data(), configs.size() * sizeof(DeviceConstructionConfig), cudaMemcpyHostToDevice), "cudaMemcpy(configs)");
            cuda_check(cudaMemcpy(d_rng_states.get(), rng_states.data(), rng_states.size() * sizeof(LegacyMt19937), cudaMemcpyHostToDevice), "cudaMemcpy(rng_states)");
            LegacyMt19937 global_rng(static_cast<unsigned int>(data.seed + run * 100000));
            cuda_check(cudaMemcpy(d_global_rng.get(), &global_rng, sizeof(global_rng), cudaMemcpyHostToDevice), "cudaMemcpy(global_rng)");
            std::vector<LegacyMt19937> search_rngs;
            search_rngs.reserve(data.num_islands);
            for (int island = 0; island < data.num_islands; ++island) {
                const unsigned int seed = data.num_islands > 1
                    ? static_cast<unsigned int>(data.seed + run * 100000 + island * 1000 + 100000)
                    : static_cast<unsigned int>(data.seed + run * 100000);
                search_rngs.emplace_back(seed);
            }
            cuda_check(cudaMemcpy(
                d_search_rng.get(), search_rngs.data(),
                search_rngs.size() * sizeof(LegacyMt19937), cudaMemcpyHostToDevice
            ), "cudaMemcpy(search_rngs)");
            cuda_check(cudaMemset(
                d_last_improvement_generation.get(), 0,
                data.num_islands * sizeof(int)
            ), "cudaMemset(last_improvement_generation)");

            cudaEvent_t start_event = nullptr;
            cudaEvent_t end_event = nullptr;
            cuda_check(cudaEventCreate(&start_event), "cudaEventCreate(full_gpu start)");
            cuda_check(cudaEventCreate(&end_event), "cudaEventCreate(full_gpu end)");
            cuda_check(cudaEventRecord(start_event), "cudaEventRecord(full_gpu start)");
            const int threads = 64;
            const int blocks = (solution_count + threads - 1) / threads;
            initialize_population_kernel<<<blocks, threads>>>(
                population.view, branch.view, scratch, problem,
                d_configs.get(), d_rng_states.get(), d_status.get()
            );
            cuda_check(cudaGetLastError(), "initialize_population_kernel launch");
            if (data.init == "sa" || data.init == "sa_random") {
                simulated_annealing_initialization_kernel<<<blocks, threads>>>(
                    population.view, branch.view, best_initialization.view, scratch, problem,
                    d_rng_states.get(), d_status.get()
                );
                cuda_check(cudaGetLastError(), "simulated_annealing_initialization_kernel launch");
            }
            initialize_island_bests_kernel<<<(data.num_islands + 63) / 64, 64>>>(
                population.view, island_bests.view, data.p_size
            );
            cuda_check(cudaGetLastError(), "initialize_island_bests_kernel launch");
            initialize_global_best_kernel<<<1, 1>>>(island_bests.view, global_best.view);
            cuda_check(cudaGetLastError(), "initialize_global_best_kernel launch");
            cuda_check(cudaEventRecord(end_event), "cudaEventRecord(full_gpu end)");
            cuda_check(cudaEventSynchronize(end_event), "cudaEventSynchronize(full_gpu initialization)");
            float initialization_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&initialization_ms, start_event, end_event), "cudaEventElapsedTime(full_gpu initialization)");
            cudaEventDestroy(start_event);
            cudaEventDestroy(end_event);

            std::vector<int> status(solution_count);
            cuda_check(cudaMemcpy(status.data(), d_status.get(), status.size() * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(status)");
            if (std::any_of(status.begin(), status.end(), [](int value) { return value != 0; })) {
                throw std::runtime_error("device construction exceeded fixed solution capacity or produced an invalid route");
            }

            float generation_ms = 0.0f;
            DevicePopulation current_population = population.view;
            DevicePopulation next_population = branch.view;
            if (data.max_iter > 0) {
                cuda_check(cudaEventCreate(&start_event), "cudaEventCreate(full_gpu generation start)");
                cuda_check(cudaEventCreate(&end_event), "cudaEventCreate(full_gpu generation end)");
                cuda_check(cudaEventRecord(start_event), "cudaEventRecord(full_gpu generation start)");
                for (int generation = 1; generation <= data.max_iter; ++generation) {
                    const double ratio = static_cast<double>(generation - 1) / data.max_iter;
                    const double a = 2.0 - 2.0 * ratio;
                    const double p_hybrid = fmax(0.15, 0.5 * (1.0 - ratio));
                    const double p_mode = data.hybrid_mode == "sho"
                        ? 1.0
                        : (data.hybrid_mode == "woa" ? 0.0 : p_hybrid);
                    cuda_check(cudaMemset(
                        d_status.get(), 0, status.size() * sizeof(int)
                    ), "cudaMemset(generation status)");
                    cuda_check(cudaMemset(d_accepted_count.get(), 0, sizeof(int)), "cudaMemset(accepted_count)");
                    for (int island = 0; island < data.num_islands; ++island) {
                        const int first = island * data.p_size;
                        DevicePopulation current_island = slice_population(
                            current_population, first, data.p_size
                        );
                        DevicePopulation next_island = slice_population(
                            next_population, first, data.p_size
                        );
                        DevicePopulation temporary_island = slice_population(
                            best_initialization.view, first, data.p_size
                        );
                        DevicePopulation diversification_backup_island = slice_population(
                            diversification_backup.view, first, data.p_size
                        );
                        DevicePopulation island_best = slice_population(island_bests.view, island, 1);
                        DeviceConstructionScratch island_scratch = slice_scratch(scratch, first);
                        prepare_generation_kernel<<<1, 1>>>(
                            current_island, d_search_rng.get() + island,
                            d_peer_workspace.get() + first, d_peer_indices.get() + first,
                            d_update_seeds.get() + first
                        );
                        cuda_check(cudaGetLastError(), "prepare_generation_kernel launch");
                        const int island_blocks = (data.p_size + threads - 1) / threads;
                        update_population_kernel<<<island_blocks, threads>>>(
                            current_island, next_island, temporary_island,
                            island_best, island_scratch, problem, d_peer_indices.get() + first,
                            d_update_seeds.get() + first, p_mode, a, generation - 1,
                            data.max_iter, data.sho_mutation_prob, d_accepted_count.get(),
                            d_status.get() + first
                        );
                        cuda_check(cudaGetLastError(), "update_population_kernel launch");
                        update_global_best_kernel<<<1, 1>>>(
                            next_island, island_best, generation,
                            d_last_improvement_generation.get() + island
                        );
                        cuda_check(cudaGetLastError(), "update island best kernel launch");
                        if (generation % data.local_search_interval == 0) {
                            DevicePopulation candidate = slice_population(local_candidate.view, island, 1);
                            DevicePopulation perturbation = slice_population(local_perturbation.view, island, 1);
                            prepare_local_search_kernel<<<1, 32>>>(island_best, candidate);
                            cuda_check(cudaGetLastError(), "prepare_local_search_kernel launch");
                            deep_local_search_kernel<<<1, 64>>>(
                                candidate, perturbation, temporary_island,
                                island_scratch, problem, data.or_opt_len, data.ex_len,
                                data.elo, static_cast<unsigned int>(data.seed + run * 100000 + island * 1000),
                                data.removal_lower, data.removal_upper, d_status.get() + island
                            );
                            cuda_check(cudaGetLastError(), "deep_local_search_kernel launch");
                            commit_local_search_and_inject_kernel<<<1, 32>>>(
                                next_island, island_best, candidate, generation,
                                d_last_improvement_generation.get() + island,
                                d_status.get() + island
                            );
                            cuda_check(cudaGetLastError(), "commit_local_search_and_inject_kernel launch");
                        }
                        if (generation % data.stagnation_interval == 0) {
                            diversify_population_kernel<<<1, 64>>>(
                                next_island, temporary_island, diversification_backup_island,
                                island_best, island_scratch,
                                problem, d_search_rng.get() + island,
                                d_population_rank.get() + first, generation,
                                d_last_improvement_generation.get() + island,
                                data.diversify_ratio, data.removal_lower, data.removal_upper,
                                data.stagnation_interval, d_status.get() + island
                            );
                            cuda_check(cudaGetLastError(), "diversify_population_kernel launch");
                        }
                    }
                    update_global_from_islands_kernel<<<1, 1>>>(island_bests.view, global_best.view);
                    cuda_check(cudaGetLastError(), "update_global_from_islands_kernel launch");
                    if (data.num_islands > 1 && generation % data.migration_interval == 0) {
                        const int migration_mode = data.migration_mode == "ring"
                            ? 0 : (data.migration_mode == "broadcast" ? 1 : 2);
                        migrate_islands_kernel<<<1, 1>>>(
                            next_population, island_bests.view, migrants.view,
                            data.p_size, migrant_count, migration_mode,
                            d_population_rank.get(), d_island_indices.get(), d_global_rng.get()
                        );
                        cuda_check(cudaGetLastError(), "migrate_islands_kernel launch");
                    }
                    const DevicePopulation temporary_population = current_population;
                    current_population = next_population;
                    next_population = temporary_population;
                }
                cuda_check(cudaEventRecord(end_event), "cudaEventRecord(full_gpu generation end)");
                cuda_check(cudaEventSynchronize(end_event), "cudaEventSynchronize(full_gpu generation)");
                cuda_check(cudaEventElapsedTime(&generation_ms, start_event, end_event), "cudaEventElapsedTime(full_gpu generation)");
                cudaEventDestroy(start_event);
                cudaEventDestroy(end_event);
                cuda_check(cudaMemcpy(status.data(), d_status.get(), status.size() * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(generation status)");
                const auto failure = std::find_if(status.begin(), status.end(), [](int value) { return value != 0; });
                if (failure != status.end()) {
                    const int solution_id = static_cast<int>(std::distance(status.begin(), failure));
                    throw std::runtime_error(
                        "full_gpu generation failed: solution_id=" + std::to_string(solution_id) +
                        ", status_code=" + std::to_string(*failure)
                    );
                }
            }

            total_gpu_ms += (initialization_ms + generation_ms);
            Solution run_best = decode_solution(global_best, 0, data);
            if (run_best.cost < best_solution.cost) {
                best_solution = run_best;
            }
            std::printf("Run %d finishes\n", run);
        }

        auto elapsed_wall = std::chrono::high_resolution_clock::now() - start_wall_time;
        int consumed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed_wall).count();

        std::printf("------------Summary-----------\n");
        std::printf("Total %d runs, total consumed %d sec\n", data.runs, consumed_sec);
        best_solution.output(data);
        if (!best_solution.check(data)) {
            throw std::runtime_error("CPU final verification rejected the full-GPU solution");
        }
        return true;
    } catch (const std::exception& exception) {
        error_message = exception.what();
        return false;
    }
}
