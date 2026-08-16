#ifdef USE_CUDA

#include <cuda_runtime.h>
#include "compute_backend.h"
#include "profile.h"
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>

struct DeviceAttr {
    int num_cus = 0;
    double dist = 0.0;
    int s = 0;
    int e = 0;
    double T_D = 0.0;
    double T_E = 0.0;
    double T_L = 0.0;
    double C_E = 0.0;
    double C_H = 0.0;
    double C_L = 0.0;
};

class CudaComputeBackend : public BaseComputeBackend {
private:
    struct InsertionTimingModel {
        int cpu_samples = 0;
        int gpu_samples = 0;
        double cpu_base_us = 25.0;
        double cpu_request_us = 0.35;
        double cpu_node_us = 0.02;
        double gpu_base_us = 900.0;
        double gpu_request_us = 0.05;
        double gpu_node_us = 0.004;
    };

    struct RouteTimingModel {
        int cpu_samples = 0;
        int gpu_samples = 0;
        double cpu_base_us = 20.0;
        double cpu_route_us = 1.0;
        double cpu_node_us = 0.05;
        double gpu_base_us = 400.0;
        double gpu_route_us = 0.2;
        double gpu_node_us = 0.01;
    };

    struct PendingInsertionCall {
        std::vector<std::vector<int>> routes;
        std::vector<InsertionRequest> requests;
        std::vector<InsertionScore> results;
        std::exception_ptr error;
        bool done = false;
    };

    struct PendingRouteCall {
        std::vector<std::vector<int>> routes;
        std::vector<RouteEval> results;
        std::exception_ptr error;
        bool done = false;
    };

    const Data& data;
    BackendSnapshot snapshot;
    mutable std::mutex gpu_mutex;
    std::mutex insertion_broker_mutex;
    std::condition_variable insertion_broker_cv;
    std::vector<std::shared_ptr<PendingInsertionCall>> insertion_broker_queue;
    bool insertion_broker_active = false;
    mutable std::mutex insertion_timing_mutex;
    InsertionTimingModel insertion_timing;
    std::mutex route_broker_mutex;
    std::condition_variable route_broker_cv;
    std::vector<std::shared_ptr<PendingRouteCall>> route_broker_queue;
    bool route_broker_active = false;
    mutable std::mutex route_timing_mutex;
    RouteTimingModel route_timing;
    size_t adaptive_route_min_items = 4096;
    size_t adaptive_route_min_nodes = 12288;
    size_t adaptive_insertion_min_items = 8192;
    size_t adaptive_insertion_min_nodes = 1;

    // Host staging arrays are reused under gpu_mutex. This removes repeated
    // vector allocation/reallocation from the insertion hot path.
    std::vector<int> h_insert_batch_routes_flat;
    std::vector<int> h_insert_batch_route_offsets;
    std::vector<int> h_insert_batch_route_lengths;
    std::vector<int> h_insert_batch_request_ids;
    std::vector<int> h_insert_batch_request_route_ids;
    std::vector<int> h_insert_batch_request_customer_ids;
    std::vector<int> h_insert_batch_request_positions;
    std::vector<DeviceAttr> h_insert_batch_prefix_attrs;
    std::vector<DeviceAttr> h_insert_batch_suffix_attrs;
    std::vector<double> h_insert_batch_base_distances;

    double* d_delivery = nullptr;
    double* d_pickup = nullptr;
    double* d_start = nullptr;
    double* d_end = nullptr;
    double* d_service = nullptr;
    double* d_dist = nullptr;
    double* d_time = nullptr;

    int* d_route_batch = nullptr;
    int* d_route_offsets = nullptr;
    int* d_route_lengths = nullptr;
    int* d_route_feasible = nullptr;
    double* d_route_costs = nullptr;
    size_t route_batch_node_capacity = 0;
    size_t route_offsets_capacity = 0;
    size_t route_lengths_capacity = 0;
    size_t route_feasible_capacity = 0;
    size_t route_costs_capacity = 0;
    int* h_route_batch_pinned = nullptr;
    int* h_route_offsets_pinned = nullptr;
    int* h_route_lengths_pinned = nullptr;
    int* h_route_feasible_pinned = nullptr;
    double* h_route_costs_pinned = nullptr;
    size_t h_route_batch_capacity = 0;
    size_t h_route_offsets_capacity = 0;
    size_t h_route_lengths_capacity = 0;
    size_t h_route_feasible_capacity = 0;
    size_t h_route_costs_capacity = 0;
    cudaStream_t coordinator_stream = nullptr;

    int* d_insert_route = nullptr;
    int* d_insert_candidates = nullptr;
    int* d_insert_feasible = nullptr;
    double* d_insert_costs = nullptr;
    size_t insert_route_capacity = 0;
    size_t insert_candidate_capacity = 0;
    size_t insert_feasible_capacity = 0;
    size_t insert_costs_capacity = 0;

    int* d_insert_batch_routes_flat = nullptr;
    int* d_insert_batch_route_offsets = nullptr;
    int* d_insert_batch_route_lengths = nullptr;
    int* d_insert_batch_request_ids = nullptr;
    int* d_insert_batch_request_route_ids = nullptr;
    int* d_insert_batch_request_customer_ids = nullptr;
    int* d_insert_batch_request_positions = nullptr;
    int* d_insert_batch_out_request_ids = nullptr;
    int* d_insert_batch_out_feasible = nullptr;
    int* d_insert_batch_out_flags = nullptr;
    double* d_insert_batch_out_delta_distance = nullptr;
    double* d_insert_batch_out_total_distance = nullptr;
    double* d_insert_batch_out_fitness = nullptr;
    DeviceAttr* d_insert_batch_prefix_attrs = nullptr;
    DeviceAttr* d_insert_batch_suffix_attrs = nullptr;
    double* d_insert_batch_base_distances = nullptr;
    size_t insert_batch_routes_flat_capacity = 0;
    size_t insert_batch_route_offsets_capacity = 0;
    size_t insert_batch_route_lengths_capacity = 0;
    size_t insert_batch_request_ids_capacity = 0;
    size_t insert_batch_request_route_ids_capacity = 0;
    size_t insert_batch_request_customer_ids_capacity = 0;
    size_t insert_batch_request_positions_capacity = 0;
    size_t insert_batch_out_request_ids_capacity = 0;
    size_t insert_batch_out_feasible_capacity = 0;
    size_t insert_batch_out_flags_capacity = 0;
    size_t insert_batch_out_delta_distance_capacity = 0;
    size_t insert_batch_out_total_distance_capacity = 0;
    size_t insert_batch_out_fitness_capacity = 0;
    size_t insert_batch_prefix_attrs_capacity = 0;
    size_t insert_batch_suffix_attrs_capacity = 0;
    size_t insert_batch_base_distances_capacity = 0;
    unsigned char* h_insert_batch_input_pinned = nullptr;
    unsigned char* h_insert_batch_output_pinned = nullptr;
    size_t h_insert_batch_input_capacity = 0;
    size_t h_insert_batch_output_capacity = 0;
    cudaEvent_t route_stage_events[4]{};
    cudaEvent_t insertion_stage_events[4]{};

    int* d_solution_routes_flat = nullptr;
    int* d_solution_route_offsets = nullptr;
    int* d_solution_route_lengths = nullptr;
    int* d_solution_offsets = nullptr;
    int* d_solution_route_counts = nullptr;
    int* d_solution_eval_ids = nullptr;
    int* d_solution_seen = nullptr;
    int* d_solution_feasible = nullptr;
    int* d_solution_vehicle_counts = nullptr;
    int* d_solution_violation_flags = nullptr;
    double* d_solution_fitness = nullptr;
    double* d_solution_distance = nullptr;
    size_t solution_routes_flat_capacity = 0;
    size_t solution_route_offsets_capacity = 0;
    size_t solution_route_lengths_capacity = 0;
    size_t solution_offsets_capacity = 0;
    size_t solution_route_counts_capacity = 0;
    size_t solution_eval_ids_capacity = 0;
    size_t solution_feasible_capacity = 0;
    size_t solution_vehicle_counts_capacity = 0;
    size_t solution_violation_flags_capacity = 0;
    size_t solution_fitness_capacity = 0;
    size_t solution_distance_capacity = 0;
    size_t solution_seen_capacity = 0;

    void ensure_device_cache();
    void free_device_cache();
    void ensure_route_batch_capacity(size_t num_routes, size_t max_len);
    void ensure_insert_capacity(size_t route_len, size_t num_candidates);
    void ensure_insert_batch_capacity(size_t flat_nodes, size_t route_count, size_t request_count);
    void ensure_insert_attr_capacity(size_t attr_count, size_t route_count);
    void ensure_insert_pinned_capacity(size_t input_bytes, size_t output_bytes);
    void ensure_solution_capacity(const EncodedPopulation& encoded);
    void calibrate_execution_policy();
    std::vector<InsertionScore> evaluate_insertion_batch_direct(
        const std::vector<std::vector<int>>& routes,
        const std::vector<InsertionRequest>& requests
    );
    std::vector<InsertionScore> evaluate_insertion_batch_brokered(
        const std::vector<std::vector<int>>& routes,
        const std::vector<InsertionRequest>& requests
    );
    std::vector<RouteEval> evaluate_route_batch_direct(const std::vector<std::vector<int>>& routes);
    std::vector<RouteEval> evaluate_route_batch_brokered(const std::vector<std::vector<int>>& routes);
    bool insertion_scheduler_prefers_cpu(
        const std::vector<std::vector<int>>& routes,
        const std::vector<InsertionRequest>& requests
    ) const;
    void observe_insertion_cpu(double elapsed_us, size_t request_count, size_t route_nodes);
    void observe_insertion_gpu(double elapsed_us, size_t request_count, size_t route_nodes);

    bool route_scheduler_prefers_cpu(size_t num_routes, size_t route_work) const;
    void observe_route_cpu(double elapsed_us, size_t num_routes, size_t route_work);
    void observe_route_gpu(double elapsed_us, size_t num_routes, size_t route_work);

public:
    CudaComputeBackend(const Data& d);
    ~CudaComputeBackend() override;
    bool is_gpu_backend() const override { return true; }
    RouteEval evaluate_route(const std::vector<int>& route) override;
    std::vector<RouteEval> evaluate_routes(const std::vector<std::vector<int>>& routes) override;
    void evaluate_insertions(
        const std::vector<int>& route,
        const std::vector<int>& candidates,
        std::vector<int>& out_feasible,
        std::vector<double>& out_costs
    ) override;
    std::vector<InsertionScore> evaluate_insertion_batch(
        const std::vector<std::vector<int>>& routes,
        const std::vector<InsertionRequest>& requests
    ) override;
    std::vector<SolutionEval> evaluate_solutions(const EncodedPopulation& encoded) override;
    EvalTarget choose_target(const WorkShape& shape) const override;

private:
    // Fixed-capacity population slots keep one solution's route changes from
    // shifting every following solution in the device mirror.
    size_t solution_slot_node_capacity = 0;
    size_t solution_slot_route_capacity = 0;
    int persistent_solution_count = 0;
    std::vector<int> h_persistent_route_offsets;
    std::vector<int> h_persistent_route_lengths;
    std::vector<int> h_persistent_solution_offsets;
    std::vector<int> h_persistent_solution_route_counts;
    std::vector<std::uint64_t> h_persistent_solution_revisions;
    std::vector<SolutionEval> h_persistent_solution_results;
};

BaseComputeBackend* create_cuda_backend(const Data& data) {
    return new CudaComputeBackend(data);
}
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

template <typename T>
void ensure_device_buffer(T*& ptr, size_t& capacity, size_t needed, const char* what) {
    if (needed <= capacity && ptr != nullptr) {
        return;
    }
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
    if (needed == 0) {
        capacity = 0;
        return;
    }
    cuda_check(cudaMalloc((void**)&ptr, needed * sizeof(T)), what);
    capacity = needed;
}

template <typename T>
void ensure_pinned_buffer(T*& ptr, size_t& capacity, size_t needed, const char* what) {
    if (needed <= capacity && ptr != nullptr) return;
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
        ptr = nullptr;
    }
    if (needed == 0) {
        capacity = 0;
        return;
    }
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&ptr), needed * sizeof(T), cudaHostAllocDefault), what);
    capacity = needed;
}

size_t reserve_aligned_bytes(size_t& cursor, size_t count, size_t element_size, size_t alignment) {
    cursor = (cursor + alignment - 1) & ~(alignment - 1);
    const size_t offset = cursor;
    cursor += count * element_size;
    return offset;
}
} // namespace

static DeviceAttr host_attr_for_node(const BackendSnapshot& snapshot, int node) {
    DeviceAttr attr;
    attr.s = node;
    attr.e = node;
    if (node == snapshot.depot) {
        attr.num_cus = 0;
        attr.T_E = snapshot.start_time;
        attr.T_L = snapshot.end_time;
        return attr;
    }
    attr.num_cus = 1;
    attr.T_D = snapshot.service[node];
    attr.T_E = snapshot.start[node];
    attr.T_L = snapshot.end[node];
    attr.C_E = snapshot.delivery[node];
    attr.C_L = snapshot.pickup[node];
    attr.C_H = std::max(attr.C_E, attr.C_L);
    return attr;
}

static bool host_connect_attr(
    const BackendSnapshot& snapshot,
    const DeviceAttr& a,
    const DeviceAttr& b,
    DeviceAttr& out
) {
    const int stride = snapshot.customer_num + 1;
    const double edge_dist = snapshot.dist[a.e * stride + b.s];
    const double edge_time = snapshot.time[a.e * stride + b.s];
    if (a.T_E + a.T_D + edge_time - b.T_L > 0.0 ||
        std::max(a.C_H + b.C_E, a.C_L + b.C_H) > snapshot.capacity) {
        return false;
    }

    out.num_cus = a.num_cus + b.num_cus;
    out.dist = a.dist + edge_dist + b.dist;
    const double delta = a.T_D + edge_time;
    const double waiting = std::max(b.T_E - delta - a.T_L, 0.0);
    out.T_D = a.T_D + b.T_D + edge_time + waiting;
    out.T_E = std::max(b.T_E - delta, a.T_E) - waiting;
    out.T_L = std::min(b.T_L - delta, a.T_L);
    out.C_E = a.C_E + b.C_E;
    out.C_H = std::max(a.C_H + b.C_E, a.C_L + b.C_H);
    out.C_L = a.C_L + b.C_L;
    out.s = a.s;
    out.e = b.e;
    return true;
}

__device__ bool device_connect_attr(
    const DeviceAttr& a,
    const DeviceAttr& b,
    double edge_dist,
    double edge_time,
    double capacity,
    DeviceAttr& out
) {
    if (a.T_E + a.T_D + edge_time - b.T_L > 0.0 ||
        max(a.C_H + b.C_E, a.C_L + b.C_H) > capacity) {
        return false;
    }
    out.num_cus = a.num_cus + b.num_cus;
    out.dist = a.dist + edge_dist + b.dist;
    const double delta = a.T_D + edge_time;
    const double waiting = max(b.T_E - delta - a.T_L, 0.0);
    out.T_D = a.T_D + b.T_D + edge_time + waiting;
    out.T_E = max(b.T_E - delta, a.T_E) - waiting;
    out.T_L = min(b.T_L - delta, a.T_L);
    out.C_E = a.C_E + b.C_E;
    out.C_H = max(a.C_H + b.C_E, a.C_L + b.C_H);
    out.C_L = a.C_L + b.C_L;
    out.s = a.s;
    out.e = b.e;
    return true;
}

__device__ int device_attr_violation(
    const DeviceAttr& a,
    const DeviceAttr& b,
    double edge_time,
    double capacity
) {
    int flags = INSERTION_OK;
    if (a.T_E + a.T_D + edge_time - b.T_L > 0.0) {
        flags |= INSERTION_TIME_WINDOW;
    }
    if (max(a.C_H + b.C_E, a.C_L + b.C_H) > capacity) {
        flags |= INSERTION_CAPACITY;
    }
    return flags;
}

// --- CUDA Kernels ---

__global__ void evaluate_route_batch_kernel(
    const int* routes,
    const int* offsets,
    const int* lengths,
    const int num_routes,
    const int stride,
    const int depot,
    const double start_time,
    const double capacity,
    const double dispatch_cost,
    const double unit_cost,
    const double* delivery,
    const double* pickup,
    const double* start,
    const double* end,
    const double* service,
    const double* dist,
    const double* time_matrix,
    int* out_feasible,
    double* out_costs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_routes) return;

    int length = lengths[idx];
    if (length < 2) {
        out_feasible[idx] = 0;
        out_costs[idx] = 0.0;
        return;
    }

    const int offset = offsets[idx];
    int first_node = routes[offset];
    int last_node = routes[offset + length - 1];

    if (first_node != depot || last_node != depot) {
        out_feasible[idx] = 0;
        out_costs[idx] = 0.0;
        return;
    }

    if (length == 2) {
        out_feasible[idx] = 1;
        out_costs[idx] = 0.0;
        return;
    }

    double load = 0.0;
    for (int pos = 1; pos < length - 1; ++pos) {
        int node = routes[offset + pos];
        load += delivery[node];
    }

    if (load > capacity) {
        out_feasible[idx] = 0;
        out_costs[idx] = 0.0;
        return;
    }

    double distance = 0.0;
    double time_val = start_time;
    int prev = first_node;

    for (int pos = 1; pos < length; ++pos) {
        int node = routes[offset + pos];
        load = load - delivery[node] + pickup[node];
        if (load < 0.0 || load > capacity) {
            out_feasible[idx] = 0;
            out_costs[idx] = 0.0;
            return;
        }

        time_val += time_matrix[prev * stride + node];
        if (time_val > end[node]) {
            out_feasible[idx] = 0;
            out_costs[idx] = 0.0;
            return;
        }
        if (time_val < start[node]) {
            time_val = start[node];
        }
        time_val += service[node];

        distance += dist[prev * stride + node];
        prev = node;
    }

    out_feasible[idx] = 1;
    out_costs[idx] = dispatch_cost + distance * unit_cost;
}

__global__ void evaluate_insertions_cuda_kernel(
    const int* route,
    const int route_len,
    const int* candidates,
    const int num_candidates,
    const int depot,
    const double start_time,
    const double capacity,
    const double dispatch_cost,
    const double unit_cost,
    const double* delivery,
    const double* pickup,
    const double* start,
    const double* end,
    const double* service,
    const double* dist,
    const double* time_matrix,
    const int stride,
    int* out_feasible,
    double* out_costs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_positions = route_len - 1;
    int total_evals = num_candidates * num_positions;
    if (idx >= total_evals) return;

    int c_idx = idx / num_positions;
    int pos = (idx % num_positions) + 1;
    int candidate = candidates[c_idx];

    if (route[0] != depot || route[route_len - 1] != depot) {
        out_feasible[idx] = 0;
        out_costs[idx] = 0.0;
        return;
    }

    double base_load = 0.0;
    for (int i = 1; i < route_len - 1; ++i) {
        base_load += delivery[route[i]];
    }
    base_load += delivery[candidate];
    if (base_load > capacity) {
        out_feasible[idx] = 0;
        out_costs[idx] = 0.0;
        return;
    }

    double load = base_load;
    double distance = 0.0;
    double time_val = start_time;
    int prev = route[0];
    bool is_feasible = true;

    for (int i = 1; i <= route_len; ++i) {
        int node = 0;
        if (i == pos) {
            node = candidate;
        } else if (i < pos) {
            node = route[i];
        } else {
            node = route[i - 1];
        }

        load = load - delivery[node] + pickup[node];
        if (load < 0.0 || load > capacity) {
            is_feasible = false;
            break;
        }

        time_val += time_matrix[prev * stride + node];
        if (time_val > end[node]) {
            is_feasible = false;
            break;
        }
        if (time_val < start[node]) {
            time_val = start[node];
        }
        time_val += service[node];

        distance += dist[prev * stride + node];
        prev = node;
    }

    if (is_feasible) {
        out_feasible[idx] = 1;
        out_costs[idx] = dispatch_cost + distance * unit_cost;
    } else {
        out_feasible[idx] = 0;
        out_costs[idx] = 0.0;
    }
}

__global__ void evaluateInsertionBatchKernel(
    const int* routes_flat,
    const int* route_offsets,
    const int* route_lengths,
    const int route_count,
    const int* request_ids,
    const int* request_route_ids,
    const int* request_customer_ids,
    const int* request_positions,
    const int request_count,
    const int customer_count,
    const int depot,
    const double start_time,
    const double capacity,
    const double dispatch_cost,
    const double unit_cost,
    const double* delivery,
    const double* pickup,
    const double* start,
    const double* end,
    const double* service,
    const double* dist,
    const double* time_matrix,
    const int stride,
    int* out_request_ids,
    int* out_feasible,
    double* out_delta_distance,
    double* out_total_distance,
    double* out_fitness,
    int* out_flags
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= request_count) return;

    int req_id = request_ids[idx];
    int route_id = request_route_ids[idx];
    int customer = request_customer_ids[idx];
    int insert_position = request_positions[idx];
    out_request_ids[idx] = req_id;
    out_feasible[idx] = 0;
    out_delta_distance[idx] = 0.0;
    out_total_distance[idx] = 0.0;
    out_fitness[idx] = 0.0;

    if (route_id < 0 || route_id >= route_count) {
        out_flags[idx] = INSERTION_BAD_ROUTE;
        return;
    }
    if (customer <= 0 || customer > customer_count || customer == depot) {
        out_flags[idx] = INSERTION_BAD_CUSTOMER;
        return;
    }

    int offset = route_offsets[route_id];
    int route_len = route_lengths[route_id];
    if (route_len < 2) {
        out_flags[idx] = INSERTION_BAD_ROUTE;
        return;
    }
    if (insert_position < 1 || insert_position >= route_len) {
        out_flags[idx] = INSERTION_BAD_POSITION;
        return;
    }

    int first = routes_flat[offset];
    int last = routes_flat[offset + route_len - 1];
    if (first != depot || last != depot) {
        out_flags[idx] = INSERTION_BAD_ROUTE;
        return;
    }

    double base_distance = 0.0;
    double load = delivery[customer];
    int flags = INSERTION_OK;
    for (int pos = 1; pos < route_len; ++pos) {
        int prev_node = routes_flat[offset + pos - 1];
        int node = routes_flat[offset + pos];
        if (prev_node < 0 || prev_node > customer_count || node < 0 || node > customer_count) {
            flags |= INSERTION_BAD_CUSTOMER;
            break;
        }
        base_distance += dist[prev_node * stride + node];
        if (pos < route_len - 1) {
            load += delivery[node];
        }
    }
    if (flags != INSERTION_OK) {
        out_flags[idx] = flags;
        return;
    }
    if (load > capacity) {
        out_flags[idx] = INSERTION_CAPACITY;
        return;
    }

    double route_distance = 0.0;
    double time_val = start_time;
    int prev = first;

    for (int pos = 1; pos <= route_len; ++pos) {
        int node = depot;
        if (pos == insert_position) {
            node = customer;
        } else if (pos < insert_position) {
            node = routes_flat[offset + pos];
        } else {
            node = routes_flat[offset + pos - 1];
        }

        if (node < 0 || node > customer_count) {
            flags |= INSERTION_BAD_CUSTOMER;
            break;
        }

        load = load - delivery[node] + pickup[node];
        if (load < 0.0 || load > capacity) {
            flags |= INSERTION_CAPACITY;
            break;
        }

        time_val += time_matrix[prev * stride + node];
        if (time_val > end[node]) {
            flags |= INSERTION_TIME_WINDOW;
            break;
        }
        if (time_val < start[node]) {
            time_val = start[node];
        }
        time_val += service[node];

        route_distance += dist[prev * stride + node];
        prev = node;
    }

    out_flags[idx] = flags;
    if (flags == INSERTION_OK) {
        out_feasible[idx] = 1;
        out_delta_distance[idx] = route_distance - base_distance;
        out_total_distance[idx] = route_distance;
        out_fitness[idx] = dispatch_cost + route_distance * unit_cost;
    }
}

__global__ void evaluateInsertionBatchAttrKernel(
    const int* route_lengths,
    int route_count,
    int max_route_len,
    const int* request_ids,
    const int* request_route_ids,
    const int* request_customer_ids,
    const int* request_positions,
    int request_count,
    int customer_num,
    int depot,
    double capacity,
    double dispatch_cost,
    double unit_cost,
    const double* delivery,
    const double* pickup,
    const double* start,
    const double* end,
    const double* service,
    const double* dist,
    const double* time,
    int stride,
    const DeviceAttr* prefix_attrs,
    const DeviceAttr* suffix_attrs,
    const double* base_distances,
    int* out_request_ids,
    int* out_feasible,
    double* out_delta_distance,
    double* out_total_distance,
    double* out_fitness,
    int* out_flags
) {
    int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count) return;

    out_request_ids[request_index] = request_ids[request_index];
    out_feasible[request_index] = 0;
    out_delta_distance[request_index] = 0.0;
    out_total_distance[request_index] = 0.0;
    out_fitness[request_index] = 0.0;
    out_flags[request_index] = INSERTION_OK;

    int route_id = request_route_ids[request_index];
    int customer = request_customer_ids[request_index];
    int position = request_positions[request_index];
    if (route_id < 0 || route_id >= route_count) {
        out_flags[request_index] = INSERTION_BAD_ROUTE;
        return;
    }
    if (customer <= 0 || customer > customer_num || customer == depot) {
        out_flags[request_index] = INSERTION_BAD_CUSTOMER;
        return;
    }
    int route_len = route_lengths[route_id];
    if (route_len < 2 || position < 1 || position >= route_len) {
        out_flags[request_index] = (route_len < 2) ? INSERTION_BAD_ROUTE : INSERTION_BAD_POSITION;
        return;
    }

    int attr_base = route_id * max_route_len;
    DeviceAttr customer_attr;
    customer_attr.num_cus = 1;
    customer_attr.s = customer;
    customer_attr.e = customer;
    customer_attr.T_D = service[customer];
    customer_attr.T_E = start[customer];
    customer_attr.T_L = end[customer];
    customer_attr.C_E = delivery[customer];
    customer_attr.C_L = pickup[customer];
    customer_attr.C_H = max(customer_attr.C_E, customer_attr.C_L);

    DeviceAttr merged;
    const DeviceAttr& prefix = prefix_attrs[attr_base + position - 1];
    const DeviceAttr& suffix = suffix_attrs[attr_base + position];
    int flags = device_attr_violation(
        prefix,
        customer_attr,
        time[prefix.e * stride + customer_attr.s],
        capacity
    );
    if (flags != INSERTION_OK) {
        out_flags[request_index] = flags;
        return;
    }
    device_connect_attr(
        prefix,
        customer_attr,
        dist[prefix.e * stride + customer_attr.s],
        time[prefix.e * stride + customer_attr.s],
        capacity,
        merged
    );

    flags = device_attr_violation(
        merged,
        suffix,
        time[merged.e * stride + suffix.s],
        capacity
    );
    if (flags != INSERTION_OK) {
        out_flags[request_index] = flags;
        return;
    }
    DeviceAttr result;
    device_connect_attr(
        merged,
        suffix,
        dist[merged.e * stride + suffix.s],
        time[merged.e * stride + suffix.s],
        capacity,
        result
    );
    out_feasible[request_index] = 1;
    out_total_distance[request_index] = result.dist;
    out_delta_distance[request_index] = result.dist - base_distances[route_id];
    out_fitness[request_index] = dispatch_cost + result.dist * unit_cost;
}

__global__ void evaluateSolutionsKernel(
    const int* routes_flat,
    const int* route_offsets,
    const int* route_lengths,
    const int* solution_offsets,
    const int* solution_route_counts,
    const int* eval_solution_ids,
    const int eval_count,
    const int customer_count,
    const int depot,
    const double start_time,
    const double capacity,
    const double dispatch_cost,
    const double unit_cost,
    const double* delivery,
    const double* pickup,
    const double* start,
    const double* end,
    const double* service,
    const double* dist,
    const double* time_matrix,
    const int stride,
    int* seen,
    int* out_feasible,
    double* out_fitness,
    int* out_vehicle_counts,
    double* out_distance,
    int* out_violation_flags
) {
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= eval_count) return;
    const int solution_id = eval_solution_ids[output_index];

    int seen_base = solution_id * (customer_count + 1);
    for (int node = 0; node <= customer_count; ++node) {
        seen[seen_base + node] = 0;
    }

    int flags = SOLUTION_OK;
    int vehicle_count = 0;
    double total_distance = 0.0;
    int route_start = solution_offsets[solution_id];
    int route_count = solution_route_counts[solution_id];

    for (int r = 0; r < route_count; ++r) {
        int route_id = route_start + r;
        int offset = route_offsets[route_id];
        int length = route_lengths[route_id];

        if (length < 2) {
            flags |= SOLUTION_BAD_DEPOT;
            continue;
        }

        int first = routes_flat[offset];
        int last = routes_flat[offset + length - 1];
        if (first != depot || last != depot) {
            flags |= SOLUTION_BAD_DEPOT;
            continue;
        }

        if (length == 2) {
            continue;
        }

        vehicle_count += 1;
        double load = 0.0;
        for (int pos = 1; pos < length - 1; ++pos) {
            int node = routes_flat[offset + pos];
            if (node < 0 || node > customer_count || node == depot) {
                flags |= SOLUTION_BAD_NODE;
                continue;
            }
            load += delivery[node];
            int seen_idx = seen_base + node;
            seen[seen_idx] += 1;
            if (seen[seen_idx] > 1) {
                flags |= SOLUTION_DUPLICATE_CUSTOMER;
            }
        }
        if (load > capacity) {
            flags |= SOLUTION_CAPACITY;
        }

        double time_val = start_time;
        int prev = first;
        for (int pos = 1; pos < length; ++pos) {
            int node = routes_flat[offset + pos];
            if (node < 0 || node > customer_count) {
                flags |= SOLUTION_BAD_NODE;
                break;
            }

            load = load - delivery[node] + pickup[node];
            if (load < 0.0 || load > capacity) {
                flags |= SOLUTION_CAPACITY;
            }

            time_val += time_matrix[prev * stride + node];
            if (time_val > end[node]) {
                flags |= SOLUTION_TIME_WINDOW;
            }
            if (time_val < start[node]) {
                time_val = start[node];
            }
            time_val += service[node];

            total_distance += dist[prev * stride + node];
            prev = node;
        }
    }

    for (int node = 0; node <= customer_count; ++node) {
        if (node == depot) {
            continue;
        }
        if (seen[seen_base + node] == 0) {
            flags |= SOLUTION_MISSING_CUSTOMER;
        }
    }

    out_vehicle_counts[output_index] = vehicle_count;
    out_distance[output_index] = total_distance;
    out_fitness[output_index] = dispatch_cost * vehicle_count + unit_cost * total_distance;
    out_violation_flags[output_index] = flags;
    out_feasible[output_index] = (flags == SOLUTION_OK) ? 1 : 0;
}

// --- CudaComputeBackend Implementation ---

CudaComputeBackend::CudaComputeBackend(const Data& d) : data(d), snapshot(d) {
    cudaError_t err = cudaSuccess;
    int device_count = 0;
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        throw std::runtime_error("CUDA device count error or no devices found");
    }

    cuda_check(cudaStreamCreateWithFlags(&coordinator_stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(coordinator_stream)");
    for (cudaEvent_t& event : route_stage_events) {
        cuda_check(cudaEventCreate(&event), "cudaEventCreate(route_stage_event)");
    }
    for (cudaEvent_t& event : insertion_stage_events) {
        cuda_check(cudaEventCreate(&event), "cudaEventCreate(insertion_stage_event)");
    }
    ensure_device_cache();
    if (data.execution_policy == "adaptive") {
        calibrate_execution_policy();
        profile_reset();
    }
}

CudaComputeBackend::~CudaComputeBackend() {
    free_device_cache();
    for (cudaEvent_t& event : route_stage_events) {
        if (event != nullptr) cudaEventDestroy(event);
    }
    for (cudaEvent_t& event : insertion_stage_events) {
        if (event != nullptr) cudaEventDestroy(event);
    }
    if (coordinator_stream != nullptr) cudaStreamDestroy(coordinator_stream);
}

EvalTarget CudaComputeBackend::choose_target(const WorkShape& shape) const {
    if (data.execution_policy == "cuda_force") {
        return EvalTarget::CudaBatch;
    }
    if (data.execution_policy == "legacy") {
        if (shape.context == EVALUATION_CONTEXT_LOCAL_SEARCH &&
            shape.item_count < static_cast<size_t>(std::max(1, data.gpu_route_min_batch)) &&
            (data.gpu_route_min_work <= 0 ||
             shape.total_nodes < static_cast<size_t>(data.gpu_route_min_work))) {
            return EvalTarget::CpuIncremental;
        }
        return EvalTarget::CudaBatch;
    }

    // The measured crossover for the paper instances is above their problem
    // size. Keep these workloads on the allocation-free CPU paths. Larger
    // populations and instances still reach CUDA through conservative guards.
    if (data.customer_num <= 100 && data.p_size <= 64 && data.num_islands == 1) {
        return shape.context == EVALUATION_CONTEXT_LOCAL_SEARCH
            ? EvalTarget::CpuIncremental
            : EvalTarget::CpuBatch;
    }

    size_t min_items = adaptive_insertion_min_items;
    size_t min_nodes = adaptive_insertion_min_nodes;
    if (shape.context == EVALUATION_CONTEXT_POPULATION ||
        shape.context == EVALUATION_CONTEXT_ROUTE_BATCH) {
        min_items = adaptive_route_min_items;
        min_nodes = adaptive_route_min_nodes;
    } else if (shape.context == EVALUATION_CONTEXT_LOCAL_SEARCH) {
        min_items = adaptive_route_min_items;
        min_nodes = adaptive_route_min_nodes;
    }
    if (shape.item_count >= min_items && shape.total_nodes >= min_nodes) {
        return EvalTarget::CudaBatch;
    }
    return shape.context == EVALUATION_CONTEXT_LOCAL_SEARCH
        ? EvalTarget::CpuIncremental
        : EvalTarget::CpuBatch;
}

void CudaComputeBackend::ensure_device_cache() {
    ScopedProfileTimer alloc_timer(data.profile, profile_registry().cuda_alloc);
    size_t n = snapshot.delivery.size();
    size_t size_dbl = n * sizeof(double);
    size_t size_matrix = n * n * sizeof(double);

    cuda_check(cudaMalloc((void**)&d_delivery, size_dbl), "cudaMalloc(d_delivery)");
    cuda_check(cudaMalloc((void**)&d_pickup, size_dbl), "cudaMalloc(d_pickup)");
    cuda_check(cudaMalloc((void**)&d_start, size_dbl), "cudaMalloc(d_start)");
    cuda_check(cudaMalloc((void**)&d_end, size_dbl), "cudaMalloc(d_end)");
    cuda_check(cudaMalloc((void**)&d_service, size_dbl), "cudaMalloc(d_service)");
    cuda_check(cudaMalloc((void**)&d_dist, size_matrix), "cudaMalloc(d_dist)");
    cuda_check(cudaMalloc((void**)&d_time, size_matrix), "cudaMalloc(d_time)");

    {
        ScopedProfileTimer copy_timer(data.profile, profile_registry().cuda_h2d, static_cast<long long>((5 * size_dbl + 2 * size_matrix)));
        cuda_check(cudaMemcpy(d_delivery, snapshot.delivery.data(), size_dbl, cudaMemcpyHostToDevice), "cudaMemcpy(d_delivery)");
        cuda_check(cudaMemcpy(d_pickup, snapshot.pickup.data(), size_dbl, cudaMemcpyHostToDevice), "cudaMemcpy(d_pickup)");
        cuda_check(cudaMemcpy(d_start, snapshot.start.data(), size_dbl, cudaMemcpyHostToDevice), "cudaMemcpy(d_start)");
        cuda_check(cudaMemcpy(d_end, snapshot.end.data(), size_dbl, cudaMemcpyHostToDevice), "cudaMemcpy(d_end)");
        cuda_check(cudaMemcpy(d_service, snapshot.service.data(), size_dbl, cudaMemcpyHostToDevice), "cudaMemcpy(d_service)");
        cuda_check(cudaMemcpy(d_dist, snapshot.dist.data(), size_matrix, cudaMemcpyHostToDevice), "cudaMemcpy(d_dist)");
        cuda_check(cudaMemcpy(d_time, snapshot.time.data(), size_matrix, cudaMemcpyHostToDevice), "cudaMemcpy(d_time)");
    }
}

void CudaComputeBackend::free_device_cache() {
    if (d_delivery) cudaFree(d_delivery);
    if (d_pickup) cudaFree(d_pickup);
    if (d_start) cudaFree(d_start);
    if (d_end) cudaFree(d_end);
    if (d_service) cudaFree(d_service);
    if (d_dist) cudaFree(d_dist);
    if (d_time) cudaFree(d_time);
    if (d_route_batch) cudaFree(d_route_batch);
    if (d_route_offsets) cudaFree(d_route_offsets);
    if (d_route_lengths) cudaFree(d_route_lengths);
    if (d_route_feasible) cudaFree(d_route_feasible);
    if (d_route_costs) cudaFree(d_route_costs);
    if (h_route_batch_pinned) cudaFreeHost(h_route_batch_pinned);
    if (h_route_offsets_pinned) cudaFreeHost(h_route_offsets_pinned);
    if (h_route_lengths_pinned) cudaFreeHost(h_route_lengths_pinned);
    if (h_route_feasible_pinned) cudaFreeHost(h_route_feasible_pinned);
    if (h_route_costs_pinned) cudaFreeHost(h_route_costs_pinned);
    if (d_insert_route) cudaFree(d_insert_route);
    if (d_insert_candidates) cudaFree(d_insert_candidates);
    if (d_insert_feasible) cudaFree(d_insert_feasible);
    if (d_insert_costs) cudaFree(d_insert_costs);
    if (d_insert_batch_routes_flat) cudaFree(d_insert_batch_routes_flat);
    if (d_insert_batch_route_offsets) cudaFree(d_insert_batch_route_offsets);
    if (d_insert_batch_route_lengths) cudaFree(d_insert_batch_route_lengths);
    if (d_insert_batch_request_ids) cudaFree(d_insert_batch_request_ids);
    if (d_insert_batch_request_route_ids) cudaFree(d_insert_batch_request_route_ids);
    if (d_insert_batch_request_customer_ids) cudaFree(d_insert_batch_request_customer_ids);
    if (d_insert_batch_request_positions) cudaFree(d_insert_batch_request_positions);
    if (d_insert_batch_out_request_ids) cudaFree(d_insert_batch_out_request_ids);
    if (d_insert_batch_out_feasible) cudaFree(d_insert_batch_out_feasible);
    if (d_insert_batch_out_flags) cudaFree(d_insert_batch_out_flags);
    if (d_insert_batch_out_delta_distance) cudaFree(d_insert_batch_out_delta_distance);
    if (d_insert_batch_out_total_distance) cudaFree(d_insert_batch_out_total_distance);
    if (d_insert_batch_out_fitness) cudaFree(d_insert_batch_out_fitness);
    if (d_insert_batch_prefix_attrs) cudaFree(d_insert_batch_prefix_attrs);
    if (d_insert_batch_suffix_attrs) cudaFree(d_insert_batch_suffix_attrs);
    if (d_insert_batch_base_distances) cudaFree(d_insert_batch_base_distances);
    if (h_insert_batch_input_pinned) cudaFreeHost(h_insert_batch_input_pinned);
    if (h_insert_batch_output_pinned) cudaFreeHost(h_insert_batch_output_pinned);
    if (d_solution_routes_flat) cudaFree(d_solution_routes_flat);
    if (d_solution_route_offsets) cudaFree(d_solution_route_offsets);
    if (d_solution_route_lengths) cudaFree(d_solution_route_lengths);
    if (d_solution_offsets) cudaFree(d_solution_offsets);
    if (d_solution_route_counts) cudaFree(d_solution_route_counts);
    if (d_solution_eval_ids) cudaFree(d_solution_eval_ids);
    if (d_solution_seen) cudaFree(d_solution_seen);
    if (d_solution_feasible) cudaFree(d_solution_feasible);
    if (d_solution_vehicle_counts) cudaFree(d_solution_vehicle_counts);
    if (d_solution_violation_flags) cudaFree(d_solution_violation_flags);
    if (d_solution_fitness) cudaFree(d_solution_fitness);
    if (d_solution_distance) cudaFree(d_solution_distance);

    d_delivery = nullptr;
    d_pickup = nullptr;
    d_start = nullptr;
    d_end = nullptr;
    d_service = nullptr;
    d_dist = nullptr;
    d_time = nullptr;
    d_route_batch = nullptr;
    d_route_offsets = nullptr;
    d_route_lengths = nullptr;
    d_route_feasible = nullptr;
    d_route_costs = nullptr;
    h_route_batch_pinned = nullptr;
    h_route_offsets_pinned = nullptr;
    h_route_lengths_pinned = nullptr;
    h_route_feasible_pinned = nullptr;
    h_route_costs_pinned = nullptr;
    d_insert_route = nullptr;
    d_insert_candidates = nullptr;
    d_insert_feasible = nullptr;
    d_insert_costs = nullptr;
    d_insert_batch_routes_flat = nullptr;
    d_insert_batch_route_offsets = nullptr;
    d_insert_batch_route_lengths = nullptr;
    d_insert_batch_request_ids = nullptr;
    d_insert_batch_request_route_ids = nullptr;
    d_insert_batch_request_customer_ids = nullptr;
    d_insert_batch_request_positions = nullptr;
    d_insert_batch_out_request_ids = nullptr;
    d_insert_batch_out_feasible = nullptr;
    d_insert_batch_out_flags = nullptr;
    d_insert_batch_out_delta_distance = nullptr;
    d_insert_batch_out_total_distance = nullptr;
    d_insert_batch_out_fitness = nullptr;
    d_insert_batch_prefix_attrs = nullptr;
    d_insert_batch_suffix_attrs = nullptr;
    d_insert_batch_base_distances = nullptr;
    h_insert_batch_input_pinned = nullptr;
    h_insert_batch_output_pinned = nullptr;
    d_solution_routes_flat = nullptr;
    d_solution_route_offsets = nullptr;
    d_solution_route_lengths = nullptr;
    d_solution_offsets = nullptr;
    d_solution_route_counts = nullptr;
    d_solution_eval_ids = nullptr;
    d_solution_seen = nullptr;
    d_solution_feasible = nullptr;
    d_solution_vehicle_counts = nullptr;
    d_solution_violation_flags = nullptr;
    d_solution_fitness = nullptr;
    d_solution_distance = nullptr;
    route_batch_node_capacity = 0;
    route_offsets_capacity = 0;
    route_lengths_capacity = 0;
    route_feasible_capacity = 0;
    route_costs_capacity = 0;
    h_route_batch_capacity = 0;
    h_route_offsets_capacity = 0;
    h_route_lengths_capacity = 0;
    h_route_feasible_capacity = 0;
    h_route_costs_capacity = 0;
    insert_route_capacity = 0;
    insert_candidate_capacity = 0;
    insert_feasible_capacity = 0;
    insert_costs_capacity = 0;
    insert_batch_routes_flat_capacity = 0;
    insert_batch_route_offsets_capacity = 0;
    insert_batch_route_lengths_capacity = 0;
    insert_batch_request_ids_capacity = 0;
    insert_batch_request_route_ids_capacity = 0;
    insert_batch_request_customer_ids_capacity = 0;
    insert_batch_request_positions_capacity = 0;
    insert_batch_out_request_ids_capacity = 0;
    insert_batch_out_feasible_capacity = 0;
    insert_batch_out_flags_capacity = 0;
    insert_batch_out_delta_distance_capacity = 0;
    insert_batch_out_total_distance_capacity = 0;
    insert_batch_out_fitness_capacity = 0;
    insert_batch_prefix_attrs_capacity = 0;
    insert_batch_suffix_attrs_capacity = 0;
    insert_batch_base_distances_capacity = 0;
    h_insert_batch_input_capacity = 0;
    h_insert_batch_output_capacity = 0;
    solution_routes_flat_capacity = 0;
    solution_route_offsets_capacity = 0;
    solution_route_lengths_capacity = 0;
    solution_offsets_capacity = 0;
    solution_route_counts_capacity = 0;
    solution_eval_ids_capacity = 0;
    solution_feasible_capacity = 0;
    solution_vehicle_counts_capacity = 0;
    solution_violation_flags_capacity = 0;
    solution_fitness_capacity = 0;
    solution_distance_capacity = 0;
    solution_seen_capacity = 0;
}

void CudaComputeBackend::ensure_route_batch_capacity(size_t num_routes, size_t flat_nodes) {
    ScopedProfileTimer timer(data.profile, profile_registry().cuda_alloc);
    ensure_device_buffer(d_route_batch, route_batch_node_capacity, flat_nodes, "cudaMalloc(d_route_batch)");
    ensure_device_buffer(d_route_offsets, route_offsets_capacity, num_routes, "cudaMalloc(d_route_offsets)");
    ensure_device_buffer(d_route_lengths, route_lengths_capacity, num_routes, "cudaMalloc(d_route_lengths)");
    ensure_device_buffer(d_route_feasible, route_feasible_capacity, num_routes, "cudaMalloc(d_route_feasible)");
    ensure_device_buffer(d_route_costs, route_costs_capacity, num_routes, "cudaMalloc(d_route_costs)");
    ensure_pinned_buffer(h_route_batch_pinned, h_route_batch_capacity, flat_nodes, "cudaHostAlloc(h_route_batch)");
    ensure_pinned_buffer(h_route_offsets_pinned, h_route_offsets_capacity, num_routes, "cudaHostAlloc(h_route_offsets)");
    ensure_pinned_buffer(h_route_lengths_pinned, h_route_lengths_capacity, num_routes, "cudaHostAlloc(h_route_lengths)");
    ensure_pinned_buffer(h_route_feasible_pinned, h_route_feasible_capacity, num_routes, "cudaHostAlloc(h_route_feasible)");
    ensure_pinned_buffer(h_route_costs_pinned, h_route_costs_capacity, num_routes, "cudaHostAlloc(h_route_costs)");
}

void CudaComputeBackend::ensure_insert_capacity(size_t route_len, size_t num_candidates) {
    ScopedProfileTimer timer(data.profile, profile_registry().cuda_insertion_alloc);
    size_t total_evals = num_candidates * (route_len > 0 ? route_len - 1 : 0);
    ensure_device_buffer(d_insert_route, insert_route_capacity, route_len, "cudaMalloc(d_insert_route)");
    ensure_device_buffer(d_insert_candidates, insert_candidate_capacity, num_candidates, "cudaMalloc(d_insert_candidates)");
    ensure_device_buffer(d_insert_feasible, insert_feasible_capacity, total_evals, "cudaMalloc(d_insert_feasible)");
    ensure_device_buffer(d_insert_costs, insert_costs_capacity, total_evals, "cudaMalloc(d_insert_costs)");
}

void CudaComputeBackend::ensure_insert_batch_capacity(size_t flat_nodes, size_t route_count, size_t request_count) {
    ScopedProfileTimer timer(data.profile, profile_registry().cuda_insertion_alloc);
    ensure_device_buffer(d_insert_batch_routes_flat, insert_batch_routes_flat_capacity, flat_nodes, "cudaMalloc(d_insert_batch_routes_flat)");
    ensure_device_buffer(d_insert_batch_route_offsets, insert_batch_route_offsets_capacity, route_count, "cudaMalloc(d_insert_batch_route_offsets)");
    ensure_device_buffer(d_insert_batch_route_lengths, insert_batch_route_lengths_capacity, route_count, "cudaMalloc(d_insert_batch_route_lengths)");
    ensure_device_buffer(d_insert_batch_request_ids, insert_batch_request_ids_capacity, request_count, "cudaMalloc(d_insert_batch_request_ids)");
    ensure_device_buffer(d_insert_batch_request_route_ids, insert_batch_request_route_ids_capacity, request_count, "cudaMalloc(d_insert_batch_request_route_ids)");
    ensure_device_buffer(d_insert_batch_request_customer_ids, insert_batch_request_customer_ids_capacity, request_count, "cudaMalloc(d_insert_batch_request_customer_ids)");
    ensure_device_buffer(d_insert_batch_request_positions, insert_batch_request_positions_capacity, request_count, "cudaMalloc(d_insert_batch_request_positions)");
    ensure_device_buffer(d_insert_batch_out_request_ids, insert_batch_out_request_ids_capacity, request_count, "cudaMalloc(d_insert_batch_out_request_ids)");
    ensure_device_buffer(d_insert_batch_out_feasible, insert_batch_out_feasible_capacity, request_count, "cudaMalloc(d_insert_batch_out_feasible)");
    ensure_device_buffer(d_insert_batch_out_flags, insert_batch_out_flags_capacity, request_count, "cudaMalloc(d_insert_batch_out_flags)");
    ensure_device_buffer(d_insert_batch_out_delta_distance, insert_batch_out_delta_distance_capacity, request_count, "cudaMalloc(d_insert_batch_out_delta_distance)");
    ensure_device_buffer(d_insert_batch_out_total_distance, insert_batch_out_total_distance_capacity, request_count, "cudaMalloc(d_insert_batch_out_total_distance)");
    ensure_device_buffer(d_insert_batch_out_fitness, insert_batch_out_fitness_capacity, request_count, "cudaMalloc(d_insert_batch_out_fitness)");
}

void CudaComputeBackend::ensure_insert_attr_capacity(size_t attr_count, size_t route_count) {
    ScopedProfileTimer timer(data.profile, profile_registry().cuda_insertion_alloc);
    ensure_device_buffer(d_insert_batch_prefix_attrs, insert_batch_prefix_attrs_capacity, attr_count, "cudaMalloc(d_insert_batch_prefix_attrs)");
    ensure_device_buffer(d_insert_batch_suffix_attrs, insert_batch_suffix_attrs_capacity, attr_count, "cudaMalloc(d_insert_batch_suffix_attrs)");
    ensure_device_buffer(d_insert_batch_base_distances, insert_batch_base_distances_capacity, route_count, "cudaMalloc(d_insert_batch_base_distances)");
}

void CudaComputeBackend::ensure_solution_capacity(const EncodedPopulation& encoded) {
    ScopedProfileTimer timer(data.profile, profile_registry().cuda_alloc);
    size_t solution_count = static_cast<size_t>(encoded.solution_count);
    size_t required_routes_per_solution = 0;
    size_t required_nodes_per_solution = 0;
    for (int solution_id = 0; solution_id < encoded.solution_count; ++solution_id) {
        const int route_start = encoded.solution_offsets[solution_id];
        const int route_count = encoded.solution_route_counts[solution_id];
        required_routes_per_solution = std::max(
            required_routes_per_solution,
            static_cast<size_t>(std::max(0, route_count))
        );
        size_t node_count = 0;
        for (int route = 0; route < route_count; ++route) {
            node_count += static_cast<size_t>(std::max(0, encoded.route_lengths[route_start + route]));
        }
        required_nodes_per_solution = std::max(required_nodes_per_solution, node_count);
    }

    const size_t theoretical_routes = static_cast<size_t>(std::max(1, snapshot.customer_num));
    const size_t theoretical_nodes = static_cast<size_t>(std::max(2, 3 * snapshot.customer_num));
    solution_slot_route_capacity = std::max(
        solution_slot_route_capacity,
        std::max(theoretical_routes, required_routes_per_solution)
    );
    solution_slot_node_capacity = std::max(
        solution_slot_node_capacity,
        std::max(theoretical_nodes, required_nodes_per_solution)
    );

    const size_t flat_count = solution_count * solution_slot_node_capacity;
    const size_t route_count = solution_count * solution_slot_route_capacity;
    size_t seen_count = solution_count * static_cast<size_t>(encoded.customer_count + 1);

    ensure_device_buffer(d_solution_routes_flat, solution_routes_flat_capacity, flat_count, "cudaMalloc(d_solution_routes_flat)");
    ensure_device_buffer(d_solution_route_offsets, solution_route_offsets_capacity, route_count, "cudaMalloc(d_solution_route_offsets)");
    ensure_device_buffer(d_solution_route_lengths, solution_route_lengths_capacity, route_count, "cudaMalloc(d_solution_route_lengths)");
    ensure_device_buffer(d_solution_offsets, solution_offsets_capacity, solution_count, "cudaMalloc(d_solution_offsets)");
    ensure_device_buffer(d_solution_route_counts, solution_route_counts_capacity, solution_count, "cudaMalloc(d_solution_route_counts)");
    ensure_device_buffer(d_solution_eval_ids, solution_eval_ids_capacity, solution_count, "cudaMalloc(d_solution_eval_ids)");
    ensure_device_buffer(d_solution_seen, solution_seen_capacity, seen_count, "cudaMalloc(d_solution_seen)");
    ensure_device_buffer(d_solution_feasible, solution_feasible_capacity, solution_count, "cudaMalloc(d_solution_feasible)");
    ensure_device_buffer(d_solution_vehicle_counts, solution_vehicle_counts_capacity, solution_count, "cudaMalloc(d_solution_vehicle_counts)");
    ensure_device_buffer(d_solution_violation_flags, solution_violation_flags_capacity, solution_count, "cudaMalloc(d_solution_violation_flags)");
    ensure_device_buffer(d_solution_fitness, solution_fitness_capacity, solution_count, "cudaMalloc(d_solution_fitness)");
    ensure_device_buffer(d_solution_distance, solution_distance_capacity, solution_count, "cudaMalloc(d_solution_distance)");
}

RouteEval CudaComputeBackend::evaluate_route(const std::vector<int>& route) {
    CpuComputeBackend cpu_fallback(data);
    return cpu_fallback.evaluate_route(route);
}

std::vector<RouteEval> CudaComputeBackend::evaluate_routes(const std::vector<std::vector<int>>& routes) {
    if (routes.empty()) return {};

    int num_routes = static_cast<int>(routes.size());
    int max_len = 0;
    for (const auto& r : routes) {
        max_len = std::max(max_len, static_cast<int>(r.size()));
    }
    long long route_work = static_cast<long long>(num_routes) * static_cast<long long>(max_len);

    if (data.execution_policy != "legacy") {
        WorkShape shape;
        shape.context = EVALUATION_CONTEXT_ROUTE_BATCH;
        shape.item_count = routes.size();
        shape.max_length = static_cast<size_t>(max_len);
        for (const auto& route : routes) shape.total_nodes += route.size();
        shape.transfer_bytes = shape.total_nodes * sizeof(int);
        const EvalTarget target = choose_target(shape);
        if (target != EvalTarget::CudaBatch) {
            profile_registry().dispatch_cpu_batch.add(0, static_cast<long long>(routes.size()));
            CpuComputeBackend cpu(data);
            return cpu.evaluate_routes(routes);
        }
        profile_registry().dispatch_cuda_batch.add(0, static_cast<long long>(routes.size()));
        return evaluate_route_batch_direct(routes);
    }

    const bool route_count_threshold_enabled = data.gpu_route_min_batch > 0;
    const bool route_work_threshold_enabled = data.gpu_route_min_work > 0;
    const bool below_static_threshold =
        (route_count_threshold_enabled || route_work_threshold_enabled)
        && (!route_count_threshold_enabled || num_routes < data.gpu_route_min_batch)
        && (!route_work_threshold_enabled || route_work < data.gpu_route_min_work);
    const bool broker_eligible = data.architecture == "hybrid_v2"
        && data.gpu_request_broker
        && !data.gpu_serialize_workers
        && data.parallel_workers != 1;

    if (!below_static_threshold && !broker_eligible && route_scheduler_prefers_cpu(num_routes, route_work)) {
        const auto start = std::chrono::high_resolution_clock::now();
        ScopedProfileTimer fallback_timer(data.profile, profile_registry().route_scheduler_cpu, route_work);
        CpuComputeBackend cpu_fallback(data);
        auto results = cpu_fallback.evaluate_routes(routes);
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start
        ).count();
        observe_route_cpu(static_cast<double>(elapsed), routes.size(), route_work);
        return results;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    std::vector<RouteEval> results;
    if (data.architecture == "hybrid_v2" && data.gpu_request_broker &&
        !data.gpu_serialize_workers && data.parallel_workers != 1) {
        results = evaluate_route_batch_brokered(routes);
    } else {
        results = evaluate_route_batch_direct(routes);
    }

    if (!broker_eligible) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start
        ).count();
        if (below_static_threshold) {
            observe_route_cpu(static_cast<double>(elapsed), routes.size(), route_work);
        } else {
            observe_route_gpu(static_cast<double>(elapsed), routes.size(), route_work);
        }
    }
    return results;
}

std::vector<RouteEval> CudaComputeBackend::evaluate_route_batch_brokered(const std::vector<std::vector<int>>& routes) {
    auto pending = std::make_shared<PendingRouteCall>();
    pending->routes = routes;

    bool leader = false;
    {
        std::unique_lock<std::mutex> lock(route_broker_mutex);
        route_broker_queue.push_back(pending);
        if (!route_broker_active) {
            route_broker_active = true;
            leader = true;
        } else {
            route_broker_cv.notify_all();
            route_broker_cv.wait(lock, [&]() { return pending->done; });
        }
    }

    if (!leader) {
        if (pending->error) std::rethrow_exception(pending->error);
        return pending->results;
    }

    while (true) {
        std::vector<std::shared_ptr<PendingRouteCall>> batch;
        {
            std::unique_lock<std::mutex> lock(route_broker_mutex);
            const int target_routes = std::max(1, data.gpu_route_min_batch);
            route_broker_cv.wait_for(lock, std::chrono::microseconds(75), [&]() {
                long long queued_routes = 0;
                for (const auto& call : route_broker_queue) {
                    queued_routes += static_cast<long long>(call->routes.size());
                }
                return queued_routes >= target_routes;
            });
            batch.swap(route_broker_queue);
        }

        std::vector<std::vector<int>> merged_routes;
        std::vector<size_t> route_offsets;
        route_offsets.reserve(batch.size() + 1);
        route_offsets.push_back(0);

        for (const auto& call : batch) {
            merged_routes.insert(merged_routes.end(), call->routes.begin(), call->routes.end());
            route_offsets.push_back(merged_routes.size());
        }

        std::vector<RouteEval> merged_results;
        std::exception_ptr batch_error;
        try {
            size_t merged_max_route_len = 0;
            for (const auto& route : merged_routes) {
                merged_max_route_len = std::max(merged_max_route_len, route.size());
            }
            const size_t merged_route_work = merged_routes.size() * merged_max_route_len;
            const auto scoring_start = std::chrono::high_resolution_clock::now();
            merged_results = evaluate_route_batch_direct(merged_routes);
            const auto scoring_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - scoring_start
            ).count();
            const bool route_count_threshold_enabled = data.gpu_route_min_batch > 0;
            const bool route_work_threshold_enabled = data.gpu_route_min_work > 0;
            const bool below_static_threshold =
                (route_count_threshold_enabled || route_work_threshold_enabled)
                && (!route_count_threshold_enabled || merged_routes.size() < static_cast<size_t>(data.gpu_route_min_batch))
                && (!route_work_threshold_enabled || merged_route_work < static_cast<size_t>(data.gpu_route_min_work));
            if (below_static_threshold) {
                observe_route_cpu(static_cast<double>(scoring_elapsed), merged_routes.size(), merged_route_work);
            } else {
                observe_route_gpu(static_cast<double>(scoring_elapsed), merged_routes.size(), merged_route_work);
            }
            if (merged_results.size() != merged_routes.size()) {
                throw std::runtime_error("Route broker received an incomplete result batch");
            }
        } catch (...) {
            batch_error = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(route_broker_mutex);
            for (size_t i = 0; i < batch.size(); ++i) {
                batch[i]->error = batch_error;
                if (!batch_error) {
                    batch[i]->results.assign(
                        merged_results.begin() + route_offsets[i],
                        merged_results.begin() + route_offsets[i + 1]
                    );
                }
                batch[i]->done = true;
            }
            route_broker_cv.notify_all();
            if (route_broker_queue.empty()) {
                route_broker_active = false;
                break;
            }
        }
    }

    if (pending->error) std::rethrow_exception(pending->error);
    return pending->results;
}

std::vector<RouteEval> CudaComputeBackend::evaluate_route_batch_direct(const std::vector<std::vector<int>>& routes) {
    ScopedProfileTimer timer(data.profile, profile_registry().feasibility_batch, static_cast<long long>(routes.size()));
    if (routes.empty()) return {};

    int num_routes = static_cast<int>(routes.size());
    int max_len = 0;
    for (const auto& r : routes) {
        max_len = std::max(max_len, static_cast<int>(r.size()));
    }
    long long route_work = static_cast<long long>(num_routes) * static_cast<long long>(max_len);

    const bool route_count_threshold_enabled = data.gpu_route_min_batch > 0;
    const bool route_work_threshold_enabled = data.gpu_route_min_work > 0;
    const bool below_static_threshold =
        (route_count_threshold_enabled || route_work_threshold_enabled)
        && (!route_count_threshold_enabled || num_routes < data.gpu_route_min_batch)
        && (!route_work_threshold_enabled || route_work < data.gpu_route_min_work);
    if (below_static_threshold && data.execution_policy != "cuda_force") {
        ScopedProfileTimer fallback_timer(data.profile, profile_registry().cuda_route_cpu_fallback, route_work);
        CpuComputeBackend cpu_fallback(data);
        return cpu_fallback.evaluate_routes(routes);
    }

    std::lock_guard<std::mutex> lock(gpu_mutex);

    if (max_len == 0) {
        std::vector<RouteEval> empty_results(routes.size(), RouteEval{false, 0.0});
        return empty_results;
    }

    size_t flat_nodes = 0;
    for (const auto& route : routes) flat_nodes += route.size();
    ensure_route_batch_capacity(routes.size(), flat_nodes);

    size_t node_offset = 0;
    for (int i = 0; i < num_routes; ++i) {
        h_route_offsets_pinned[i] = static_cast<int>(node_offset);
        h_route_lengths_pinned[i] = static_cast<int>(routes[i].size());
        std::copy(routes[i].begin(), routes[i].end(), h_route_batch_pinned + node_offset);
        node_offset += routes[i].size();
    }

    const long long h2d_copy_bytes = static_cast<long long>(
        flat_nodes * sizeof(int) + 2 * routes.size() * sizeof(int)
    );
    const long long d2h_copy_bytes = static_cast<long long>(
        routes.size() * (sizeof(int) + sizeof(double))
    );
    cuda_check(cudaEventRecord(route_stage_events[0], coordinator_stream), "cudaEventRecord(route_h2d_start)");
    cuda_check(cudaMemcpyAsync(d_route_batch, h_route_batch_pinned, flat_nodes * sizeof(int),
                               cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_route_batch)");
    cuda_check(cudaMemcpyAsync(d_route_offsets, h_route_offsets_pinned, routes.size() * sizeof(int),
                               cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_route_offsets)");
    cuda_check(cudaMemcpyAsync(d_route_lengths, h_route_lengths_pinned, routes.size() * sizeof(int),
                               cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_route_lengths)");
    cuda_check(cudaEventRecord(route_stage_events[1], coordinator_stream), "cudaEventRecord(route_h2d_end)");

    int threads_per_block = 128;
    int blocks = (num_routes + threads_per_block - 1) / threads_per_block;
    int stride = snapshot.customer_num + 1;

    evaluate_route_batch_kernel<<<blocks, threads_per_block, 0, coordinator_stream>>>(
            d_route_batch,
            d_route_offsets,
            d_route_lengths,
            num_routes,
            stride,
            snapshot.depot,
            snapshot.start_time,
            snapshot.capacity,
            snapshot.dispatch_cost,
            snapshot.unit_cost,
            d_delivery,
            d_pickup,
            d_start,
            d_end,
            d_service,
            d_dist,
            d_time,
            d_route_feasible,
            d_route_costs
    );
    cuda_check(cudaGetLastError(), "evaluate_route_batch_kernel launch");
    cuda_check(cudaEventRecord(route_stage_events[2], coordinator_stream), "cudaEventRecord(route_kernel_end)");

    cuda_check(cudaMemcpyAsync(h_route_feasible_pinned, d_route_feasible, routes.size() * sizeof(int),
                               cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(h_route_feasible)");
    cuda_check(cudaMemcpyAsync(h_route_costs_pinned, d_route_costs, routes.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(h_route_costs)");
    cuda_check(cudaEventRecord(route_stage_events[3], coordinator_stream), "cudaEventRecord(route_d2h_end)");
    cuda_check(cudaEventSynchronize(route_stage_events[3]), "cudaEventSynchronize(route_d2h_end)");

    if (data.profile) {
        float h2d_ms = 0.0f;
        float kernel_ms = 0.0f;
        float d2h_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&h2d_ms, route_stage_events[0], route_stage_events[1]), "cudaEventElapsedTime(route_h2d)");
        cuda_check(cudaEventElapsedTime(&kernel_ms, route_stage_events[1], route_stage_events[2]), "cudaEventElapsedTime(route_kernel)");
        cuda_check(cudaEventElapsedTime(&d2h_ms, route_stage_events[2], route_stage_events[3]), "cudaEventElapsedTime(route_d2h)");
        profile_registry().cuda_h2d.add(static_cast<long long>(h2d_ms * 1e6), h2d_copy_bytes);
        profile_registry().cuda_kernel.add(static_cast<long long>(kernel_ms * 1e6), num_routes);
        profile_registry().cuda_d2h.add(static_cast<long long>(d2h_ms * 1e6), d2h_copy_bytes);
    }

    std::vector<RouteEval> results(num_routes);
    for (int i = 0; i < num_routes; ++i) {
        results[i] = RouteEval{h_route_feasible_pinned[i] == 1, h_route_costs_pinned[i]};
    }
    return results;
}
void CudaComputeBackend::evaluate_insertions(
    const std::vector<int>& route,
    const std::vector<int>& candidates,
    std::vector<int>& out_feasible,
    std::vector<double>& out_costs
) {
    long long eval_units = route.size() > 1
        ? static_cast<long long>(candidates.size()) * static_cast<long long>(route.size() - 1)
        : 0;
    ScopedProfileTimer timer(data.profile, profile_registry().insertion, eval_units);
    std::lock_guard<std::mutex> lock(gpu_mutex);

    int route_len = static_cast<int>(route.size());
    if (route_len < 2 || candidates.empty()) {
        out_feasible.clear();
        out_costs.clear();
        return;
    }

    int num_candidates = static_cast<int>(candidates.size());
    int num_positions = route_len - 1;
    int total_evals = num_candidates * num_positions;
    out_feasible.assign(total_evals, 0);
    out_costs.assign(total_evals, 0.0);

    ensure_insert_capacity(route.size(), candidates.size());

    {
        long long copy_bytes = static_cast<long long>(route.size() * sizeof(int) + candidates.size() * sizeof(int));
        ScopedProfileTimer copy_timer(data.profile, profile_registry().cuda_insertion_h2d, copy_bytes);
        cuda_check(cudaMemcpy(d_insert_route, route.data(), route.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(d_insert_route)");
        cuda_check(cudaMemcpy(d_insert_candidates, candidates.data(), candidates.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(d_insert_candidates)");
    }

    int threads_per_block = 256;
    int blocks = (total_evals + threads_per_block - 1) / threads_per_block;
    int stride = snapshot.customer_num + 1;

    {
        ScopedProfileTimer kernel_timer(data.profile, profile_registry().cuda_insertion_kernel, total_evals);
        evaluate_insertions_cuda_kernel<<<blocks, threads_per_block>>>(
            d_insert_route,
            route_len,
            d_insert_candidates,
            num_candidates,
            snapshot.depot,
            snapshot.start_time,
            snapshot.capacity,
            snapshot.dispatch_cost,
            snapshot.unit_cost,
            d_delivery,
            d_pickup,
            d_start,
            d_end,
            d_service,
            d_dist,
            d_time,
            stride,
            d_insert_feasible,
            d_insert_costs
        );
        cuda_check(cudaGetLastError(), "evaluate_insertions_cuda_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "evaluate_insertions_cuda_kernel synchronize");
    }

    {
        long long copy_bytes = static_cast<long long>(total_evals * (sizeof(int) + sizeof(double)));
        ScopedProfileTimer copy_timer(data.profile, profile_registry().cuda_insertion_d2h, copy_bytes);
        cuda_check(cudaMemcpy(out_feasible.data(), d_insert_feasible, total_evals * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(out_feasible)");
        cuda_check(cudaMemcpy(out_costs.data(), d_insert_costs, total_evals * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(out_costs)");
    }
}

bool CudaComputeBackend::insertion_scheduler_prefers_cpu(
    const std::vector<std::vector<int>>& routes,
    const std::vector<InsertionRequest>& requests
) const {
    if (data.architecture != "hybrid_v2" || requests.empty()) {
        return false;
    }

    size_t route_nodes = 0;
    for (const auto& route : routes) {
        route_nodes += route.size();
    }

    std::lock_guard<std::mutex> lock(insertion_timing_mutex);
    if (insertion_timing.cpu_samples < 3 || insertion_timing.gpu_samples < 3) {
        return false;
    }

    const double request_count = static_cast<double>(requests.size());
    const double node_count = static_cast<double>(route_nodes);
    const double cpu_prediction = insertion_timing.cpu_base_us
        + insertion_timing.cpu_request_us * request_count
        + insertion_timing.cpu_node_us * node_count;
    const double gpu_prediction = insertion_timing.gpu_base_us
        + insertion_timing.gpu_request_us * request_count
        + insertion_timing.gpu_node_us * node_count;

    // Keep a hysteresis margin so noisy measurements do not oscillate between
    // CPU and GPU for similarly shaped batches.
    return cpu_prediction * 1.15 < gpu_prediction;
}

void CudaComputeBackend::observe_insertion_cpu(
    double elapsed_us,
    size_t request_count,
    size_t route_nodes
) {
    if (request_count == 0) return;
    const double request_value = static_cast<double>(request_count);
    const double node_value = static_cast<double>(route_nodes);
    const double per_request = elapsed_us / request_value;
    const double per_node = elapsed_us / std::max(1.0, node_value);
    std::lock_guard<std::mutex> lock(insertion_timing_mutex);
    const double alpha = insertion_timing.cpu_samples < 8 ? 0.5 : 0.2;
    insertion_timing.cpu_request_us = (1.0 - alpha) * insertion_timing.cpu_request_us + alpha * per_request;
    insertion_timing.cpu_node_us = (1.0 - alpha) * insertion_timing.cpu_node_us + alpha * per_node;
    insertion_timing.cpu_base_us = std::max(1.0, elapsed_us
        - insertion_timing.cpu_request_us * request_value
        - insertion_timing.cpu_node_us * node_value);
    insertion_timing.cpu_samples++;
}

void CudaComputeBackend::observe_insertion_gpu(
    double elapsed_us,
    size_t request_count,
    size_t route_nodes
) {
    if (request_count == 0) return;
    const double request_value = static_cast<double>(request_count);
    const double node_value = static_cast<double>(route_nodes);
    const double per_request = elapsed_us / request_value;
    const double per_node = elapsed_us / std::max(1.0, node_value);
    std::lock_guard<std::mutex> lock(insertion_timing_mutex);
    const double alpha = insertion_timing.gpu_samples < 8 ? 0.5 : 0.2;
    insertion_timing.gpu_request_us = (1.0 - alpha) * insertion_timing.gpu_request_us + alpha * per_request;
    insertion_timing.gpu_node_us = (1.0 - alpha) * insertion_timing.gpu_node_us + alpha * per_node;
    insertion_timing.gpu_base_us = std::max(1.0, elapsed_us
        - insertion_timing.gpu_request_us * request_value
        - insertion_timing.gpu_node_us * node_value);
    insertion_timing.gpu_samples++;
}
bool CudaComputeBackend::route_scheduler_prefers_cpu(
    size_t num_routes,
    size_t route_work
) const {
    if (data.architecture != "hybrid_v2" || num_routes == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(route_timing_mutex);
    if (route_timing.cpu_samples < 3 || route_timing.gpu_samples < 3) {
        return false;
    }

    const double route_count = static_cast<double>(num_routes);
    const double node_count = static_cast<double>(route_work);
    const double cpu_prediction = route_timing.cpu_base_us
        + route_timing.cpu_route_us * route_count
        + route_timing.cpu_node_us * node_count;
    const double gpu_prediction = route_timing.gpu_base_us
        + route_timing.gpu_route_us * route_count
        + route_timing.gpu_node_us * node_count;

    return cpu_prediction * 1.15 < gpu_prediction;
}

void CudaComputeBackend::observe_route_cpu(
    double elapsed_us,
    size_t num_routes,
    size_t route_work
) {
    if (num_routes == 0) return;
    const double route_value = static_cast<double>(num_routes);
    const double node_value = static_cast<double>(route_work);
    const double per_route = elapsed_us / route_value;
    const double per_node = elapsed_us / std::max(1.0, node_value);
    std::lock_guard<std::mutex> lock(route_timing_mutex);
    const double alpha = route_timing.cpu_samples < 8 ? 0.5 : 0.2;
    route_timing.cpu_route_us = (1.0 - alpha) * route_timing.cpu_route_us + alpha * per_route;
    route_timing.cpu_node_us = (1.0 - alpha) * route_timing.cpu_node_us + alpha * per_node;
    route_timing.cpu_base_us = std::max(1.0, elapsed_us
        - route_timing.cpu_route_us * route_value
        - route_timing.cpu_node_us * node_value);
    route_timing.cpu_samples++;
}

void CudaComputeBackend::observe_route_gpu(
    double elapsed_us,
    size_t num_routes,
    size_t route_work
) {
    if (num_routes == 0) return;
    const double route_value = static_cast<double>(num_routes);
    const double node_value = static_cast<double>(route_work);
    const double per_route = elapsed_us / route_value;
    const double per_node = elapsed_us / std::max(1.0, node_value);
    std::lock_guard<std::mutex> lock(route_timing_mutex);
    const double alpha = route_timing.gpu_samples < 8 ? 0.5 : 0.2;
    route_timing.gpu_route_us = (1.0 - alpha) * route_timing.gpu_route_us + alpha * per_route;
    route_timing.gpu_node_us = (1.0 - alpha) * route_timing.gpu_node_us + alpha * per_node;
    route_timing.gpu_base_us = std::max(1.0, elapsed_us
        - route_timing.gpu_route_us * route_value
        - route_timing.gpu_node_us * node_value);
    route_timing.gpu_samples++;
}


std::vector<InsertionScore> CudaComputeBackend::evaluate_insertion_batch(
    const std::vector<std::vector<int>>& routes,
    const std::vector<InsertionRequest>& requests
) {
    if (requests.empty()) return {};

    size_t route_nodes = 0;
    for (const auto& route : routes) {
        route_nodes += route.size();
    }

    if (data.execution_policy != "legacy") {
        WorkShape shape;
        shape.context = requests.front().source_context;
        shape.item_count = requests.size();
        shape.total_nodes = route_nodes;
        for (const auto& route : routes) shape.max_length = std::max(shape.max_length, route.size());
        shape.transfer_bytes = route_nodes * sizeof(int) + requests.size() * 4 * sizeof(int);
        const EvalTarget target = choose_target(shape);
        if (target != EvalTarget::CudaBatch) {
            profile_registry().dispatch_cpu_batch.add(0, static_cast<long long>(requests.size()));
            CpuComputeBackend cpu(data);
            return cpu.evaluate_insertion_batch(routes, requests);
        }
        profile_registry().dispatch_cuda_batch.add(0, static_cast<long long>(requests.size()));
        return evaluate_insertion_batch_direct(routes, requests);
    }

    const bool below_static_threshold = data.gpu_insertion_min_batch > 0
        && requests.size() < static_cast<size_t>(data.gpu_insertion_min_batch);
    const bool broker_eligible = data.architecture == "hybrid_v2"
        && data.gpu_request_broker
        && !data.gpu_serialize_workers
        && data.parallel_workers != 1
        && requests.front().source_context != INSERTION_CONTEXT_INITIALIZATION;

    if (!below_static_threshold && !broker_eligible && insertion_scheduler_prefers_cpu(routes, requests)) {
        const auto start = std::chrono::high_resolution_clock::now();
        ScopedProfileTimer fallback_timer(data.profile, profile_registry().insertion_scheduler_cpu, requests.size());
        CpuComputeBackend cpu_fallback(data);
        std::vector<InsertionScore> results = cpu_fallback.evaluate_insertion_batch(routes, requests);
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start
        ).count();
        observe_insertion_cpu(static_cast<double>(elapsed), requests.size(), route_nodes);
        return results;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    const bool aggregation_context = requests.front().source_context != INSERTION_CONTEXT_INITIALIZATION;
    std::vector<InsertionScore> results;
    if (data.architecture == "hybrid_v2" && data.gpu_request_broker && !data.gpu_serialize_workers &&
        data.parallel_workers != 1 && aggregation_context) {
        results = evaluate_insertion_batch_brokered(routes, requests);
    } else {
        results = evaluate_insertion_batch_direct(routes, requests);
    }

    // Brokered calls include queue wait time and are intentionally excluded
    // from the model; only direct GPU observations describe kernel/transfer
    // cost and remain stable across worker counts.
    if (!broker_eligible) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start
        ).count();
        if (below_static_threshold) {
            observe_insertion_cpu(static_cast<double>(elapsed), requests.size(), route_nodes);
        } else {
            observe_insertion_gpu(static_cast<double>(elapsed), requests.size(), route_nodes);
        }
    }
    return results;
}

std::vector<InsertionScore> CudaComputeBackend::evaluate_insertion_batch_brokered(
    const std::vector<std::vector<int>>& routes,
    const std::vector<InsertionRequest>& requests
) {
    profile_registry().insertion_broker_submission.add(0, static_cast<long long>(requests.size()));

    auto pending = std::make_shared<PendingInsertionCall>();
    pending->routes = routes;
    pending->requests = requests;

    bool leader = false;
    {
        std::unique_lock<std::mutex> lock(insertion_broker_mutex);
        insertion_broker_queue.push_back(pending);
        if (!insertion_broker_active) {
            insertion_broker_active = true;
            leader = true;
        } else {
            insertion_broker_cv.notify_all();
            insertion_broker_cv.wait(lock, [&]() { return pending->done; });
        }
    }

    if (!leader) {
        if (pending->error) std::rethrow_exception(pending->error);
        return pending->results;
    }

    while (true) {
        std::vector<std::shared_ptr<PendingInsertionCall>> batch;
        {
            std::unique_lock<std::mutex> lock(insertion_broker_mutex);
            const int target_requests = std::max(1, data.gpu_insertion_min_batch);
            insertion_broker_cv.wait_for(lock, std::chrono::microseconds(75), [&]() {
                long long queued_requests = 0;
                for (const auto& call : insertion_broker_queue) {
                    queued_requests += static_cast<long long>(call->requests.size());
                }
                return queued_requests >= target_requests;
            });
            batch.swap(insertion_broker_queue);
        }

        std::vector<std::vector<int>> merged_routes;
        std::vector<InsertionRequest> merged_requests;
        std::vector<size_t> request_offsets;
        request_offsets.reserve(batch.size() + 1);
        request_offsets.push_back(0);

        for (const auto& call : batch) {
            const int route_base = static_cast<int>(merged_routes.size());
            merged_routes.insert(merged_routes.end(), call->routes.begin(), call->routes.end());
            for (const InsertionRequest& original : call->requests) {
                InsertionRequest adjusted = original;
                if (adjusted.route_id >= 0) adjusted.route_id += route_base;
                merged_requests.push_back(adjusted);
            }
            request_offsets.push_back(merged_requests.size());
        }

        std::vector<InsertionScore> merged_results;
        std::exception_ptr batch_error;
        try {
            size_t merged_route_nodes = 0;
            for (const auto& route : merged_routes) {
                merged_route_nodes += route.size();
            }
            const auto scoring_start = std::chrono::high_resolution_clock::now();
            merged_results = evaluate_insertion_batch_direct(merged_routes, merged_requests);
            const auto scoring_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - scoring_start
            ).count();
            if (data.gpu_insertion_min_batch > 0 &&
                merged_requests.size() < static_cast<size_t>(data.gpu_insertion_min_batch)) {
                observe_insertion_cpu(static_cast<double>(scoring_elapsed), merged_requests.size(), merged_route_nodes);
            } else {
                observe_insertion_gpu(static_cast<double>(scoring_elapsed), merged_requests.size(), merged_route_nodes);
            }
            if (merged_results.size() != merged_requests.size()) {
                throw std::runtime_error("Insertion broker received an incomplete result batch");
            }
        } catch (...) {
            batch_error = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(insertion_broker_mutex);
            for (size_t i = 0; i < batch.size(); ++i) {
                batch[i]->error = batch_error;
                if (!batch_error) {
                    batch[i]->results.assign(
                        merged_results.begin() + request_offsets[i],
                        merged_results.begin() + request_offsets[i + 1]
                    );
                }
                batch[i]->done = true;
            }
            insertion_broker_cv.notify_all();
            if (insertion_broker_queue.empty()) {
                insertion_broker_active = false;
                break;
            }
        }
    }

    if (pending->error) std::rethrow_exception(pending->error);
    return pending->results;
}

std::vector<InsertionScore> CudaComputeBackend::evaluate_insertion_batch_direct(
    const std::vector<std::vector<int>>& routes,
    const std::vector<InsertionRequest>& requests
) {
    if (requests.empty()) {
        return {};
    }

    // Bound transient host/device request storage on unusually large
    // generation batches. Request order and request_id values are retained
    // when the packs are merged.
    static constexpr size_t kMaxInsertionPack = 1u << 20;
    if (requests.size() > kMaxInsertionPack) {
        std::vector<InsertionScore> merged;
        merged.reserve(requests.size());
        for (size_t begin = 0; begin < requests.size(); begin += kMaxInsertionPack) {
            const size_t end = std::min(requests.size(), begin + kMaxInsertionPack);
            std::vector<InsertionRequest> pack(requests.begin() + begin, requests.begin() + end);
            std::vector<InsertionScore> scores = evaluate_insertion_batch_direct(routes, pack);
            merged.insert(
                merged.end(),
                std::make_move_iterator(scores.begin()),
                std::make_move_iterator(scores.end())
            );
        }
        return merged;
    }

    int request_count = static_cast<int>(requests.size());
    if (data.execution_policy != "cuda_force" && data.gpu_insertion_min_batch > 0 &&
        request_count < data.gpu_insertion_min_batch) {
        ScopedProfileTimer fallback_timer(data.profile, profile_registry().cuda_insertion_cpu_fallback, request_count);
        CpuComputeBackend cpu_fallback(data);
        return cpu_fallback.evaluate_insertion_batch(routes, requests);
    }

    ScopedProfileTimer timer(data.profile, profile_registry().insertion_batch, static_cast<long long>(requests.size()));
    std::lock_guard<std::mutex> lock(gpu_mutex);

    int route_count = static_cast<int>(routes.size());
    h_insert_batch_route_offsets.assign(route_count, 0);
    h_insert_batch_route_lengths.assign(route_count, 0);
    h_insert_batch_routes_flat.clear();
    size_t flat_nodes = 0;
    for (const auto& route : routes) {
        flat_nodes += route.size();
    }
    if (h_insert_batch_routes_flat.capacity() < flat_nodes) {
        h_insert_batch_routes_flat.reserve(flat_nodes);
    }
    for (int r = 0; r < route_count; ++r) {
        h_insert_batch_route_offsets[r] = static_cast<int>(h_insert_batch_routes_flat.size());
        h_insert_batch_route_lengths[r] = static_cast<int>(routes[r].size());
        h_insert_batch_routes_flat.insert(h_insert_batch_routes_flat.end(), routes[r].begin(), routes[r].end());
    }

    h_insert_batch_request_ids.resize(request_count);
    h_insert_batch_request_route_ids.resize(request_count);
    h_insert_batch_request_customer_ids.resize(request_count);
    h_insert_batch_request_positions.resize(request_count);
    for (int i = 0; i < request_count; ++i) {
        h_insert_batch_request_ids[i] = requests[i].request_id;
        h_insert_batch_request_route_ids[i] = requests[i].route_id;
        h_insert_batch_request_customer_ids[i] = requests[i].customer_id;
        h_insert_batch_request_positions[i] = requests[i].insert_position;
    }

    int max_route_len = 0;
    for (const auto& route : routes) {
        max_route_len = std::max(max_route_len, static_cast<int>(route.size()));
    }
    bool all_routes_valid = max_route_len > 0;
    h_insert_batch_prefix_attrs.assign(
        static_cast<size_t>(route_count) * static_cast<size_t>(max_route_len),
        DeviceAttr{}
    );
    h_insert_batch_suffix_attrs.assign(
        static_cast<size_t>(route_count) * static_cast<size_t>(max_route_len),
        DeviceAttr{}
    );
    h_insert_batch_base_distances.assign(route_count, 0.0);
    for (int route_id = 0; route_id < route_count; ++route_id) {
        const std::vector<int>& route = routes[route_id];
        const size_t attr_base = static_cast<size_t>(route_id) * static_cast<size_t>(max_route_len);
        if (route.size() < 2 || route.front() != snapshot.depot || route.back() != snapshot.depot) {
            all_routes_valid = false;
            continue;
        }

        h_insert_batch_prefix_attrs[attr_base] = host_attr_for_node(snapshot, route[0]);
        bool route_valid = true;
        for (size_t pos = 1; pos < route.size(); ++pos) {
            DeviceAttr node_attr = host_attr_for_node(snapshot, route[pos]);
            DeviceAttr merged;
            if (!route_valid || !host_connect_attr(
                    snapshot,
                    h_insert_batch_prefix_attrs[attr_base + pos - 1],
                    node_attr,
                    merged)) {
                route_valid = false;
                continue;
            }
            h_insert_batch_prefix_attrs[attr_base + pos] = merged;
        }

        h_insert_batch_suffix_attrs[attr_base + route.size() - 1] =
            host_attr_for_node(snapshot, route.back());
        for (int pos = static_cast<int>(route.size()) - 2; pos >= 0; --pos) {
            DeviceAttr node_attr = host_attr_for_node(snapshot, route[pos]);
            DeviceAttr merged;
            if (!route_valid || !host_connect_attr(
                    snapshot,
                    node_attr,
                    h_insert_batch_suffix_attrs[attr_base + pos + 1],
                    merged)) {
                route_valid = false;
                continue;
            }
            h_insert_batch_suffix_attrs[attr_base + pos] = merged;
        }
        if (!route_valid) {
            all_routes_valid = false;
        } else {
            h_insert_batch_base_distances[route_id] =
                h_insert_batch_prefix_attrs[attr_base + route.size() - 1].dist;
        }
    }
    const bool use_attr_kernel = data.architecture == "hybrid_v2" && all_routes_valid;

    ensure_insert_batch_capacity(h_insert_batch_routes_flat.size(), routes.size(), requests.size());
    if (use_attr_kernel) {
        ensure_insert_attr_capacity(
            static_cast<size_t>(route_count) * static_cast<size_t>(max_route_len),
            routes.size()
        );
    }

    size_t input_bytes = 0;
    const size_t routes_offset = reserve_aligned_bytes(input_bytes, h_insert_batch_routes_flat.size(), sizeof(int), alignof(int));
    const size_t route_offsets_offset = reserve_aligned_bytes(input_bytes, h_insert_batch_route_offsets.size(), sizeof(int), alignof(int));
    const size_t route_lengths_offset = reserve_aligned_bytes(input_bytes, h_insert_batch_route_lengths.size(), sizeof(int), alignof(int));
    const size_t request_ids_offset = reserve_aligned_bytes(input_bytes, h_insert_batch_request_ids.size(), sizeof(int), alignof(int));
    const size_t request_route_ids_offset = reserve_aligned_bytes(input_bytes, h_insert_batch_request_route_ids.size(), sizeof(int), alignof(int));
    const size_t request_customer_ids_offset = reserve_aligned_bytes(input_bytes, h_insert_batch_request_customer_ids.size(), sizeof(int), alignof(int));
    const size_t request_positions_offset = reserve_aligned_bytes(input_bytes, h_insert_batch_request_positions.size(), sizeof(int), alignof(int));
    size_t prefix_attrs_offset = 0;
    size_t suffix_attrs_offset = 0;
    size_t base_distances_offset = 0;
    if (use_attr_kernel) {
        prefix_attrs_offset = reserve_aligned_bytes(input_bytes, h_insert_batch_prefix_attrs.size(), sizeof(DeviceAttr), alignof(DeviceAttr));
        suffix_attrs_offset = reserve_aligned_bytes(input_bytes, h_insert_batch_suffix_attrs.size(), sizeof(DeviceAttr), alignof(DeviceAttr));
        base_distances_offset = reserve_aligned_bytes(input_bytes, h_insert_batch_base_distances.size(), sizeof(double), alignof(double));
    }

    size_t output_bytes = 0;
    const size_t out_request_ids_offset = reserve_aligned_bytes(output_bytes, requests.size(), sizeof(int), alignof(int));
    const size_t out_feasible_offset = reserve_aligned_bytes(output_bytes, requests.size(), sizeof(int), alignof(int));
    const size_t out_flags_offset = reserve_aligned_bytes(output_bytes, requests.size(), sizeof(int), alignof(int));
    const size_t out_delta_offset = reserve_aligned_bytes(output_bytes, requests.size(), sizeof(double), alignof(double));
    const size_t out_total_offset = reserve_aligned_bytes(output_bytes, requests.size(), sizeof(double), alignof(double));
    const size_t out_fitness_offset = reserve_aligned_bytes(output_bytes, requests.size(), sizeof(double), alignof(double));
    ensure_insert_pinned_capacity(input_bytes, output_bytes);

    int* h_routes = reinterpret_cast<int*>(h_insert_batch_input_pinned + routes_offset);
    int* h_route_offsets = reinterpret_cast<int*>(h_insert_batch_input_pinned + route_offsets_offset);
    int* h_route_lengths = reinterpret_cast<int*>(h_insert_batch_input_pinned + route_lengths_offset);
    int* h_request_ids = reinterpret_cast<int*>(h_insert_batch_input_pinned + request_ids_offset);
    int* h_request_route_ids = reinterpret_cast<int*>(h_insert_batch_input_pinned + request_route_ids_offset);
    int* h_request_customer_ids = reinterpret_cast<int*>(h_insert_batch_input_pinned + request_customer_ids_offset);
    int* h_request_positions = reinterpret_cast<int*>(h_insert_batch_input_pinned + request_positions_offset);
    DeviceAttr* h_prefix_attrs = use_attr_kernel
        ? reinterpret_cast<DeviceAttr*>(h_insert_batch_input_pinned + prefix_attrs_offset)
        : nullptr;
    DeviceAttr* h_suffix_attrs = use_attr_kernel
        ? reinterpret_cast<DeviceAttr*>(h_insert_batch_input_pinned + suffix_attrs_offset)
        : nullptr;
    double* h_base_distances = use_attr_kernel
        ? reinterpret_cast<double*>(h_insert_batch_input_pinned + base_distances_offset)
        : nullptr;
    int* h_out_request_ids = reinterpret_cast<int*>(h_insert_batch_output_pinned + out_request_ids_offset);
    int* h_out_feasible = reinterpret_cast<int*>(h_insert_batch_output_pinned + out_feasible_offset);
    int* h_out_flags = reinterpret_cast<int*>(h_insert_batch_output_pinned + out_flags_offset);
    double* h_out_delta_distance = reinterpret_cast<double*>(h_insert_batch_output_pinned + out_delta_offset);
    double* h_out_total_distance = reinterpret_cast<double*>(h_insert_batch_output_pinned + out_total_offset);
    double* h_out_fitness = reinterpret_cast<double*>(h_insert_batch_output_pinned + out_fitness_offset);

    if (!h_insert_batch_routes_flat.empty()) {
        std::memcpy(h_routes, h_insert_batch_routes_flat.data(), h_insert_batch_routes_flat.size() * sizeof(int));
    }
    if (!h_insert_batch_route_offsets.empty()) {
        std::memcpy(h_route_offsets, h_insert_batch_route_offsets.data(), h_insert_batch_route_offsets.size() * sizeof(int));
        std::memcpy(h_route_lengths, h_insert_batch_route_lengths.data(), h_insert_batch_route_lengths.size() * sizeof(int));
    }
    std::memcpy(h_request_ids, h_insert_batch_request_ids.data(), h_insert_batch_request_ids.size() * sizeof(int));
    std::memcpy(h_request_route_ids, h_insert_batch_request_route_ids.data(), h_insert_batch_request_route_ids.size() * sizeof(int));
    std::memcpy(h_request_customer_ids, h_insert_batch_request_customer_ids.data(), h_insert_batch_request_customer_ids.size() * sizeof(int));
    std::memcpy(h_request_positions, h_insert_batch_request_positions.data(), h_insert_batch_request_positions.size() * sizeof(int));
    if (use_attr_kernel) {
        std::memcpy(h_prefix_attrs, h_insert_batch_prefix_attrs.data(), h_insert_batch_prefix_attrs.size() * sizeof(DeviceAttr));
        std::memcpy(h_suffix_attrs, h_insert_batch_suffix_attrs.data(), h_insert_batch_suffix_attrs.size() * sizeof(DeviceAttr));
        std::memcpy(h_base_distances, h_insert_batch_base_distances.data(), h_insert_batch_base_distances.size() * sizeof(double));
    }

    const long long h2d_copy_bytes = static_cast<long long>(
        h_insert_batch_routes_flat.size() * sizeof(int)
        + h_insert_batch_route_offsets.size() * sizeof(int)
        + h_insert_batch_route_lengths.size() * sizeof(int)
        + h_insert_batch_request_ids.size() * sizeof(int)
        + h_insert_batch_request_route_ids.size() * sizeof(int)
        + h_insert_batch_request_customer_ids.size() * sizeof(int)
        + h_insert_batch_request_positions.size() * sizeof(int)
        + (use_attr_kernel ? (
            h_insert_batch_prefix_attrs.size() * sizeof(DeviceAttr)
            + h_insert_batch_suffix_attrs.size() * sizeof(DeviceAttr)
            + h_insert_batch_base_distances.size() * sizeof(double)
        ) : 0)
    );
    const long long d2h_copy_bytes = static_cast<long long>(requests.size()) *
        static_cast<long long>(3 * sizeof(int) + 3 * sizeof(double));

    cuda_check(cudaEventRecord(insertion_stage_events[0], coordinator_stream), "cudaEventRecord(insertion_h2d_start)");
    if (!h_insert_batch_routes_flat.empty()) {
        cuda_check(cudaMemcpyAsync(d_insert_batch_routes_flat, h_routes, h_insert_batch_routes_flat.size() * sizeof(int), cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_routes_flat)");
    }
    if (!h_insert_batch_route_offsets.empty()) {
        cuda_check(cudaMemcpyAsync(d_insert_batch_route_offsets, h_route_offsets, h_insert_batch_route_offsets.size() * sizeof(int), cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_route_offsets)");
        cuda_check(cudaMemcpyAsync(d_insert_batch_route_lengths, h_route_lengths, h_insert_batch_route_lengths.size() * sizeof(int), cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_route_lengths)");
    }
    cuda_check(cudaMemcpyAsync(d_insert_batch_request_ids, h_request_ids, h_insert_batch_request_ids.size() * sizeof(int), cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_request_ids)");
    cuda_check(cudaMemcpyAsync(d_insert_batch_request_route_ids, h_request_route_ids, h_insert_batch_request_route_ids.size() * sizeof(int), cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_request_route_ids)");
    cuda_check(cudaMemcpyAsync(d_insert_batch_request_customer_ids, h_request_customer_ids, h_insert_batch_request_customer_ids.size() * sizeof(int), cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_request_customer_ids)");
    cuda_check(cudaMemcpyAsync(d_insert_batch_request_positions, h_request_positions, h_insert_batch_request_positions.size() * sizeof(int), cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_request_positions)");
    if (use_attr_kernel) {
        cuda_check(cudaMemcpyAsync(d_insert_batch_prefix_attrs, h_prefix_attrs, h_insert_batch_prefix_attrs.size() * sizeof(DeviceAttr), cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_prefix_attrs)");
        cuda_check(cudaMemcpyAsync(d_insert_batch_suffix_attrs, h_suffix_attrs, h_insert_batch_suffix_attrs.size() * sizeof(DeviceAttr), cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_suffix_attrs)");
        cuda_check(cudaMemcpyAsync(d_insert_batch_base_distances, h_base_distances, h_insert_batch_base_distances.size() * sizeof(double), cudaMemcpyHostToDevice, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_base_distances)");
    }
    cuda_check(cudaEventRecord(insertion_stage_events[1], coordinator_stream), "cudaEventRecord(insertion_h2d_end)");

    int threads_per_block = 256;
    int blocks = (request_count + threads_per_block - 1) / threads_per_block;
    int stride = snapshot.customer_num + 1;

    if (use_attr_kernel) {
            evaluateInsertionBatchAttrKernel<<<blocks, threads_per_block, 0, coordinator_stream>>>(
                d_insert_batch_route_lengths,
                route_count,
                max_route_len,
                d_insert_batch_request_ids,
                d_insert_batch_request_route_ids,
                d_insert_batch_request_customer_ids,
                d_insert_batch_request_positions,
                request_count,
                snapshot.customer_num,
                snapshot.depot,
                snapshot.capacity,
                snapshot.dispatch_cost,
                snapshot.unit_cost,
                d_delivery,
                d_pickup,
                d_start,
                d_end,
                d_service,
                d_dist,
                d_time,
                stride,
                d_insert_batch_prefix_attrs,
                d_insert_batch_suffix_attrs,
                d_insert_batch_base_distances,
                d_insert_batch_out_request_ids,
                d_insert_batch_out_feasible,
                d_insert_batch_out_delta_distance,
                d_insert_batch_out_total_distance,
                d_insert_batch_out_fitness,
                d_insert_batch_out_flags
            );
            cuda_check(cudaGetLastError(), "evaluateInsertionBatchAttrKernel launch");
    } else {
            evaluateInsertionBatchKernel<<<blocks, threads_per_block, 0, coordinator_stream>>>(
                d_insert_batch_routes_flat,
                d_insert_batch_route_offsets,
                d_insert_batch_route_lengths,
                route_count,
                d_insert_batch_request_ids,
                d_insert_batch_request_route_ids,
                d_insert_batch_request_customer_ids,
                d_insert_batch_request_positions,
                request_count,
                snapshot.customer_num,
                snapshot.depot,
                snapshot.start_time,
                snapshot.capacity,
                snapshot.dispatch_cost,
                snapshot.unit_cost,
                d_delivery,
                d_pickup,
                d_start,
                d_end,
                d_service,
                d_dist,
                d_time,
                stride,
                d_insert_batch_out_request_ids,
                d_insert_batch_out_feasible,
                d_insert_batch_out_delta_distance,
                d_insert_batch_out_total_distance,
                d_insert_batch_out_fitness,
                d_insert_batch_out_flags
            );
            cuda_check(cudaGetLastError(), "evaluateInsertionBatchKernel launch");
    }
    cuda_check(cudaEventRecord(insertion_stage_events[2], coordinator_stream), "cudaEventRecord(insertion_kernel_end)");
    cuda_check(cudaMemcpyAsync(h_out_request_ids, d_insert_batch_out_request_ids, requests.size() * sizeof(int), cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_out_request_ids)");
    cuda_check(cudaMemcpyAsync(h_out_feasible, d_insert_batch_out_feasible, requests.size() * sizeof(int), cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_out_feasible)");
    cuda_check(cudaMemcpyAsync(h_out_flags, d_insert_batch_out_flags, requests.size() * sizeof(int), cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_out_flags)");
    cuda_check(cudaMemcpyAsync(h_out_delta_distance, d_insert_batch_out_delta_distance, requests.size() * sizeof(double), cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_out_delta_distance)");
    cuda_check(cudaMemcpyAsync(h_out_total_distance, d_insert_batch_out_total_distance, requests.size() * sizeof(double), cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_out_total_distance)");
    cuda_check(cudaMemcpyAsync(h_out_fitness, d_insert_batch_out_fitness, requests.size() * sizeof(double), cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(d_insert_batch_out_fitness)");
    cuda_check(cudaEventRecord(insertion_stage_events[3], coordinator_stream), "cudaEventRecord(insertion_d2h_end)");
    cuda_check(cudaEventSynchronize(insertion_stage_events[3]), "cudaEventSynchronize(insertion_d2h_end)");

    if (data.profile) {
        float h2d_ms = 0.0f;
        float kernel_ms = 0.0f;
        float d2h_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&h2d_ms, insertion_stage_events[0], insertion_stage_events[1]), "cudaEventElapsedTime(insertion_h2d)");
        cuda_check(cudaEventElapsedTime(&kernel_ms, insertion_stage_events[1], insertion_stage_events[2]), "cudaEventElapsedTime(insertion_kernel)");
        cuda_check(cudaEventElapsedTime(&d2h_ms, insertion_stage_events[2], insertion_stage_events[3]), "cudaEventElapsedTime(insertion_d2h)");
        profile_registry().cuda_insertion_h2d.add(static_cast<long long>(h2d_ms * 1e6), h2d_copy_bytes);
        profile_registry().cuda_insertion_kernel.add(static_cast<long long>(kernel_ms * 1e6), request_count);
        profile_registry().cuda_insertion_d2h.add(static_cast<long long>(d2h_ms * 1e6), d2h_copy_bytes);
    }

    std::vector<InsertionScore> results(request_count);
    for (int i = 0; i < request_count; ++i) {
        results[i].request_id = h_out_request_ids[i];
        results[i].feasible = h_out_feasible[i] == 1;
        results[i].delta_distance = h_out_delta_distance[i];
        results[i].total_distance_after = h_out_total_distance[i];
        results[i].fitness_after = h_out_fitness[i];
        results[i].violation_flags = h_out_flags[i];
    }
    return results;
}

void CudaComputeBackend::calibrate_execution_policy() {
    if (snapshot.customer_num <= 0) return;

    CpuComputeBackend cpu(data);
    std::vector<int> route{snapshot.depot, snapshot.depot};
    std::vector<unsigned char> in_route(snapshot.customer_num + 1, 0);
    const int representative_customer_limit = std::min(snapshot.customer_num, 64);
    for (int customer = 0; customer <= snapshot.customer_num; ++customer) {
        if (customer == snapshot.depot) continue;
        std::vector<int> candidate = route;
        candidate.insert(candidate.end() - 1, customer);
        if (cpu.evaluate_route(candidate).feasible) {
            route.swap(candidate);
            in_route[customer] = 1;
            if (static_cast<int>(route.size()) - 2 >= representative_customer_limit) break;
        }
    }
    if (route.size() == 2) return;

    int insertion_customer = -1;
    for (int customer = 0; customer <= snapshot.customer_num; ++customer) {
        if (customer != snapshot.depot && in_route[customer] == 0) {
            insertion_customer = customer;
            break;
        }
    }
    if (insertion_customer == -1) {
        insertion_customer = route[1];
    }

    std::vector<size_t> route_counts{128, 256, 512, 1024, 2048, 4096};
    if (snapshot.customer_num > 100) {
        route_counts.push_back(16384);
        route_counts.push_back(65536);
    }
    adaptive_route_min_items = std::numeric_limits<size_t>::max();
    adaptive_route_min_nodes = std::numeric_limits<size_t>::max();

    for (size_t count : route_counts) {
        std::vector<std::vector<int>> routes(count, route);
        const auto cpu_start = std::chrono::high_resolution_clock::now();
        auto cpu_results = cpu.evaluate_routes(routes);
        const auto cpu_us = std::chrono::duration<double, std::micro>(
            std::chrono::high_resolution_clock::now() - cpu_start
        ).count();

        const auto gpu_start = std::chrono::high_resolution_clock::now();
        auto gpu_results = evaluate_route_batch_direct(routes);
        const auto gpu_us = std::chrono::duration<double, std::micro>(
            std::chrono::high_resolution_clock::now() - gpu_start
        ).count();
        if (cpu_results.size() == gpu_results.size() && gpu_us * 1.15 < cpu_us) {
            adaptive_route_min_items = count;
            adaptive_route_min_nodes = count * route.size();
            break;
        }
    }

    std::vector<size_t> request_counts{128, 512, 2048, 8192};
    if (snapshot.customer_num > 100) {
        request_counts.push_back(32768);
        request_counts.push_back(131072);
    }
    const std::vector<std::vector<int>> insertion_routes{route};
    adaptive_insertion_min_items = std::numeric_limits<size_t>::max();
    adaptive_insertion_min_nodes = insertion_routes[0].size();
    for (size_t count : request_counts) {
        std::vector<InsertionRequest> requests(count);
        for (size_t i = 0; i < count; ++i) {
            requests[i].request_id = static_cast<int>(i);
            requests[i].route_id = 0;
            requests[i].customer_id = insertion_customer;
            requests[i].insert_position = 1 + static_cast<int>(i % (route.size() - 1));
            requests[i].source_context = INSERTION_CONTEXT_REPAIR;
        }

        const auto cpu_start = std::chrono::high_resolution_clock::now();
        auto cpu_results = cpu.evaluate_insertion_batch(insertion_routes, requests);
        const auto cpu_us = std::chrono::duration<double, std::micro>(
            std::chrono::high_resolution_clock::now() - cpu_start
        ).count();

        const auto gpu_start = std::chrono::high_resolution_clock::now();
        auto gpu_results = evaluate_insertion_batch_direct(insertion_routes, requests);
        const auto gpu_us = std::chrono::duration<double, std::micro>(
            std::chrono::high_resolution_clock::now() - gpu_start
        ).count();
        if (cpu_results.size() == gpu_results.size() && gpu_us * 1.15 < cpu_us) {
            adaptive_insertion_min_items = count;
            break;
        }
    }

    std::cout << "Adaptive CUDA crossover: route_items=";
    if (adaptive_route_min_items == std::numeric_limits<size_t>::max()) {
        std::cout << "disabled";
    } else {
        std::cout << adaptive_route_min_items << ", route_nodes=" << adaptive_route_min_nodes;
    }
    std::cout << ", insertion_items=";
    if (adaptive_insertion_min_items == std::numeric_limits<size_t>::max()) {
        std::cout << "disabled";
    } else {
        std::cout << adaptive_insertion_min_items;
    }
    std::cout << std::endl;
}

void CudaComputeBackend::ensure_insert_pinned_capacity(size_t input_bytes, size_t output_bytes) {
    ScopedProfileTimer timer(data.profile, profile_registry().cuda_insertion_alloc);
    ensure_pinned_buffer(
        h_insert_batch_input_pinned,
        h_insert_batch_input_capacity,
        input_bytes,
        "cudaHostAlloc(h_insert_batch_input)"
    );
    ensure_pinned_buffer(
        h_insert_batch_output_pinned,
        h_insert_batch_output_capacity,
        output_bytes,
        "cudaHostAlloc(h_insert_batch_output)"
    );
}

std::vector<SolutionEval> CudaComputeBackend::evaluate_solutions(const EncodedPopulation& encoded) {
    if (data.execution_policy == "adaptive") {
        WorkShape shape;
        shape.context = EVALUATION_CONTEXT_POPULATION;
        shape.item_count = static_cast<size_t>(encoded.solution_count);
        shape.total_nodes = encoded.routes_flat.size();
        shape.max_length = static_cast<size_t>(std::max(0, encoded.max_route_len));
        shape.transfer_bytes = encoded.routes_flat.size() * sizeof(int);
        if (choose_target(shape) != EvalTarget::CudaBatch) {
            profile_registry().dispatch_cpu_batch.add(0, encoded.solution_count);
            CpuComputeBackend cpu(data);
            return cpu.evaluate_solutions(encoded);
        }
    }
    ScopedProfileTimer timer(data.profile, profile_registry().feasibility_batch, encoded.solution_count);
    std::lock_guard<std::mutex> lock(gpu_mutex);

    if (encoded.solution_count == 0) {
        return {};
    }

    const size_t old_node_slot_capacity = solution_slot_node_capacity;
    const size_t old_route_slot_capacity = solution_slot_route_capacity;
    ensure_solution_capacity(encoded);

    const bool layout_changed =
        persistent_solution_count != encoded.solution_count ||
        old_node_slot_capacity != solution_slot_node_capacity ||
        old_route_slot_capacity != solution_slot_route_capacity;
    const bool same_layout = !layout_changed;
    const size_t total_route_slots =
        static_cast<size_t>(encoded.solution_count) * solution_slot_route_capacity;

    if (layout_changed) {
        h_persistent_route_offsets.assign(total_route_slots, 0);
        h_persistent_route_lengths.assign(total_route_slots, 0);
        h_persistent_solution_offsets.resize(encoded.solution_count);
        h_persistent_solution_route_counts.assign(encoded.solution_count, 0);
        h_persistent_solution_revisions.assign(encoded.solution_count, 0);
        for (int solution_id = 0; solution_id < encoded.solution_count; ++solution_id) {
            h_persistent_solution_offsets[solution_id] = static_cast<int>(
                static_cast<size_t>(solution_id) * solution_slot_route_capacity
            );
        }
        persistent_solution_count = encoded.solution_count;
    }

    long long copy_bytes = 0;
    const auto h2d_start = std::chrono::high_resolution_clock::now();
    {
        if (layout_changed && !h_persistent_solution_offsets.empty()) {
            cuda_check(cudaMemcpyAsync(
                d_solution_offsets,
                h_persistent_solution_offsets.data(),
                h_persistent_solution_offsets.size() * sizeof(int),
                cudaMemcpyHostToDevice,
                coordinator_stream
            ), "cudaMemcpyAsync(d_solution_offsets)");
            copy_bytes += static_cast<long long>(h_persistent_solution_offsets.size() * sizeof(int));
        }

        for (int solution_id = 0; solution_id < encoded.solution_count; ++solution_id) {
            const bool dirty = layout_changed || encoded.dirty_flags.empty() ||
                encoded.dirty_flags[solution_id];
            if (!dirty) {
                continue;
            }

            const int encoded_route_start = encoded.solution_offsets[solution_id];
            const int route_count = encoded.solution_route_counts[solution_id];
            const size_t route_slot_start =
                static_cast<size_t>(solution_id) * solution_slot_route_capacity;
            const size_t node_slot_start =
                static_cast<size_t>(solution_id) * solution_slot_node_capacity;
            if (route_count < 0 || static_cast<size_t>(route_count) > solution_slot_route_capacity) {
                throw std::runtime_error("Solution route count exceeds persistent CUDA slot capacity");
            }

            size_t node_cursor = 0;
            int encoded_node_start = 0;
            for (int route = 0; route < route_count; ++route) {
                const int encoded_route = encoded_route_start + route;
                if (route == 0) {
                    encoded_node_start = encoded.route_offsets[encoded_route];
                }
                const int length = encoded.route_lengths[encoded_route];
                h_persistent_route_offsets[route_slot_start + route] = static_cast<int>(
                    node_slot_start + node_cursor
                );
                h_persistent_route_lengths[route_slot_start + route] = length;
                node_cursor += static_cast<size_t>(std::max(0, length));
            }
            if (node_cursor > solution_slot_node_capacity) {
                throw std::runtime_error("Solution node count exceeds persistent CUDA slot capacity");
            }
            if (node_cursor > 0) {
                cuda_check(cudaMemcpyAsync(
                    d_solution_routes_flat + node_slot_start,
                    encoded.routes_flat.data() + encoded_node_start,
                    node_cursor * sizeof(int),
                    cudaMemcpyHostToDevice,
                    coordinator_stream
                ), "cudaMemcpyAsync(d_solution_routes_flat slot)");
                copy_bytes += static_cast<long long>(node_cursor * sizeof(int));
            }
            if (route_count > 0) {
                cuda_check(cudaMemcpyAsync(
                    d_solution_route_offsets + route_slot_start,
                    h_persistent_route_offsets.data() + route_slot_start,
                    static_cast<size_t>(route_count) * sizeof(int),
                    cudaMemcpyHostToDevice,
                    coordinator_stream
                ), "cudaMemcpyAsync(d_solution_route_offsets slot)");
                cuda_check(cudaMemcpyAsync(
                    d_solution_route_lengths + route_slot_start,
                    h_persistent_route_lengths.data() + route_slot_start,
                    static_cast<size_t>(route_count) * sizeof(int),
                    cudaMemcpyHostToDevice,
                    coordinator_stream
                ), "cudaMemcpyAsync(d_solution_route_lengths slot)");
                copy_bytes += static_cast<long long>(
                    2 * static_cast<size_t>(route_count) * sizeof(int)
                );
            }
            h_persistent_solution_route_counts[solution_id] = route_count;
            cuda_check(cudaMemcpyAsync(
                d_solution_route_counts + solution_id,
                &h_persistent_solution_route_counts[solution_id],
                sizeof(int),
                cudaMemcpyHostToDevice,
                coordinator_stream
            ), "cudaMemcpyAsync(d_solution_route_counts slot)");
            copy_bytes += sizeof(int);
            h_persistent_solution_revisions[solution_id]++;
        }
    }
    if (data.profile) {
        const long long elapsed_nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - h2d_start
        ).count();
        profile_registry().cuda_h2d.add(elapsed_nanos, copy_bytes);
    }

    std::vector<int> eval_solution_ids;
    if (!same_layout || h_persistent_solution_results.size() != static_cast<size_t>(encoded.solution_count) ||
        encoded.dirty_flags.empty()) {
        eval_solution_ids.resize(encoded.solution_count);
        std::iota(eval_solution_ids.begin(), eval_solution_ids.end(), 0);
        h_persistent_solution_results.assign(encoded.solution_count, SolutionEval{});
    } else {
        eval_solution_ids.reserve(encoded.solution_count);
        for (int i = 0; i < encoded.solution_count; ++i) {
            if (encoded.dirty_flags[i]) eval_solution_ids.push_back(i);
        }
    }

    if (eval_solution_ids.empty()) {
        return h_persistent_solution_results;
    }

    cuda_check(cudaEventRecord(route_stage_events[0], coordinator_stream), "cudaEventRecord(solution_eval_ids_start)");
    cuda_check(cudaMemcpyAsync(
        d_solution_eval_ids,
        eval_solution_ids.data(),
        eval_solution_ids.size() * sizeof(int),
        cudaMemcpyHostToDevice,
        coordinator_stream
    ), "cudaMemcpyAsync(d_solution_eval_ids)");
    cuda_check(cudaEventRecord(route_stage_events[1], coordinator_stream), "cudaEventRecord(solution_eval_ids_end)");

    int threads_per_block = 128;
    const int eval_count = static_cast<int>(eval_solution_ids.size());
    int blocks = (eval_count + threads_per_block - 1) / threads_per_block;
    int stride = snapshot.customer_num + 1;

    evaluateSolutionsKernel<<<blocks, threads_per_block, 0, coordinator_stream>>>(
            d_solution_routes_flat,
            d_solution_route_offsets,
            d_solution_route_lengths,
            d_solution_offsets,
            d_solution_route_counts,
            d_solution_eval_ids,
            eval_count,
            snapshot.customer_num,
            snapshot.depot,
            snapshot.start_time,
            snapshot.capacity,
            snapshot.dispatch_cost,
            snapshot.unit_cost,
            d_delivery,
            d_pickup,
            d_start,
            d_end,
            d_service,
            d_dist,
            d_time,
            stride,
            d_solution_seen,
            d_solution_feasible,
            d_solution_fitness,
            d_solution_vehicle_counts,
            d_solution_distance,
            d_solution_violation_flags
    );
    cuda_check(cudaGetLastError(), "evaluateSolutionsKernel launch");
    cuda_check(cudaEventRecord(route_stage_events[2], coordinator_stream), "cudaEventRecord(solution_kernel_end)");

    std::vector<int> h_feasible(eval_count);
    std::vector<int> h_vehicle_counts(eval_count);
    std::vector<int> h_violation_flags(eval_count);
    std::vector<double> h_fitness(eval_count);
    std::vector<double> h_distance(eval_count);

    const long long d2h_copy_bytes = static_cast<long long>(
        h_feasible.size() * sizeof(int)
        + h_vehicle_counts.size() * sizeof(int)
        + h_violation_flags.size() * sizeof(int)
        + h_fitness.size() * sizeof(double)
        + h_distance.size() * sizeof(double)
    );
    cuda_check(cudaMemcpyAsync(h_feasible.data(), d_solution_feasible, h_feasible.size() * sizeof(int), cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(d_solution_feasible)");
    cuda_check(cudaMemcpyAsync(h_vehicle_counts.data(), d_solution_vehicle_counts, h_vehicle_counts.size() * sizeof(int), cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(d_solution_vehicle_counts)");
    cuda_check(cudaMemcpyAsync(h_violation_flags.data(), d_solution_violation_flags, h_violation_flags.size() * sizeof(int), cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(d_solution_violation_flags)");
    cuda_check(cudaMemcpyAsync(h_fitness.data(), d_solution_fitness, h_fitness.size() * sizeof(double), cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(d_solution_fitness)");
    cuda_check(cudaMemcpyAsync(h_distance.data(), d_solution_distance, h_distance.size() * sizeof(double), cudaMemcpyDeviceToHost, coordinator_stream), "cudaMemcpyAsync(d_solution_distance)");
    cuda_check(cudaEventRecord(route_stage_events[3], coordinator_stream), "cudaEventRecord(solution_d2h_end)");
    cuda_check(cudaEventSynchronize(route_stage_events[3]), "cudaEventSynchronize(solution_d2h_end)");
    if (data.profile) {
        float eval_id_h2d_ms = 0.0f;
        float kernel_ms = 0.0f;
        float d2h_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&eval_id_h2d_ms, route_stage_events[0], route_stage_events[1]), "cudaEventElapsedTime(solution_eval_ids)");
        cuda_check(cudaEventElapsedTime(&kernel_ms, route_stage_events[1], route_stage_events[2]), "cudaEventElapsedTime(solution_kernel)");
        cuda_check(cudaEventElapsedTime(&d2h_ms, route_stage_events[2], route_stage_events[3]), "cudaEventElapsedTime(solution_d2h)");
        profile_registry().cuda_h2d.add(
            static_cast<long long>(eval_id_h2d_ms * 1e6),
            static_cast<long long>(eval_solution_ids.size() * sizeof(int))
        );
        profile_registry().cuda_kernel.add(static_cast<long long>(kernel_ms * 1e6), eval_count);
        profile_registry().cuda_d2h.add(static_cast<long long>(d2h_ms * 1e6), d2h_copy_bytes);
    }

    for (int output_index = 0; output_index < eval_count; ++output_index) {
        const int solution_id = eval_solution_ids[output_index];
        SolutionEval& result = h_persistent_solution_results[solution_id];
        result.feasible = h_feasible[output_index] == 1;
        result.fitness = h_fitness[output_index];
        result.distance = h_distance[output_index];
        result.vehicle_count = h_vehicle_counts[output_index];
        result.violation_flags = h_violation_flags[output_index];
    }
    return h_persistent_solution_results;
}

#endif
