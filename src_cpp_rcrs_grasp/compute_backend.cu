#ifdef USE_CUDA

#include "compute_backend.h"
#include <cuda_runtime.h>
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
} // namespace

// --- CUDA Kernels ---

__global__ void evaluate_route_batch_kernel(
    const int* routes,
    const int* lengths,
    const int num_routes,
    const int max_len,
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
    const double* dist,       // flat (customer_num + 1) * (customer_num + 1)
    const double* time_matrix, // flat
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

    int pitch = max_len;
    int first_node = routes[idx * pitch];
    int last_node = routes[idx * pitch + length - 1];

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
        int node = routes[idx * pitch + pos];
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
        int node = routes[idx * pitch + pos];
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
    const int stride, // customer_num + 1
    int* out_feasible,
    double* out_costs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_evals = num_candidates * route_len;
    if (idx >= total_evals) return;

    int c_idx = idx / route_len;
    int pos = idx % route_len;

    if (pos == 0) {
        out_feasible[idx] = 0;
        out_costs[idx] = 0.0;
        return;
    }

    int candidate = candidates[c_idx];

    // Fast reject capacity
    double load = 0.0;
    for (int i = 1; i < route_len - 1; ++i) {
        load += delivery[route[i]];
    }
    load += delivery[candidate];
    if (load > capacity) {
        out_feasible[idx] = 0;
        out_costs[idx] = 0.0;
        return;
    }

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

// --- CudaComputeBackend Implementation ---

CudaComputeBackend::CudaComputeBackend(const Data& d) : data(d) {
    cudaError_t err = cudaSuccess;
    int device_count = 0;
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        throw std::runtime_error("CUDA device count error or no devices found");
    }

    ensure_device_cache();
}

CudaComputeBackend::~CudaComputeBackend() {
    free_device_cache();
}

void CudaComputeBackend::ensure_device_cache() {
    int n = data.customer_num + 1;
    size_t size_dbl = n * sizeof(double);
    size_t size_matrix = n * n * sizeof(double);

    // Host flat representation helpers
    std::vector<double> h_delivery(n), h_pickup(n), h_start(n), h_end(n), h_service(n);
    std::vector<double> h_dist(n * n), h_time(n * n);

    for (int i = 0; i < n; ++i) {
        h_delivery[i] = data.node[i].delivery;
        h_pickup[i] = data.node[i].pickup;
        h_start[i] = data.node[i].start;
        h_end[i] = data.node[i].end;
        h_service[i] = data.node[i].s_time;
        for (int j = 0; j < n; ++j) {
            h_dist[i * n + j] = data.dist[i][j];
            h_time[i * n + j] = data.time[i][j];
        }
    }

    cuda_check(cudaMalloc((void**)&d_delivery, size_dbl), "cudaMalloc(d_delivery)");
    cuda_check(cudaMalloc((void**)&d_pickup, size_dbl), "cudaMalloc(d_pickup)");
    cuda_check(cudaMalloc((void**)&d_start, size_dbl), "cudaMalloc(d_start)");
    cuda_check(cudaMalloc((void**)&d_end, size_dbl), "cudaMalloc(d_end)");
    cuda_check(cudaMalloc((void**)&d_service, size_dbl), "cudaMalloc(d_service)");
    cuda_check(cudaMalloc((void**)&d_dist, size_matrix), "cudaMalloc(d_dist)");
    cuda_check(cudaMalloc((void**)&d_time, size_matrix), "cudaMalloc(d_time)");

    cuda_check(cudaMemcpy(d_delivery, h_delivery.data(), size_dbl, cudaMemcpyHostToDevice), "cudaMemcpy(d_delivery)");
    cuda_check(cudaMemcpy(d_pickup, h_pickup.data(), size_dbl, cudaMemcpyHostToDevice), "cudaMemcpy(d_pickup)");
    cuda_check(cudaMemcpy(d_start, h_start.data(), size_dbl, cudaMemcpyHostToDevice), "cudaMemcpy(d_start)");
    cuda_check(cudaMemcpy(d_end, h_end.data(), size_dbl, cudaMemcpyHostToDevice), "cudaMemcpy(d_end)");
    cuda_check(cudaMemcpy(d_service, h_service.data(), size_dbl, cudaMemcpyHostToDevice), "cudaMemcpy(d_service)");
    cuda_check(cudaMemcpy(d_dist, h_dist.data(), size_matrix, cudaMemcpyHostToDevice), "cudaMemcpy(d_dist)");
    cuda_check(cudaMemcpy(d_time, h_time.data(), size_matrix, cudaMemcpyHostToDevice), "cudaMemcpy(d_time)");
}

void CudaComputeBackend::free_device_cache() {
    if (d_delivery) cudaFree(d_delivery);
    if (d_pickup) cudaFree(d_pickup);
    if (d_start) cudaFree(d_start);
    if (d_end) cudaFree(d_end);
    if (d_service) cudaFree(d_service);
    if (d_dist) cudaFree(d_dist);
    if (d_time) cudaFree(d_time);

    d_delivery = nullptr;
    d_pickup = nullptr;
    d_start = nullptr;
    d_end = nullptr;
    d_service = nullptr;
    d_dist = nullptr;
    d_time = nullptr;
}

RouteEval CudaComputeBackend::evaluate_route(const std::vector<int>& route) {
    // Falls back to CPU logic for single route checks to bypass CUDA launch latency
    CpuComputeBackend cpu_fallback(data);
    return cpu_fallback.evaluate_route(route);
}

std::vector<RouteEval> CudaComputeBackend::evaluate_routes(const std::vector<std::vector<int>>& routes) {
    if (routes.empty()) return {};

    int num_routes = routes.size();
    int max_len = 0;
    for (const auto& r : routes) {
        max_len = std::max(max_len, (int)r.size());
    }

    std::vector<int> h_packed(num_routes * max_len, data.DC);
    std::vector<int> h_lengths(num_routes);

    for (int i = 0; i < num_routes; ++i) {
        h_lengths[i] = routes[i].size();
        for (int j = 0; j < routes[i].size(); ++j) {
            h_packed[i * max_len + j] = routes[i][j];
        }
    }

    int* d_packed = nullptr;
    int* d_lengths = nullptr;
    int* d_feasible = nullptr;
    double* d_costs = nullptr;

    cuda_check(cudaMalloc((void**)&d_packed, num_routes * max_len * sizeof(int)), "cudaMalloc(d_packed)");
    cuda_check(cudaMalloc((void**)&d_lengths, num_routes * sizeof(int)), "cudaMalloc(d_lengths)");
    cuda_check(cudaMalloc((void**)&d_feasible, num_routes * sizeof(int)), "cudaMalloc(d_feasible)");
    cuda_check(cudaMalloc((void**)&d_costs, num_routes * sizeof(double)), "cudaMalloc(d_costs)");

    cuda_check(cudaMemcpy(d_packed, h_packed.data(), num_routes * max_len * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(d_packed)");
    cuda_check(cudaMemcpy(d_lengths, h_lengths.data(), num_routes * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(d_lengths)");

    int threads_per_block = 128;
    int blocks = (num_routes + threads_per_block - 1) / threads_per_block;
    int stride = data.customer_num + 1;

    evaluate_route_batch_kernel<<<blocks, threads_per_block>>>(
        d_packed, d_lengths, num_routes, max_len, stride, data.DC,
        data.start_time, data.vehicle.capacity, data.vehicle.d_cost, data.vehicle.unit_cost,
        d_delivery, d_pickup, d_start, d_end, d_service, d_dist, d_time,
        d_feasible, d_costs
    );
    cuda_check(cudaGetLastError(), "evaluate_route_batch_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "evaluate_route_batch_kernel synchronize");

    std::vector<int> h_feasible(num_routes);
    std::vector<double> h_costs(num_routes);

    cuda_check(cudaMemcpy(h_feasible.data(), d_feasible, num_routes * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(h_feasible)");
    cuda_check(cudaMemcpy(h_costs.data(), d_costs, num_routes * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(h_costs)");

    cudaFree(d_packed);
    cudaFree(d_lengths);
    cudaFree(d_feasible);
    cudaFree(d_costs);

    std::vector<RouteEval> results(num_routes);
    for (int i = 0; i < num_routes; ++i) {
        results[i] = RouteEval{h_feasible[i] == 1, h_costs[i]};
    }
    return results;
}

void CudaComputeBackend::evaluate_insertions(
    const std::vector<int>& route,
    const std::vector<int>& candidates,
    std::vector<int>& out_feasible,
    std::vector<double>& out_costs
) {
    if (candidates.empty()) return;

    int route_len = route.size();
    int num_candidates = candidates.size();
    int total_evals = num_candidates * route_len;

    out_feasible.assign(total_evals, 0);
    out_costs.assign(total_evals, 0.0);

    int* d_route = nullptr;
    int* d_candidates = nullptr;
    int* d_feasible = nullptr;
    double* d_costs = nullptr;

    cuda_check(cudaMalloc((void**)&d_route, route_len * sizeof(int)), "cudaMalloc(d_route)");
    cuda_check(cudaMalloc((void**)&d_candidates, num_candidates * sizeof(int)), "cudaMalloc(d_candidates)");
    cuda_check(cudaMalloc((void**)&d_feasible, total_evals * sizeof(int)), "cudaMalloc(d_feasible)");
    cuda_check(cudaMalloc((void**)&d_costs, total_evals * sizeof(double)), "cudaMalloc(d_costs)");

    cuda_check(cudaMemcpy(d_route, route.data(), route_len * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(d_route)");
    cuda_check(cudaMemcpy(d_candidates, candidates.data(), num_candidates * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(d_candidates)");

    int threads_per_block = 256;
    int blocks = (total_evals + threads_per_block - 1) / threads_per_block;

    int stride = data.customer_num + 1;

    evaluate_insertions_cuda_kernel<<<blocks, threads_per_block>>>(
        d_route, route_len, d_candidates, num_candidates, data.DC,
        data.start_time, data.vehicle.capacity, data.vehicle.d_cost, data.vehicle.unit_cost,
        d_delivery, d_pickup, d_start, d_end, d_service, d_dist, d_time,
        stride, d_feasible, d_costs
    );
    cuda_check(cudaGetLastError(), "evaluate_insertions_cuda_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "evaluate_insertions_cuda_kernel synchronize");

    cuda_check(cudaMemcpy(out_feasible.data(), d_feasible, total_evals * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(out_feasible)");
    cuda_check(cudaMemcpy(out_costs.data(), d_costs, total_evals * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(out_costs)");

    cudaFree(d_route);
    cudaFree(d_candidates);
    cudaFree(d_feasible);
    cudaFree(d_costs);
}

#endif
