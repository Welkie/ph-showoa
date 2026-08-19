#include "compute_backend.h"
#include <algorithm>
#include <cctype>
#include <exception>
#include <iostream>

RouteEval CpuComputeBackend::evaluate_route(const std::vector<int>& route) {
    int length = route.size();
    if (length < 2) {
        return RouteEval{false, 0.0};
    }

    int depot = data.DC;
    if (route[0] != depot || route[length - 1] != depot) {
        return RouteEval{false, 0.0};
    }

    if (length == 2) {
        return RouteEval{true, 0.0};
    }

    double capacity = data.vehicle.capacity;
    double load = 0.0;
    for (int i = 1; i < length - 1; ++i) {
        load += data.node[route[i]].delivery;
    }

    if (load > capacity) {
        return RouteEval{false, 0.0};
    }

    double distance = 0.0;
    double time_val = data.start_time;
    int prev = route[0];

    for (int i = 1; i < length; ++i) {
        int node = route[i];
        load = load - data.node[node].delivery + data.node[node].pickup;
        if (load < 0.0 || load > capacity) {
            return RouteEval{false, 0.0};
        }

        time_val += data.time[prev][node];
        if (time_val > data.node[node].end) {
            return RouteEval{false, 0.0};
        }
        if (time_val < data.node[node].start) {
            time_val = data.node[node].start;
        }
        time_val += data.node[node].s_time;

        distance += data.dist[prev][node];
        prev = node;
    }

    double cost = data.vehicle.d_cost + distance * data.vehicle.unit_cost;
    return RouteEval{true, cost};
}

std::vector<RouteEval> CpuComputeBackend::evaluate_routes(const std::vector<std::vector<int>>& routes) {
    std::vector<RouteEval> results;
    results.reserve(routes.size());
    for (const auto& r : routes) {
        results.push_back(evaluate_route(r));
    }
    return results;
}

void CpuComputeBackend::evaluate_insertions(
    const std::vector<int>& route,
    const std::vector<int>& candidates,
    std::vector<int>& out_feasible,
    std::vector<double>& out_costs
) {
    int route_len = route.size();
    int total_evals = candidates.size() * route_len;
    out_feasible.assign(total_evals, 0);
    out_costs.assign(total_evals, 0.0);

    double capacity = data.vehicle.capacity;
    double dispatch_cost = data.vehicle.d_cost;
    double unit_cost = data.vehicle.unit_cost;
    double start_time = data.start_time;

    for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
        int candidate = candidates[c_idx];
        for (int pos = 1; pos < route_len; ++pos) {
            int out_idx = c_idx * route_len + pos;

            // Fast reject capacity
            double load = 0.0;
            for (int i = 1; i < route_len - 1; ++i) {
                load += data.node[route[i]].delivery;
            }
            load += data.node[candidate].delivery;
            if (load > capacity) {
                out_feasible[out_idx] = 0;
                continue;
            }

            double distance = 0.0;
            double time_val = start_time;
            int prev = route[0];
            bool is_feasible = true;

            for (int i = 1; i <= route_len; ++i) {
                int node = route[0]; // dummy init
                if (i == pos) {
                    node = candidate;
                } else if (i < pos) {
                    node = route[i];
                } else {
                    node = route[i - 1];
                }

                if (i == route_len) {
                    break;
                }

                load = load - data.node[node].delivery + data.node[node].pickup;
                if (load < 0.0 || load > capacity) {
                    is_feasible = false;
                    break;
                }

                time_val += data.time[prev][node];
                if (time_val > data.node[node].end) {
                    is_feasible = false;
                    break;
                }
                if (time_val < data.node[node].start) {
                    time_val = data.node[node].start;
                }
                time_val += data.node[node].s_time;

                distance += data.dist[prev][node];
                prev = node;
            }

            if (is_feasible) {
                out_feasible[out_idx] = 1;
                out_costs[out_idx] = dispatch_cost + distance * unit_cost;
            } else {
                out_feasible[out_idx] = 0;
            }
        }
    }
}

BaseComputeBackend* create_backend(const Data& data, const std::string& mode) {
    std::string requested = mode;
    std::transform(requested.begin(), requested.end(), requested.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

#ifdef USE_CUDA
    if (requested == "auto" || requested == "cuda") {
        try {
            return new CudaComputeBackend(data);
        } catch (const std::exception& e) {
            std::cout << "CUDA backend unavailable: " << e.what() << ". Falling back to CPU backend." << std::endl;
        }
    }
#else
    if (requested == "cuda") {
        std::cout << "CUDA backend requested but this binary was built without CUDA support. Using CPU backend." << std::endl;
    }
#endif

    return new CpuComputeBackend(data);
}
