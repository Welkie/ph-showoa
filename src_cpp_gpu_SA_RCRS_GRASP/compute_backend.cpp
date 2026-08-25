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
    int num_positions = route_len > 1 ? route_len - 1 : 0;
    int total_evals = static_cast<int>(candidates.size()) * num_positions;
    out_feasible.assign(total_evals, 0);
    out_costs.assign(total_evals, 0.0);

    if (route_len < 2 || route.front() != data.DC || route.back() != data.DC) {
        return;
    }

    double capacity = data.vehicle.capacity;
    double dispatch_cost = data.vehicle.d_cost;
    double unit_cost = data.vehicle.unit_cost;
    double start_time = data.start_time;

    for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
        int candidate = candidates[c_idx];
        if (candidate < 0 || candidate > data.customer_num || candidate == data.DC) {
            continue;
        }
        for (int pos = 1; pos < route_len; ++pos) {
            int out_idx = static_cast<int>(c_idx) * num_positions + (pos - 1);

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
        const bool paper_scale_cpu = data.execution_policy == "adaptive" &&
            data.customer_num <= 100 && data.p_size <= 64 && data.num_islands == 1 &&
            data.gpu_solution_verify_interval == 0;
        if (paper_scale_cpu) {
            std::cout << "Adaptive execution policy selected the CPU backend for this workload."
                      << std::endl;
            return new CpuComputeBackend(data);
        }
        try {
            return create_cuda_backend(data);
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

std::vector<InsertionScore> CpuComputeBackend::evaluate_insertion_batch(
    const std::vector<std::vector<int>>& routes,
    const std::vector<InsertionRequest>& requests
) {
    std::vector<InsertionScore> results;
    results.reserve(requests.size());

    struct Attr {
        int s = 0;
        int e = 0;
        int num_cus = 0;
        double dist = 0.0;
        double T_D = 0.0;
        double T_E = 0.0;
        double T_L = 0.0;
        double C_E = 0.0;
        double C_H = 0.0;
        double C_L = 0.0;
        bool valid = false;
    };

    auto attr_for_one_node = [&](int node, Attr& out) {
        out.s = node;
        out.e = node;
        out.valid = true;
        if (node == data.DC) {
            out.num_cus = 0;
            out.T_E = data.start_time;
            out.T_L = data.end_time;
            return;
        }
        out.num_cus = 1;
        out.T_D = data.node[node].s_time;
        out.T_E = data.node[node].start;
        out.T_L = data.node[node].end;
        out.C_E = data.node[node].delivery;
        out.C_L = data.node[node].pickup;
        out.C_H = std::max(out.C_E, out.C_L);
    };

    auto connect_into = [&](const Attr& a, const Attr& b, Attr& out, double dist_ab, double time_ab) {
        if (!a.valid || !b.valid ||
            a.T_E + a.T_D + time_ab - b.T_L > 0.0 ||
            std::max(a.C_H + b.C_E, a.C_L + b.C_H) > data.vehicle.capacity) {
            out.valid = false;
            return false;
        }

        out.s = a.s;
        out.e = b.e;
        out.num_cus = a.num_cus + b.num_cus;
        out.dist = a.dist + dist_ab + b.dist;
        const double delta = a.T_D + time_ab;
        const double waiting = std::max(b.T_E - delta - a.T_L, 0.0);
        out.T_D = a.T_D + b.T_D + time_ab + waiting;
        out.T_E = std::max(b.T_E - delta, a.T_E) - waiting;
        out.T_L = std::min(b.T_L - delta, a.T_L);
        out.C_E = a.C_E + b.C_E;
        out.C_H = std::max(a.C_H + b.C_E, a.C_L + b.C_H);
        out.C_L = a.C_L + b.C_L;
        out.valid = true;
        return true;
    };

    struct RouteResourceState {
        bool valid = false;
        double base_distance = 0.0;
        std::vector<Attr> prefix;
        std::vector<Attr> suffix;
    };

    std::vector<RouteResourceState> route_resources(routes.size());
    std::vector<bool> route_referenced(routes.size(), false);
    for (const InsertionRequest& req : requests) {
        if (req.route_id >= 0 && req.route_id < static_cast<int>(routes.size())) {
            route_referenced[req.route_id] = true;
        }
    }

    for (int r_idx = 0; r_idx < static_cast<int>(routes.size()); ++r_idx) {
        if (!route_referenced[r_idx]) continue;
        const std::vector<int>& route = routes[r_idx];

        bool valid = route.size() >= 2 && route.front() == data.DC && route.back() == data.DC;
        for (int node : route) {
            if (node < 0 || node > data.customer_num) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;

        RouteResourceState& state = route_resources[r_idx];
        state.prefix.resize(route.size());
        state.suffix.resize(route.size());

        attr_for_one_node(route.front(), state.prefix.front());
        state.valid = true;
        for (size_t pos = 1; pos < route.size(); ++pos) {
            Attr node_attr;
            attr_for_one_node(route[pos], node_attr);
            if (!connect_into(
                state.prefix[pos - 1],
                node_attr,
                state.prefix[pos],
                data.dist[route[pos - 1]][route[pos]],
                data.time[route[pos - 1]][route[pos]]
            )) {
                state.valid = false;
                break;
            }
        }
        if (!state.valid) continue;

        attr_for_one_node(route.back(), state.suffix.back());
        for (int pos = static_cast<int>(route.size()) - 2; pos >= 0; --pos) {
            Attr node_attr;
            attr_for_one_node(route[pos], node_attr);
            if (!connect_into(
                node_attr,
                state.suffix[pos + 1],
                state.suffix[pos],
                data.dist[route[pos]][route[pos + 1]],
                data.time[route[pos]][route[pos + 1]]
            )) {
                state.valid = false;
                break;
            }
        }
        if (!state.valid) continue;
        state.base_distance = state.prefix.back().dist;
    }

    for (const InsertionRequest& req : requests) {
        InsertionScore score;
        score.request_id = req.request_id;

        if (req.route_id < 0 || req.route_id >= static_cast<int>(routes.size())) {
            score.violation_flags = INSERTION_BAD_ROUTE;
            results.push_back(score);
            continue;
        }

        const std::vector<int>& route = routes[req.route_id];
        int route_len = static_cast<int>(route.size());
        if (!route_resources[req.route_id].valid) {
            score.violation_flags = INSERTION_BAD_ROUTE;
            results.push_back(score);
            continue;
        }

        if (req.customer_id <= 0 || req.customer_id > data.customer_num || req.customer_id == data.DC) {
            score.violation_flags = INSERTION_BAD_CUSTOMER;
            results.push_back(score);
            continue;
        }

        if (req.insert_position < 1 || req.insert_position >= route_len) {
            score.violation_flags = INSERTION_BAD_POSITION;
            results.push_back(score);
            continue;
        }

        const RouteResourceState& resource_route = route_resources[req.route_id];
        Attr merged = resource_route.prefix[req.insert_position - 1];
        Attr customer_attr;
        attr_for_one_node(req.customer_id, customer_attr);
        int flags = INSERTION_OK;
        auto append_attr = [&](const Attr& next) {
            if (merged.T_E + merged.T_D + data.time[merged.e][next.s] - next.T_L > 0.0) {
                flags |= INSERTION_TIME_WINDOW;
            }
            if (std::max(merged.C_H + next.C_E, merged.C_L + next.C_H) > data.vehicle.capacity) {
                flags |= INSERTION_CAPACITY;
            }
            if (flags == INSERTION_OK) {
                Attr out;
                if (connect_into(
                    merged,
                    next,
                    out,
                    data.dist[merged.e][next.s],
                    data.time[merged.e][next.s]
                )) {
                    merged = out;
                }
            }
        };

        append_attr(customer_attr);
        if (flags == INSERTION_OK) {
            append_attr(resource_route.suffix[req.insert_position]);
        }

        score.violation_flags = flags;
        score.feasible = (flags == INSERTION_OK);
        if (score.feasible) {
            score.total_distance_after = merged.dist;
            score.delta_distance = merged.dist - resource_route.base_distance;
            score.fitness_after = data.vehicle.d_cost + merged.dist * data.vehicle.unit_cost;
        }
        results.push_back(score);
    }

    return results;
}

BackendSnapshot::BackendSnapshot(const Data& data) {
    depot = data.DC;
    customer_num = data.customer_num;
    capacity = data.vehicle.capacity;
    start_time = data.node[data.DC].start;
    end_time = data.node[data.DC].end;
    dispatch_cost = data.vehicle.d_cost;
    unit_cost = data.vehicle.unit_cost;
    int n = customer_num + 1;
    delivery.resize(n);
    pickup.resize(n);
    start.resize(n);
    end.resize(n);
    service.resize(n);
    for (int i = 0; i < n; ++i) {
        delivery[i] = data.node[i].delivery;
        pickup[i] = data.node[i].pickup;
        start[i] = data.node[i].start;
        end[i] = data.node[i].end;
        service[i] = data.node[i].s_time;
    }
    dist.resize(n * n);
    time.resize(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dist[i * n + j] = data.dist[i][j];
            time[i * n + j] = data.time[i][j];
        }
    }
}

std::vector<SolutionEval> CpuComputeBackend::evaluate_solutions(const EncodedPopulation& encoded) {
    return evaluate_encoded_population_cpu(encoded, data);
}

EvalTarget CpuComputeBackend::choose_target(const WorkShape& shape) const {
    if (shape.context == EVALUATION_CONTEXT_LOCAL_SEARCH) {
        return EvalTarget::CpuIncremental;
    }
    return EvalTarget::CpuBatch;
}
