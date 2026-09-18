#include "solution_encoding.h"
#include <cmath>
#include <limits>
#include <sstream>

namespace {
constexpr double kEvalTolerance = 1e-6;

static inline bool nearly_equal(double a, double b, double tol = kEvalTolerance) {
    return std::fabs(a - b) <= tol;
}

static void set_error(std::string* error_message, const std::string& message) {
    if (error_message != nullptr) {
        *error_message = message;
    }
}

static bool compare_route_order(
    const Solution& solution,
    const EncodedPopulation& encoded,
    int solution_id,
    std::string* error_message
) {
    if (solution_id < 0 || solution_id >= encoded.solution_count) {
        set_error(error_message, "solution_id out of range");
        return false;
    }

    int route_start = encoded.solution_offsets[solution_id];
    int route_count = encoded.solution_route_counts[solution_id];
    if (route_count != solution.len()) {
        std::ostringstream ss;
        ss << "route count mismatch: solution=" << solution.len()
           << " encoded=" << route_count;
        set_error(error_message, ss.str());
        return false;
    }

    for (int r = 0; r < route_count; ++r) {
        const Route& route = solution.get(r);
        int encoded_route = route_start + r;
        int offset = encoded.route_offsets[encoded_route];
        int length = encoded.route_lengths[encoded_route];
        if (length != static_cast<int>(route.node_list.size())) {
            std::ostringstream ss;
            ss << "route length mismatch at route " << r
               << ": solution=" << route.node_list.size()
               << " encoded=" << length;
            set_error(error_message, ss.str());
            return false;
        }
        for (int i = 0; i < length; ++i) {
            int encoded_node = encoded.routes_flat[offset + i];
            if (encoded_node != route.node_list[i]) {
                std::ostringstream ss;
                ss << "node mismatch at route " << r << ", position " << i
                   << ": solution=" << route.node_list[i]
                   << " encoded=" << encoded_node;
                set_error(error_message, ss.str());
                return false;
            }
        }
    }

    return true;
}
} // namespace

EncodedPopulation encode_population_for_gpu(const std::vector<Solution>& population, const Data& data, const std::vector<bool>& pop_dirty) {
    EncodedPopulation encoded;
    encoded.solution_count = static_cast<int>(population.size());
    encoded.customer_count = data.customer_num;
    encoded.depot = data.DC;
    encoded.solution_offsets.reserve(population.size());
    encoded.solution_route_counts.reserve(population.size());
    encoded.dirty_flags.reserve(population.size());

    int route_offset = 0;
    int node_offset = 0;
    for (size_t i = 0; i < population.size(); ++i) {
        const Solution& solution = population[i];
        bool is_dirty = pop_dirty.empty() ? true : pop_dirty[i];
        
        encoded.solution_offsets.push_back(route_offset);
        encoded.solution_route_counts.push_back(solution.len());
        encoded.dirty_flags.push_back(is_dirty);
        
        route_offset += solution.len();

        for (int r = 0; r < solution.len(); ++r) {
            const Route& route = solution.get(r);
            int length = static_cast<int>(route.node_list.size());
            encoded.route_offsets.push_back(node_offset);
            encoded.route_lengths.push_back(length);
            encoded.max_route_len = std::max(encoded.max_route_len, length);
            encoded.routes_flat.insert(encoded.routes_flat.end(), route.node_list.begin(), route.node_list.end());
            node_offset += length;
        }
    }

    encoded.route_count = route_offset;
    return encoded;
}

EncodedPopulation encode_solution_for_gpu(const Solution& solution, const Data& data) {
    std::vector<Solution> population;
    population.push_back(solution);
    return encode_population_for_gpu(population, data);
}

Solution decode_solution_from_gpu(const EncodedPopulation& encoded, int solution_id, const Data& data) {
    Solution decoded;
    if (solution_id < 0 || solution_id >= encoded.solution_count) {
        return decoded;
    }

    int route_start = encoded.solution_offsets[solution_id];
    int route_count = encoded.solution_route_counts[solution_id];
    for (int r = 0; r < route_count; ++r) {
        int encoded_route = route_start + r;
        int offset = encoded.route_offsets[encoded_route];
        int length = encoded.route_lengths[encoded_route];
        Route route(data);
        route.node_list.assign(encoded.routes_flat.begin() + offset, encoded.routes_flat.begin() + offset + length);
        route.update(data);
        decoded.append(route);
    }
    decoded.cal_cost(data);
    return decoded;
}

SolutionEval evaluate_encoded_solution_cpu(const EncodedPopulation& encoded, int solution_id, const Data& data) {
    SolutionEval result;
    if (solution_id < 0 || solution_id >= encoded.solution_count) {
        result.violation_flags |= SOLUTION_BAD_NODE;
        return result;
    }

    std::vector<int> seen(data.customer_num + 1, 0);
    int route_start = encoded.solution_offsets[solution_id];
    int route_count = encoded.solution_route_counts[solution_id];

    for (int r = 0; r < route_count; ++r) {
        int encoded_route = route_start + r;
        if (encoded_route < 0 || encoded_route >= encoded.route_count) {
            result.violation_flags |= SOLUTION_BAD_NODE;
            continue;
        }

        int offset = encoded.route_offsets[encoded_route];
        int length = encoded.route_lengths[encoded_route];
        if (length < 2 || offset < 0 || offset + length > static_cast<int>(encoded.routes_flat.size())) {
            result.violation_flags |= SOLUTION_BAD_DEPOT;
            continue;
        }

        int first = encoded.routes_flat[offset];
        int last = encoded.routes_flat[offset + length - 1];
        if (first != data.DC || last != data.DC) {
            result.violation_flags |= SOLUTION_BAD_DEPOT;
            continue;
        }

        if (length == 2) {
            continue;
        }

        result.vehicle_count += 1;
        double load = 0.0;
        for (int pos = 1; pos < length - 1; ++pos) {
            int node = encoded.routes_flat[offset + pos];
            if (node < 0 || node > data.customer_num || node == data.DC) {
                result.violation_flags |= SOLUTION_BAD_NODE;
                continue;
            }
            load += data.node[node].delivery;
            seen[node] += 1;
            if (seen[node] > 1) {
                result.violation_flags |= SOLUTION_DUPLICATE_CUSTOMER;
            }
        }
        if (load > data.vehicle.capacity) {
            result.violation_flags |= SOLUTION_CAPACITY;
        }

        double time_val = data.start_time;
        int prev = first;
        for (int pos = 1; pos < length; ++pos) {
            int node = encoded.routes_flat[offset + pos];
            if (node < 0 || node > data.customer_num) {
                result.violation_flags |= SOLUTION_BAD_NODE;
                break;
            }

            load = load - data.node[node].delivery + data.node[node].pickup;
            if (load < 0.0 || load > data.vehicle.capacity) {
                result.violation_flags |= SOLUTION_CAPACITY;
            }

            time_val += data.time[prev][node];
            if (time_val > data.node[node].end) {
                result.violation_flags |= SOLUTION_TIME_WINDOW;
            }
            if (time_val < data.node[node].start) {
                time_val = data.node[node].start;
            }
            time_val += data.node[node].s_time;
            result.distance += data.dist[prev][node];
            prev = node;
        }
    }

    for (int node = 0; node <= data.customer_num; ++node) {
        if (node == data.DC) {
            continue;
        }
        if (seen[node] == 0) {
            result.violation_flags |= SOLUTION_MISSING_CUSTOMER;
        }
    }

    result.fitness = data.vehicle.d_cost * result.vehicle_count + data.vehicle.unit_cost * result.distance;
    result.feasible = result.violation_flags == SOLUTION_OK;
    return result;
}

std::vector<SolutionEval> evaluate_encoded_population_cpu(const EncodedPopulation& encoded, const Data& data) {
    std::vector<SolutionEval> results(encoded.solution_count);
    for (int i = 0; i < encoded.solution_count; ++i) {
        results[i] = evaluate_encoded_solution_cpu(encoded, i, data);
    }
    return results;
}

bool validate_encoded_solution(
    const Solution& solution,
    const EncodedPopulation& encoded,
    int solution_id,
    const Data& data,
    std::string* error_message
) {
    if (!compare_route_order(solution, encoded, solution_id, error_message)) {
        return false;
    }

    SolutionEval encoded_eval = evaluate_encoded_solution_cpu(encoded, solution_id, data);

    Solution cpu_solution = solution;
    cpu_solution.cal_cost(data);
    bool cpu_feasible = cpu_solution.check(data, false);
    if (cpu_feasible != encoded_eval.feasible) {
        std::ostringstream ss;
        ss << "feasibility mismatch: cpu=" << cpu_feasible
           << " encoded=" << encoded_eval.feasible
           << " flags=" << encoded_eval.violation_flags;
        set_error(error_message, ss.str());
        return false;
    }

    if (cpu_feasible && !nearly_equal(cpu_solution.cost, encoded_eval.fitness)) {
        std::ostringstream ss;
        ss << "fitness mismatch: cpu=" << cpu_solution.cost
           << " encoded=" << encoded_eval.fitness;
        set_error(error_message, ss.str());
        return false;
    }

    return true;
}
