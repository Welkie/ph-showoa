#include "operator.h"
#include "profile.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <memory>
#include <iterator>

// Global helper for local search move representation
static thread_local Move TMP_MOVE;

// Simple random helpers
static inline int randint(int low, int high, LegacyMt19937& rng) {
    return legacy_randint(low, high, rng);
}

static inline double randdouble(double low, double high, LegacyMt19937& rng) {
    return legacy_randdouble(low, high, rng);
}

static inline bool _is_customer(int node, const Data& data) {
    return node != data.DC;
}

static ProfileCounter& insertion_selection_counter(int context) {
    if (context == INSERTION_CONTEXT_INITIALIZATION) {
        return profile_registry().insertion_selection_initialization;
    }
    if (context == INSERTION_CONTEXT_LOCAL_SEARCH) {
        return profile_registry().insertion_selection_local_search;
    }
    return profile_registry().insertion_selection_repair;
}

static inline RouteEval _evaluate_route_cpu(const std::vector<int>& nl, const Data& data) {
    int length = nl.size();
    if (length < 2) {
        return RouteEval{false, 0.0};
    }
    if (nl[0] != data.DC || nl[length - 1] != data.DC) {
        return RouteEval{false, 0.0};
    }
    if (length == 2) {
        return RouteEval{true, 0.0};
    }

    double capacity = data.vehicle.capacity;
    double load = 0.0;
    for (int i = 1; i < length - 1; ++i) {
        load += data.node[nl[i]].delivery;
    }
    if (load > capacity) {
        return RouteEval{false, 0.0};
    }

    double distance = 0.0;
    double time_val = data.start_time;
    int prev = nl[0];

    for (int i = 1; i < length; ++i) {
        int node = nl[i];
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

bool _chk_route_list(const std::vector<int>& nl, const Data& data, BaseComputeBackend* backend) {
    if (backend != nullptr) {
        return backend->evaluate_route(nl).feasible;
    }
    return _evaluate_route_cpu(nl, data).feasible;
}

std::vector<RouteEval> evaluate_route_batch(const std::vector<std::vector<int>>& routes, const Data& data, BaseComputeBackend* backend) {
    if (backend != nullptr) {
        if (data.execution_policy == "adaptive" && backend->is_gpu_backend()) {
            WorkShape shape;
            shape.context = EVALUATION_CONTEXT_ROUTE_BATCH;
            shape.item_count = routes.size();
            for (const auto& route : routes) {
                shape.total_nodes += route.size();
                shape.max_length = std::max(shape.max_length, route.size());
            }
            shape.transfer_bytes = shape.total_nodes * sizeof(int);
            if (backend->choose_target(shape) != EvalTarget::CudaBatch) {
                profile_registry().dispatch_cpu_batch.add(
                    0, static_cast<long long>(routes.size())
                );
                CpuComputeBackend cpu(data);
                return cpu.evaluate_routes(routes);
            }
        }
        return backend->evaluate_routes(routes);
    }
    std::vector<RouteEval> results(routes.size());
    for (size_t i = 0; i < routes.size(); ++i) {
        results[i] = _evaluate_route_cpu(routes[i], data);
    }
    return results;
}

static std::vector<InsertionScore> evaluate_insertion_requests(
    const Data& data,
    BaseComputeBackend* backend,
    const std::vector<std::vector<int>>& routes,
    const std::vector<InsertionRequest>& requests
) {
    if (backend == nullptr) {
        CpuComputeBackend cpu(data);
        return cpu.evaluate_insertion_batch(routes, requests);
    }
    if (data.execution_policy == "adaptive" && backend->is_gpu_backend() && !requests.empty()) {
        WorkShape shape;
        shape.context = requests.front().source_context;
        shape.item_count = requests.size();
        for (const auto& route : routes) {
            shape.total_nodes += route.size();
            shape.max_length = std::max(shape.max_length, route.size());
        }
        shape.transfer_bytes = shape.total_nodes * sizeof(int) +
            requests.size() * 4 * sizeof(int);
        if (backend->choose_target(shape) != EvalTarget::CudaBatch) {
            profile_registry().dispatch_cpu_batch.add(
                0, static_cast<long long>(requests.size())
            );
            CpuComputeBackend cpu(data);
            return cpu.evaluate_insertion_batch(routes, requests);
        }
    }
    return backend->evaluate_insertion_batch(routes, requests);
}

static constexpr double INSERTION_TIE_EPSILON = 1e-6;

static RouteEval evaluate_insertion_request_cpu(
    const std::vector<std::vector<int>>& routes,
    const InsertionRequest& request,
    const Data& data
) {
    if (request.route_id < 0 || request.route_id >= static_cast<int>(routes.size())) {
        return RouteEval{false, 0.0};
    }
    const std::vector<int>& route = routes[request.route_id];
    if (request.insert_position < 1 ||
        request.insert_position >= static_cast<int>(route.size())) {
        return RouteEval{false, 0.0};
    }

    std::vector<int> candidate = route;
    candidate.insert(candidate.begin() + request.insert_position, request.customer_id);
    return _evaluate_route_cpu(candidate, data);
}

static int select_cpu_verified_insertion(
    const std::vector<std::vector<int>>& routes,
    const std::vector<InsertionRequest>& requests,
    const std::vector<InsertionScore>& scores,
    std::vector<double> metrics,
    const Data& data,
    bool replace_metric_with_cpu_cost,
    double decision_epsilon,
    double* selected_metric = nullptr
) {
    const size_t count = std::min(requests.size(), scores.size());
    metrics.resize(count, std::numeric_limits<double>::infinity());
    std::vector<unsigned char> verified(count, 0);
    for (size_t i = 0; i < count; ++i) {
        if (!scores[i].feasible || scores[i].request_id != requests[i].request_id) {
            metrics[i] = std::numeric_limits<double>::infinity();
        }
    }

    while (true) {
        int best_idx = -1;
        double best_metric = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(metrics[i])) {
                continue;
            }
            if (best_idx == -1 || metrics[i] < best_metric - decision_epsilon) {
                best_idx = static_cast<int>(i);
                best_metric = metrics[i];
            }
        }
        if (best_idx == -1) {
            return -1;
        }

        bool refined = false;
        for (size_t i = 0; i < count; ++i) {
            if (verified[i] || !std::isfinite(metrics[i]) ||
                std::abs(metrics[i] - best_metric) > INSERTION_TIE_EPSILON) {
                continue;
            }
            RouteEval cpu_eval = evaluate_insertion_request_cpu(routes, requests[i], data);
            verified[i] = 1;
            refined = true;
            if (!cpu_eval.feasible) {
                metrics[i] = std::numeric_limits<double>::infinity();
            } else if (replace_metric_with_cpu_cost) {
                metrics[i] = cpu_eval.cost;
            }
        }
        if (refined) {
            continue;
        }

        if (!verified[best_idx]) {
            RouteEval cpu_eval = evaluate_insertion_request_cpu(routes, requests[best_idx], data);
            verified[best_idx] = 1;
            if (!cpu_eval.feasible) {
                metrics[best_idx] = std::numeric_limits<double>::infinity();
                continue;
            }
            if (replace_metric_with_cpu_cost) {
                metrics[best_idx] = cpu_eval.cost;
                continue;
            }
        }

        if (selected_metric != nullptr) {
            *selected_metric = metrics[best_idx];
        }
        return best_idx;
    }
}

static inline bool _best_insertion_position(
    const Solution& s,
    const Data& data,
    BaseComputeBackend* backend,
    int customer,
    int& best_r_idx,
    int& best_pos_idx,
    double& best_cost
) {
    if (s.len() == 0) {
        return false;
    }

    std::unique_ptr<CpuComputeBackend> cpu_backend;
    BaseComputeBackend* eval_backend = backend;
    if (eval_backend == nullptr) {
        cpu_backend = std::make_unique<CpuComputeBackend>(data);
        eval_backend = cpu_backend.get();
    }
    std::vector<int> candidates{customer};
    std::vector<int> out_feasible;
    std::vector<double> out_costs;
    bool found = false;

    if (data.gpu_batched_insertion) {
        std::vector<std::vector<int>> routes;
        std::vector<InsertionRequest> requests;
        routes.reserve(s.len());
        int request_id = 0;
        for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
            const Route& r = s.get(r_idx);
            routes.push_back(r.node_list);
            if (r.node_list.size() < 2) {
                continue;
            }
            for (int pos = 1; pos < static_cast<int>(r.node_list.size()); ++pos) {
                InsertionRequest req;
                req.request_id = request_id++;
                req.route_id = r_idx;
                req.customer_id = customer;
                req.insert_position = pos;
                req.source_context = INSERTION_CONTEXT_REPAIR;
                req.context.operator_type = CANDIDATE_OPERATOR_REPAIR;
                req.context.local_order = req.request_id;
                requests.push_back(req);
            }
        }

        if (requests.empty()) {
            return false;
        }

        std::vector<InsertionScore> scores = evaluate_insertion_requests(
            data, eval_backend, routes, requests
        );
        ScopedProfileTimer selection_timer(data.profile, profile_registry().insertion_selection, static_cast<long long>(scores.size()));
        ScopedProfileTimer context_selection_timer(
            data.profile,
            insertion_selection_counter(INSERTION_CONTEXT_REPAIR),
            static_cast<long long>(scores.size())
        );
        std::vector<double> metrics(scores.size(), std::numeric_limits<double>::infinity());
        for (size_t idx = 0; idx < scores.size(); ++idx) {
            metrics[idx] = scores[idx].fitness_after;
        }
        double verified_cost = std::numeric_limits<double>::infinity();
        int best_idx = select_cpu_verified_insertion(
            routes, requests, scores, std::move(metrics), data, true, 0.0, &verified_cost
        );
        if (best_idx >= 0 && verified_cost < best_cost) {
            best_cost = verified_cost;
            best_r_idx = requests[best_idx].route_id;
            best_pos_idx = requests[best_idx].insert_position;
            found = true;
        }
        return found;
    }

    std::vector<std::vector<int>> routes;
    std::vector<InsertionRequest> requests;
    std::vector<InsertionScore> scores;
    routes.reserve(s.len());
    int request_id = 0;
    for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
        const Route& r = s.get(r_idx);
        routes.push_back(r.node_list);
        if (r.node_list.size() < 2) {
            continue;
        }

        eval_backend->evaluate_insertions(r.node_list, candidates, out_feasible, out_costs);
        int num_positions = static_cast<int>(r.node_list.size()) - 1;
        for (int pos = 1; pos < static_cast<int>(r.node_list.size()); ++pos) {
            int eval_idx = pos - 1;
            if (eval_idx < 0 || eval_idx >= num_positions) {
                continue;
            }
            if (eval_idx >= static_cast<int>(out_feasible.size()) || out_feasible[eval_idx] == 0) {
                continue;
            }
            InsertionRequest request;
            request.request_id = request_id;
            request.route_id = r_idx;
            request.customer_id = customer;
            request.insert_position = pos;
            request.source_context = INSERTION_CONTEXT_REPAIR;
            requests.push_back(request);

            InsertionScore score;
            score.request_id = request_id++;
            score.feasible = true;
            score.fitness_after = out_costs[eval_idx];
            scores.push_back(score);
        }
    }

    std::vector<double> metrics(scores.size(), std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < scores.size(); ++i) {
        metrics[i] = scores[i].fitness_after;
    }
    double verified_cost = std::numeric_limits<double>::infinity();
    int best_idx = select_cpu_verified_insertion(
        routes, requests, scores, std::move(metrics), data, true, 0.0, &verified_cost
    );
    if (best_idx >= 0 && verified_cost < best_cost) {
        best_cost = verified_cost;
        best_r_idx = requests[best_idx].route_id;
        best_pos_idx = requests[best_idx].insert_position;
        found = true;
    }

    return found;
}

bool check_capacity(const Attr& a, const Attr& b, const Data& data) {
    return std::max(a.C_H + b.C_E, a.C_L + b.C_H) - data.vehicle.capacity <= 0.0;
}

bool check_tw(const Attr& a, const Attr& b, const Data& data) {
    return (a.T_E + a.T_D + data.time[a.e][b.s] - b.T_L) <= 0.0;
}

bool eval_route(const Solution& s, const Seq* seq_list, int seq_list_len, Attr& tmp_attr, const Data& data) {
    if (seq_list_len < 2) return false;

    Attr attr_1;
    if (seq_list[0].r_index == -1) {
        attr_for_one_node(data, seq_list[0].start_point, attr_1);
    } else {
        attr_1 = s.get(seq_list[0].r_index).gat(seq_list[0].start_point, seq_list[0].end_point);
    }

    Attr attr_2;
    if (seq_list[1].r_index == -1) {
        attr_for_one_node(data, seq_list[1].start_point, attr_2);
    } else {
        attr_2 = s.get(seq_list[1].r_index).gat(seq_list[1].start_point, seq_list[1].end_point);
    }

    if (!check_tw(attr_1, attr_2, data) || !check_capacity(attr_1, attr_2, data)) {
        return false;
    }
    connect_into(
        attr_1, attr_2, tmp_attr,
        data.dist[attr_1.e][attr_2.s],
        data.time[attr_1.e][attr_2.s]
    );

    for (int i = 2; i < seq_list_len; ++i) {
        Attr attr;
        if (seq_list[i].r_index == -1) {
            attr_for_one_node(data, seq_list[i].start_point, attr);
        } else {
            attr = s.get(seq_list[i].r_index).gat(seq_list[i].start_point, seq_list[i].end_point);
        }

        if (!check_tw(tmp_attr, attr, data) || !check_capacity(tmp_attr, attr, data)) {
            return false;
        }
        connect_inplace(tmp_attr, attr, data.dist[tmp_attr.e][attr.s], data.time[tmp_attr.e][attr.s]);
    }

    return true;
}

bool eval_move(const Solution& s, Move& m, const Data& data, BaseComputeBackend* backend) {
    std::vector<int> r_indice;
    r_indice.push_back(m.r_indice[0]);
    if (m.r_indice[1] != -2) {
        r_indice.push_back(m.r_indice[1]);
    }

    double ori_cost = route_total_cost(s.get(r_indice[0]), data);

    if (!data.O_1_evl) {
        std::vector<int> target_n_l;
        for (int i = 0; i < m.len_1; ++i) {
            const Seq& seq = m.seqList_1[i];
            const std::vector<int>& source_n_l = s.get(seq.r_index).node_list;
            if (seq.start_point <= seq.end_point) {
                for (int index = seq.start_point; index <= seq.end_point; ++index) {
                    target_n_l.push_back(source_n_l[index]);
                }
            } else {
                for (int index = seq.start_point; index >= seq.end_point; --index) {
                    target_n_l.push_back(source_n_l[index]);
                }
            }
        }
        bool flag = _chk_route_list(target_n_l, data, backend);
        if (!flag) return false;

        double new_cost = 0.0; // Calculate cost below if feasible
        // Manual CPU cost calculation fallback
        double distance = 0.0;
        for (size_t idx = 1; idx < target_n_l.size(); ++idx) {
            distance += data.dist[target_n_l[idx - 1]][target_n_l[idx]];
        }
        new_cost = route_total_cost_from_distance(data, distance, true);

        if (r_indice.size() == 2) {
            std::vector<int> target_n_l_2;
            for (int i = 0; i < m.len_2; ++i) {
                const Seq& seq = m.seqList_2[i];
                if (seq.r_index == -1) {
                    target_n_l_2.push_back(data.DC);
                    continue;
                }
                const std::vector<int>& source_n_l = s.get(seq.r_index).node_list;
                if (seq.start_point <= seq.end_point) {
                    for (int index = seq.start_point; index <= seq.end_point; ++index) {
                        target_n_l_2.push_back(source_n_l[index]);
                    }
                } else {
                    for (int index = seq.start_point; index >= seq.end_point; --index) {
                        target_n_l_2.push_back(source_n_l[index]);
                    }
                }
            }
            if (r_indice[1] != -1) {
                ori_cost += route_total_cost(s.get(r_indice[1]), data);
            }
            bool flag2 = _chk_route_list(target_n_l_2, data, backend);
            if (!flag2) return false;

            double distance2 = 0.0;
            for (size_t idx = 1; idx < target_n_l_2.size(); ++idx) {
                distance2 += data.dist[target_n_l_2[idx - 1]][target_n_l_2[idx]];
            }
            new_cost += route_total_cost_from_distance(data, distance2, true);
        }

        m.delta_cost = new_cost - ori_cost;
        return true;
    }

    Attr tmp_attr_1;
    if (!eval_route(s, m.seqList_1, m.len_1, tmp_attr_1, data)) {
        return false;
    }
    double new_cost = 0.0;
    if (tmp_attr_1.num_cus != 0) {
        new_cost += route_total_cost_from_distance(data, tmp_attr_1.dist, tmp_attr_1.num_cus != 0);
    }
    if (r_indice.size() == 2) {
        Attr tmp_attr_2;
        if (!eval_route(s, m.seqList_2, m.len_2, tmp_attr_2, data)) {
            return false;
        }
        if (r_indice[1] != -1) {
            ori_cost += route_total_cost(s.get(r_indice[1]), data);
        }
        if (tmp_attr_2.num_cus != 0) {
            new_cost += route_total_cost_from_distance(data, tmp_attr_2.dist, tmp_attr_2.num_cus != 0);
        }
    }
    m.delta_cost = new_cost - ori_cost;
    return true;
}

static bool _evaluate_move_batch(
    const Solution& s,
    const Data& data,
    BaseComputeBackend* backend,
    const std::vector<Move>& candidates,
    Move& best_move
);

void apply_move(Solution& s, const Move& m, const Data& data) {
    std::vector<int> r_indice;
    r_indice.push_back(m.r_indice[0]);
    if (m.r_indice[1] != -2) {
        r_indice.push_back(m.r_indice[1]);
    }

    Route& r = s.get(r_indice[0]);
    std::vector<int> target_n_l;

    for (int i = 0; i < m.len_1; ++i) {
        const Seq& seq = m.seqList_1[i];
        const std::vector<int>& source_n_l = s.get(seq.r_index).node_list;
        if (seq.start_point <= seq.end_point) {
            for (int index = seq.start_point; index <= seq.end_point; ++index) {
                target_n_l.push_back(source_n_l[index]);
            }
        } else {
            for (int index = seq.start_point; index >= seq.end_point; --index) {
                target_n_l.push_back(source_n_l[index]);
            }
        }
    }

    if (r_indice.size() == 2) {
        std::vector<int> target_n_l_2;
        for (int i = 0; i < m.len_2; ++i) {
            const Seq& seq = m.seqList_2[i];
            if (seq.r_index == -1) {
                target_n_l_2.push_back(data.DC);
                continue;
            }
            const std::vector<int>& source_n_l = s.get(seq.r_index).node_list;
            if (seq.start_point <= seq.end_point) {
                for (int index = seq.start_point; index <= seq.end_point; ++index) {
                    target_n_l_2.push_back(source_n_l[index]);
                }
            } else {
                for (int index = seq.start_point; index >= seq.end_point; --index) {
                    target_n_l_2.push_back(source_n_l[index]);
                }
            }
        }

        if (r_indice[1] == -1) {
            Route r_new(data);
            r_new.node_list = target_n_l_2;
            r_new.update(data);
            s.append(r_new);
            r_indice[1] = s.len() - 1;
        } else {
            Route& r_2 = s.get(r_indice[1]);
            r_2.node_list = target_n_l_2;
            r_2.update(data);
        }
    }

    r.node_list = target_n_l;
    r.update(data);
    s.local_update(r_indice);
    s.cal_cost(data);
}

// Local Search operators templates definitions
static void two_opt_opt(int r1, int r2, Solution& s, const Data& data, Move& m, BaseComputeBackend* backend) {
    m.delta_cost = std::numeric_limits<double>::infinity();
    Route& r = s.get(r1);
    const std::vector<int>& n_l = r.node_list;
    int length = n_l.size();
    if (length < 4) return;

    if (data.architecture == "hybrid_v2") {
        std::vector<Move> candidates;
        candidates.reserve(static_cast<size_t>(length));
        for (int start = 1; start < length - 2; ++start) {
            if (r.gat(start + 1, start).num_cus == INFEASIBLE_VAL) {
                continue;
            }
            Move candidate;
            candidate.r_indice[0] = r1;
            candidate.r_indice[1] = -2;
            candidate.len_1 = 3;
            candidate.seqList_1[0] = Seq(r1, 0, start - 1);
            candidate.seqList_1[1] = Seq(r1, start + 1, start);
            candidate.seqList_1[2] = Seq(r1, start + 2, length - 1);
            candidates.push_back(candidate);
        }
        _evaluate_move_batch(s, data, backend, candidates, m);
        return;
    }

    for (int start = 1; start < length - 2; ++start) {
        if (r.gat(start + 1, start).num_cus == INFEASIBLE_VAL) {
            continue;
        }
        TMP_MOVE.r_indice[0] = r1;
        TMP_MOVE.r_indice[1] = -2;
        TMP_MOVE.len_1 = 3;
        TMP_MOVE.seqList_1[0] = Seq(r1, 0, start - 1);
        TMP_MOVE.seqList_1[1] = Seq(r1, start + 1, start);
        TMP_MOVE.seqList_1[2] = Seq(r1, start + 2, length - 1);
        TMP_MOVE.len_2 = 0;
        if (eval_move(s, TMP_MOVE, data, backend) && TMP_MOVE.delta_cost < m.delta_cost) {
            m.copy_from(TMP_MOVE);
        }
    }
}

static void two_opt_star_opt(int r1, int r2, Solution& s, const Data& data, Move& m, BaseComputeBackend* backend) {
    m.delta_cost = std::numeric_limits<double>::infinity();
    const std::vector<int>& n_l_1 = s.get(r1).node_list;
    int len_1 = n_l_1.size();
    const std::vector<int>& n_l_2 = s.get(r2).node_list;
    int len_2 = n_l_2.size();

    if (data.architecture == "hybrid_v2") {
        std::vector<Move> candidates;
        candidates.reserve(static_cast<size_t>(len_1) * static_cast<size_t>(len_2));
        for (int pos_1 = 1; pos_1 < len_1; ++pos_1) {
            for (int pos_2 = 1; pos_2 < len_2; ++pos_2) {
                if ((pos_1 == 1 && pos_2 == 1) ||
                    (pos_1 == len_1 - 1 && pos_2 == len_2 - 1)) {
                    continue;
                }
                Move candidate;
                candidate.r_indice[0] = r1;
                candidate.r_indice[1] = r2;
                candidate.len_1 = 2;
                candidate.seqList_1[0] = Seq(r1, 0, pos_1 - 1);
                candidate.seqList_1[1] = Seq(r2, pos_2, len_2 - 1);
                candidate.len_2 = 2;
                candidate.seqList_2[0] = Seq(r2, 0, pos_2 - 1);
                candidate.seqList_2[1] = Seq(r1, pos_1, len_1 - 1);
                candidates.push_back(candidate);
            }
        }
        _evaluate_move_batch(s, data, backend, candidates, m);
        return;
    }

    for (int pos_1 = 1; pos_1 < len_1; ++pos_1) {
        for (int pos_2 = 1; pos_2 < len_2; ++pos_2) {
            if ((pos_1 == 1 && pos_2 == 1) || (pos_1 == len_1 - 1 && pos_2 == len_2 - 1)) {
                continue;
            }
            TMP_MOVE.r_indice[0] = r1;
            TMP_MOVE.r_indice[1] = r2;
            TMP_MOVE.len_1 = 2;
            TMP_MOVE.seqList_1[0] = Seq(r1, 0, pos_1 - 1);
            TMP_MOVE.seqList_1[1] = Seq(r2, pos_2, len_2 - 1);
            TMP_MOVE.len_2 = 2;
            TMP_MOVE.seqList_2[0] = Seq(r2, 0, pos_2 - 1);
            TMP_MOVE.seqList_2[1] = Seq(r1, pos_1, len_1 - 1);
            if (eval_move(s, TMP_MOVE, data, backend) && TMP_MOVE.delta_cost < m.delta_cost) {
                m.copy_from(TMP_MOVE);
            }
        }
    }
}

struct LocalSearchRouteCandidate {
    Move move;
    std::vector<int> route_nodes;
    int eval_index = -1;
    bool use_batch_eval = false;
};

static void _append_route_nodes_from_move(
    const Solution& s,
    const Move& move,
    std::vector<int>& out_nodes
) {
    out_nodes.clear();
    for (int i = 0; i < move.len_1; ++i) {
        const Seq& seq = move.seqList_1[i];
        const std::vector<int>& source_n_l = s.get(seq.r_index).node_list;
        if (seq.start_point <= seq.end_point) {
            for (int index = seq.start_point; index <= seq.end_point; ++index) {
                out_nodes.push_back(source_n_l[index]);
            }
        } else {
            for (int index = seq.start_point; index >= seq.end_point; --index) {
                out_nodes.push_back(source_n_l[index]);
            }
        }
    }
}

static void _append_or_opt_single_candidate(
    std::vector<LocalSearchRouteCandidate>& candidates,
    std::vector<std::vector<int>>& routes_to_eval,
    const Solution& s,
    const Move& move
) {
    LocalSearchRouteCandidate candidate;
    candidate.move.copy_from(move);
    _append_route_nodes_from_move(s, candidate.move, candidate.route_nodes);
    candidate.eval_index = static_cast<int>(routes_to_eval.size());
    candidate.use_batch_eval = true;
    routes_to_eval.push_back(candidate.route_nodes);
    candidates.push_back(std::move(candidate));
}

static void _append_sequence_nodes(
    const Solution& s,
    const Seq& seq,
    const Data& data,
    std::vector<int>& out_nodes
) {
    if (seq.r_index == -1) {
        out_nodes.push_back(seq.start_point);
        return;
    }

    const std::vector<int>& source = s.get(seq.r_index).node_list;
    if (seq.start_point <= seq.end_point) {
        for (int index = seq.start_point; index <= seq.end_point; ++index) {
            out_nodes.push_back(source[index]);
        }
    } else {
        for (int index = seq.start_point; index >= seq.end_point; --index) {
            out_nodes.push_back(source[index]);
        }
    }
}

static void _materialize_move_routes(
    const Solution& s,
    const Move& move,
    const Data& data,
    std::vector<std::vector<int>>& out_routes
) {
    out_routes.clear();
    std::vector<int> first;
    for (int i = 0; i < move.len_1; ++i) {
        _append_sequence_nodes(s, move.seqList_1[i], data, first);
    }
    out_routes.push_back(std::move(first));

    if (move.r_indice[1] == -2) {
        return;
    }

    std::vector<int> second;
    for (int i = 0; i < move.len_2; ++i) {
        _append_sequence_nodes(s, move.seqList_2[i], data, second);
    }
    out_routes.push_back(std::move(second));
}

static bool _evaluate_move_batch(
    const Solution& s,
    const Data& data,
    BaseComputeBackend* backend,
    const std::vector<Move>& candidates,
    Move& best_move
) {
    if (candidates.empty()) {
        return false;
    }

    if (data.execution_policy == "adaptive") {
        WorkShape shape;
        shape.context = EVALUATION_CONTEXT_LOCAL_SEARCH;
        for (const Move& candidate : candidates) {
            size_t candidate_nodes = 0;
            for (int i = 0; i < candidate.len_1; ++i) {
                const Seq& seq = candidate.seqList_1[i];
                candidate_nodes += static_cast<size_t>(std::abs(seq.end_point - seq.start_point) + 1);
            }
            shape.item_count++;
            shape.total_nodes += candidate_nodes;
            shape.max_length = std::max(shape.max_length, candidate_nodes);
            if (candidate.r_indice[1] != -2) {
                size_t second_nodes = 0;
                for (int i = 0; i < candidate.len_2; ++i) {
                    const Seq& seq = candidate.seqList_2[i];
                    second_nodes += static_cast<size_t>(
                        seq.r_index == -1 ? 1 : std::abs(seq.end_point - seq.start_point) + 1
                    );
                }
                shape.item_count++;
                shape.total_nodes += second_nodes;
                shape.max_length = std::max(shape.max_length, second_nodes);
            }
        }
        shape.transfer_bytes = shape.total_nodes * sizeof(int);
        const EvalTarget target = backend == nullptr
            ? EvalTarget::CpuIncremental
            : backend->choose_target(shape);
        if (target == EvalTarget::CpuIncremental) {
            profile_registry().dispatch_cpu_incremental.add(
                0, static_cast<long long>(candidates.size())
            );
            bool found = false;
            double best_delta = std::numeric_limits<double>::infinity();
            for (const Move& candidate : candidates) {
                Move evaluated;
                evaluated.copy_from(candidate);
                if (!eval_move(s, evaluated, data, nullptr)) {
                    continue;
                }
                if (evaluated.delta_cost < best_delta - 1e-9) {
                    best_delta = evaluated.delta_cost;
                    best_move.copy_from(evaluated);
                    found = true;
                }
            }
            return found;
        }
    }

    std::vector<std::vector<int>> routes_to_eval;
    std::vector<std::pair<size_t, size_t>> route_ranges;
    routes_to_eval.reserve(candidates.size() * 2);
    route_ranges.reserve(candidates.size());

    for (const Move& candidate : candidates) {
        size_t begin = routes_to_eval.size();
        std::vector<std::vector<int>> move_routes;
        _materialize_move_routes(s, candidate, data, move_routes);
        routes_to_eval.insert(
            routes_to_eval.end(),
            std::make_move_iterator(move_routes.begin()),
            std::make_move_iterator(move_routes.end())
        );
        route_ranges.emplace_back(begin, routes_to_eval.size());
    }

    std::vector<RouteEval> evaluations = evaluate_route_batch(routes_to_eval, data, backend);
    if (evaluations.size() != routes_to_eval.size()) {
        return false;
    }

    std::vector<double> candidate_deltas(
        candidates.size(), std::numeric_limits<double>::infinity()
    );
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto [begin, end] = route_ranges[i];
        bool feasible = true;
        double new_cost = 0.0;
        for (size_t route_index = begin; route_index < end; ++route_index) {
            if (!evaluations[route_index].feasible) {
                feasible = false;
                break;
            }
            new_cost += evaluations[route_index].cost;
        }
        if (!feasible) {
            continue;
        }

        const Move& candidate = candidates[i];
        double old_cost = route_total_cost(s.get(candidate.r_indice[0]), data);
        if (candidate.r_indice[1] >= 0) {
            old_cost += route_total_cost(s.get(candidate.r_indice[1]), data);
        }
        candidate_deltas[i] = new_cost - old_cost;
    }

    if (backend != nullptr && backend->is_gpu_backend()) {
        if (candidates.size() <= 1024) {
            bool found = false;
            double best_delta = std::numeric_limits<double>::infinity();
            for (const Move& candidate : candidates) {
                std::vector<std::vector<int>> move_routes;
                _materialize_move_routes(s, candidate, data, move_routes);
                bool feasible = true;
                double new_cost = 0.0;
                for (const auto& route : move_routes) {
                    const RouteEval evaluation = _evaluate_route_cpu(route, data);
                    if (!evaluation.feasible) {
                        feasible = false;
                        break;
                    }
                    new_cost += evaluation.cost;
                }
                if (!feasible) continue;
                double old_cost = route_total_cost(s.get(candidate.r_indice[0]), data);
                if (candidate.r_indice[1] >= 0) {
                    old_cost += route_total_cost(s.get(candidate.r_indice[1]), data);
                }
                const double delta = new_cost - old_cost;
                if (delta < best_delta - 1e-9) {
                    best_delta = delta;
                    best_move.copy_from(candidate);
                    best_move.delta_cost = delta;
                    found = true;
                }
            }
            return found;
        }
        std::vector<unsigned char> verified(candidates.size(), 0);
        while (true) {
            int best_idx = -1;
            double best_delta = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < candidate_deltas.size(); ++i) {
                if (!std::isfinite(candidate_deltas[i])) continue;
                if (best_idx == -1 || candidate_deltas[i] < best_delta - 1e-9) {
                    best_idx = static_cast<int>(i);
                    best_delta = candidate_deltas[i];
                }
            }
            if (best_idx == -1) return false;

            bool refined = false;
            for (size_t i = 0; i < candidates.size(); ++i) {
                if (verified[i] || !std::isfinite(candidate_deltas[i]) ||
                    std::abs(candidate_deltas[i] - best_delta) > INSERTION_TIE_EPSILON) {
                    continue;
                }
                Move cpu_move;
                cpu_move.copy_from(candidates[i]);
                verified[i] = 1;
                refined = true;
                if (!eval_move(s, cpu_move, data, nullptr)) {
                    candidate_deltas[i] = std::numeric_limits<double>::infinity();
                } else {
                    candidate_deltas[i] = cpu_move.delta_cost;
                }
            }
            if (refined) continue;

            best_move.copy_from(candidates[best_idx]);
            best_move.delta_cost = candidate_deltas[best_idx];
            return true;
        }
    }

    bool found = false;
    double best_delta = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidate_deltas[i] < best_delta - 1e-9) {
            best_delta = candidate_deltas[i];
            best_move.copy_from(candidates[i]);
            best_move.delta_cost = candidate_deltas[i];
            found = true;
        }
    }
    return found;
}

static void or_opt_single_opt(int r1, int r2, Solution& s, const Data& data, Move& m, BaseComputeBackend* backend) {
    m.delta_cost = std::numeric_limits<double>::infinity();
    Route& r = s.get(r1);
    const std::vector<int>& n_l = r.node_list;
    int length = n_l.size();

    if (data.execution_policy == "adaptive") {
        std::vector<Move> moves;
        for (int start = 1; start < length - 1; ++start) {
            for (int seq_len = 1; seq_len <= data.or_opt_len; ++seq_len) {
                const int end = start + seq_len - 1;
                if (end >= length - 1) continue;
                for (int pos = 1; pos < start; ++pos) {
                    Move candidate;
                    candidate.r_indice[0] = r1;
                    candidate.r_indice[1] = -2;
                    candidate.len_1 = 4;
                    candidate.seqList_1[0] = Seq(r1, 0, pos - 1);
                    candidate.seqList_1[1] = Seq(r1, start, end);
                    candidate.seqList_1[2] = Seq(r1, pos, start - 1);
                    candidate.seqList_1[3] = Seq(r1, end + 1, length - 1);
                    moves.push_back(candidate);
                }
                for (int pos = end + 2; pos < length; ++pos) {
                    Move candidate;
                    candidate.r_indice[0] = r1;
                    candidate.r_indice[1] = -2;
                    candidate.len_1 = 4;
                    candidate.seqList_1[0] = Seq(r1, 0, start - 1);
                    candidate.seqList_1[1] = Seq(r1, end + 1, pos - 1);
                    candidate.seqList_1[2] = Seq(r1, start, end);
                    candidate.seqList_1[3] = Seq(r1, pos, length - 1);
                    moves.push_back(candidate);
                }
                Move new_route;
                new_route.r_indice[0] = r1;
                new_route.r_indice[1] = -1;
                new_route.len_1 = 2;
                new_route.seqList_1[0] = Seq(r1, 0, start - 1);
                new_route.seqList_1[1] = Seq(r1, end + 1, length - 1);
                new_route.len_2 = 3;
                new_route.seqList_2[0] = Seq(-1, data.DC, data.DC);
                new_route.seqList_2[1] = Seq(r1, start, end);
                new_route.seqList_2[2] = Seq(-1, data.DC, data.DC);
                moves.push_back(new_route);
            }
        }
        _evaluate_move_batch(s, data, backend, moves, m);
        return;
    }

    std::vector<LocalSearchRouteCandidate> candidates;
    std::vector<std::vector<int>> routes_to_eval;
    const double ori_cost = route_total_cost(r, data);

    for (int start = 1; start < length - 1; ++start) {
        for (int seq_len = 1; seq_len <= data.or_opt_len; ++seq_len) {
            int end = start + seq_len - 1;
            if (end >= length - 1) continue;

            for (int pos = 1; pos < start; ++pos) {
                TMP_MOVE.r_indice[0] = r1;
                TMP_MOVE.r_indice[1] = -2;
                TMP_MOVE.len_1 = 4;
                TMP_MOVE.seqList_1[0] = Seq(r1, 0, pos - 1);
                TMP_MOVE.seqList_1[1] = Seq(r1, start, end);
                TMP_MOVE.seqList_1[2] = Seq(r1, pos, start - 1);
                TMP_MOVE.seqList_1[3] = Seq(r1, end + 1, length - 1);
                TMP_MOVE.len_2 = 0;
                _append_or_opt_single_candidate(candidates, routes_to_eval, s, TMP_MOVE);
            }
            for (int pos = end + 2; pos < length; ++pos) {
                TMP_MOVE.r_indice[0] = r1;
                TMP_MOVE.r_indice[1] = -2;
                TMP_MOVE.len_1 = 4;
                TMP_MOVE.seqList_1[0] = Seq(r1, 0, start - 1);
                TMP_MOVE.seqList_1[1] = Seq(r1, end + 1, pos - 1);
                TMP_MOVE.seqList_1[2] = Seq(r1, start, end);
                TMP_MOVE.seqList_1[3] = Seq(r1, pos, length - 1);
                TMP_MOVE.len_2 = 0;
                _append_or_opt_single_candidate(candidates, routes_to_eval, s, TMP_MOVE);
            }

            TMP_MOVE.r_indice[0] = r1;
            TMP_MOVE.r_indice[1] = -1;
            TMP_MOVE.len_1 = 2;
            TMP_MOVE.seqList_1[0] = Seq(r1, 0, start - 1);
            TMP_MOVE.seqList_1[1] = Seq(r1, end + 1, length - 1);
            TMP_MOVE.len_2 = 3;
            TMP_MOVE.seqList_2[0] = Seq(-1, data.DC, data.DC);
            TMP_MOVE.seqList_2[1] = Seq(r1, start, end);
            TMP_MOVE.seqList_2[2] = Seq(-1, data.DC, data.DC);
            LocalSearchRouteCandidate fallback_candidate;
            fallback_candidate.move.copy_from(TMP_MOVE);
            candidates.push_back(std::move(fallback_candidate));
        }
    }

    std::vector<RouteEval> batch_results;
    if (!routes_to_eval.empty()) {
        batch_results = evaluate_route_batch(routes_to_eval, data, backend);
    }

    for (LocalSearchRouteCandidate& candidate : candidates) {
        bool feasible = false;
        double new_cost = 0.0;
        if (candidate.use_batch_eval) {
            if (candidate.eval_index < 0 || candidate.eval_index >= static_cast<int>(batch_results.size())) {
                continue;
            }
            const RouteEval& eval = batch_results[candidate.eval_index];
            feasible = eval.feasible;
            new_cost = eval.cost;
            candidate.move.delta_cost = new_cost - ori_cost;
        } else {
            feasible = eval_move(s, candidate.move, data, backend);
        }

        if (feasible && candidate.move.delta_cost < m.delta_cost) {
            m.copy_from(candidate.move);
        }
    }
}

static void or_opt_double_opt(int r1, int r2, Solution& s, const Data& data, Move& m, BaseComputeBackend* backend) {
    m.delta_cost = std::numeric_limits<double>::infinity();

    if (data.architecture == "hybrid_v2") {
        std::vector<Move> candidates;
        for (int i = 0; i < 2; ++i) {
            int r1_idx = (i == 0) ? r1 : r2;
            int r2_idx = (i == 0) ? r2 : r1;
            if (r1_idx == r2_idx) continue;

            const std::vector<int>& n_l = s.get(r1_idx).node_list;
            const std::vector<int>& n_l_2 = s.get(r2_idx).node_list;
            int length = static_cast<int>(n_l.size());
            int len_2 = static_cast<int>(n_l_2.size());
            for (int start = 1; start < length - 1; ++start) {
                for (int seq_len = 1; seq_len <= data.or_opt_len; ++seq_len) {
                    int end = start + seq_len - 1;
                    if (end >= length - 1) continue;
                    for (int pos = 1; pos < len_2; ++pos) {
                        Move candidate;
                        candidate.r_indice[0] = r1_idx;
                        candidate.r_indice[1] = r2_idx;
                        candidate.len_1 = 2;
                        candidate.seqList_1[0] = Seq(r1_idx, 0, start - 1);
                        candidate.seqList_1[1] = Seq(r1_idx, end + 1, length - 1);
                        candidate.len_2 = 3;
                        candidate.seqList_2[0] = Seq(r2_idx, 0, pos - 1);
                        candidate.seqList_2[1] = Seq(r1_idx, start, end);
                        candidate.seqList_2[2] = Seq(r2_idx, pos, len_2 - 1);
                        candidates.push_back(candidate);
                    }
                }
            }
        }
        _evaluate_move_batch(s, data, backend, candidates, m);
        return;
    }

    for (int i = 0; i < 2; ++i) {
        int r1_idx = (i == 0) ? r1 : r2;
        int r2_idx = (i == 0) ? r2 : r1;
        if (r1_idx == r2_idx) continue;

        Route& r = s.get(r1_idx);
        const std::vector<int>& n_l = r.node_list;
        int length = n_l.size();

        Route& r_2 = s.get(r2_idx);
        const std::vector<int>& n_l_2 = r_2.node_list;
        int len_2 = n_l_2.size();

        for (int start = 1; start < length - 1; ++start) {
            for (int seq_len = 1; seq_len <= data.or_opt_len; ++seq_len) {
                int end = start + seq_len - 1;
                if (end >= length - 1) continue;

                for (int pos = 1; pos < len_2; ++pos) {
                    TMP_MOVE.r_indice[0] = r1_idx;
                    TMP_MOVE.r_indice[1] = r2_idx;
                    TMP_MOVE.len_1 = 2;
                    TMP_MOVE.seqList_1[0] = Seq(r1_idx, 0, start - 1);
                    TMP_MOVE.seqList_1[1] = Seq(r1_idx, end + 1, length - 1);
                    TMP_MOVE.len_2 = 3;
                    TMP_MOVE.seqList_2[0] = Seq(r2_idx, 0, pos - 1);
                    TMP_MOVE.seqList_2[1] = Seq(r1_idx, start, end);
                    TMP_MOVE.seqList_2[2] = Seq(r2_idx, pos, len_2 - 1);
                    if (eval_move(s, TMP_MOVE, data, backend) && TMP_MOVE.delta_cost < m.delta_cost) {
                        m.copy_from(TMP_MOVE);
                    }
                }
            }
        }
    }
}

static void two_exchange_opt(int r1, int r2, Solution& s, const Data& data, Move& m, BaseComputeBackend* backend) {
    m.delta_cost = std::numeric_limits<double>::infinity();

    if (data.architecture == "hybrid_v2") {
        std::vector<Move> candidates;
        const std::vector<int>& n_l_1 = s.get(r1).node_list;
        const std::vector<int>& n_l_2 = s.get(r2).node_list;
        int len_1 = static_cast<int>(n_l_1.size());
        int len_2 = static_cast<int>(n_l_2.size());
        for (int start_1 = 1; start_1 < len_1 - 1; ++start_1) {
            for (int seq_len_1 = 1; seq_len_1 <= data.ex_len; ++seq_len_1) {
                int end_1 = start_1 + seq_len_1 - 1;
                if (end_1 >= len_1 - 1) continue;
                for (int start_2 = 1; start_2 < len_2 - 1; ++start_2) {
                    for (int seq_len_2 = 1; seq_len_2 <= data.ex_len; ++seq_len_2) {
                        int end_2 = start_2 + seq_len_2 - 1;
                        if (end_2 >= len_2 - 1) continue;
                        Move candidate;
                        candidate.r_indice[0] = r1;
                        candidate.r_indice[1] = r2;
                        candidate.len_1 = 3;
                        candidate.seqList_1[0] = Seq(r1, 0, start_1 - 1);
                        candidate.seqList_1[1] = Seq(r2, start_2, end_2);
                        candidate.seqList_1[2] = Seq(r1, end_1 + 1, len_1 - 1);
                        candidate.len_2 = 3;
                        candidate.seqList_2[0] = Seq(r2, 0, start_2 - 1);
                        candidate.seqList_2[1] = Seq(r1, start_1, end_1);
                        candidate.seqList_2[2] = Seq(r2, end_2 + 1, len_2 - 1);
                        candidates.push_back(candidate);
                    }
                }
            }
        }
        _evaluate_move_batch(s, data, backend, candidates, m);
        return;
    }

    Route& r_1 = s.get(r1);
    const std::vector<int>& n_l_1 = r_1.node_list;
    int len_1 = n_l_1.size();

    Route& r_2 = s.get(r2);
    const std::vector<int>& n_l_2 = r_2.node_list;
    int len_2 = n_l_2.size();

    for (int start_1 = 1; start_1 < len_1 - 1; ++start_1) {
        for (int seq_len_1 = 1; seq_len_1 <= data.ex_len; ++seq_len_1) {
            int end_1 = start_1 + seq_len_1 - 1;
            if (end_1 >= len_1 - 1) continue;

            for (int start_2 = 1; start_2 < len_2 - 1; ++start_2) {
                for (int seq_len_2 = 1; seq_len_2 <= data.ex_len; ++seq_len_2) {
                    int end_2 = start_2 + seq_len_2 - 1;
                    if (end_2 >= len_2 - 1) continue;

                    TMP_MOVE.r_indice[0] = r1;
                    TMP_MOVE.r_indice[1] = r2;
                    TMP_MOVE.len_1 = 3;
                    TMP_MOVE.seqList_1[0] = Seq(r1, 0, start_1 - 1);
                    TMP_MOVE.seqList_1[1] = Seq(r2, start_2, end_2);
                    TMP_MOVE.seqList_1[2] = Seq(r1, end_1 + 1, len_1 - 1);
                    TMP_MOVE.len_2 = 3;
                    TMP_MOVE.seqList_2[0] = Seq(r2, 0, start_2 - 1);
                    TMP_MOVE.seqList_2[1] = Seq(r1, start_1, end_1);
                    TMP_MOVE.seqList_2[2] = Seq(r2, end_2 + 1, len_2 - 1);
                    if (eval_move(s, TMP_MOVE, data, backend) && TMP_MOVE.delta_cost < m.delta_cost) {
                        m.copy_from(TMP_MOVE);
                    }
                }
            }
        }
    }
}

static void snippet(int r1, int r2, const std::string& opt, Solution& s, const Data& data, Move& target, BaseComputeBackend* backend) {
    Move candidate;
    if (opt == "2opt") two_opt_opt(r1, r2, s, data, candidate, backend);
    else if (opt == "2opt*") two_opt_star_opt(r1, r2, s, data, candidate, backend);
    else if (opt == "oropt_single") or_opt_single_opt(r1, r2, s, data, candidate, backend);
    else if (opt == "oropt_double") or_opt_double_opt(r1, r2, s, data, candidate, backend);
    else if (opt == "2exchange") two_exchange_opt(r1, r2, s, data, candidate, backend);
    else return;

    if (candidate.delta_cost - target.delta_cost < -0.001) {
        target.copy_from(candidate);
    }
}

class LocalCandidateBatch {
public:
    LocalCandidateBatch(
        const Solution& solution,
        const Data& problem,
        BaseComputeBackend* compute,
        Move& best
    ) : s(solution), data(problem), backend(compute), best_move(best) {
        candidates.reserve(kChunkSize);
    }

    void add(const Move& candidate) {
        candidates.push_back(candidate);
        if (candidates.size() >= kChunkSize) flush();
    }

    void flush() {
        if (candidates.empty()) return;
        Move chunk_best;
        chunk_best.delta_cost = std::numeric_limits<double>::infinity();
        if (_evaluate_move_batch(s, data, backend, candidates, chunk_best) &&
            chunk_best.delta_cost - best_move.delta_cost < -0.001) {
            best_move.copy_from(chunk_best);
        }
        candidates.clear();
    }

private:
    static constexpr size_t kChunkSize = 32768;
    const Solution& s;
    const Data& data;
    BaseComputeBackend* backend;
    Move& best_move;
    std::vector<Move> candidates;
};

static void collect_adaptive_operator_candidates(
    const std::string& opt,
    Solution& s,
    const Data& data,
    BaseComputeBackend* backend,
    Move& best_move
) {
    LocalCandidateBatch batch(s, data, backend, best_move);
    const int route_count = s.len();

    if (opt == "2opt") {
        for (int r = 0; r < route_count; ++r) {
            const int length = static_cast<int>(s.get(r).node_list.size());
            for (int start = 1; start < length - 2; ++start) {
                if (s.get(r).gat(start + 1, start).num_cus == INFEASIBLE_VAL) continue;
                Move candidate;
                candidate.r_indice[0] = r;
                candidate.r_indice[1] = -2;
                candidate.len_1 = 3;
                candidate.seqList_1[0] = Seq(r, 0, start - 1);
                candidate.seqList_1[1] = Seq(r, start + 1, start);
                candidate.seqList_1[2] = Seq(r, start + 2, length - 1);
                batch.add(candidate);
            }
        }
    } else if (opt == "oropt_single") {
        for (int r = 0; r < route_count; ++r) {
            const int length = static_cast<int>(s.get(r).node_list.size());
            for (int start = 1; start < length - 1; ++start) {
                for (int seq_len = 1; seq_len <= data.or_opt_len; ++seq_len) {
                    const int end = start + seq_len - 1;
                    if (end >= length - 1) continue;
                    for (int pos = 1; pos < start; ++pos) {
                        Move candidate;
                        candidate.r_indice[0] = r;
                        candidate.r_indice[1] = -2;
                        candidate.len_1 = 4;
                        candidate.seqList_1[0] = Seq(r, 0, pos - 1);
                        candidate.seqList_1[1] = Seq(r, start, end);
                        candidate.seqList_1[2] = Seq(r, pos, start - 1);
                        candidate.seqList_1[3] = Seq(r, end + 1, length - 1);
                        batch.add(candidate);
                    }
                    for (int pos = end + 2; pos < length; ++pos) {
                        Move candidate;
                        candidate.r_indice[0] = r;
                        candidate.r_indice[1] = -2;
                        candidate.len_1 = 4;
                        candidate.seqList_1[0] = Seq(r, 0, start - 1);
                        candidate.seqList_1[1] = Seq(r, end + 1, pos - 1);
                        candidate.seqList_1[2] = Seq(r, start, end);
                        candidate.seqList_1[3] = Seq(r, pos, length - 1);
                        batch.add(candidate);
                    }
                    Move new_route;
                    new_route.r_indice[0] = r;
                    new_route.r_indice[1] = -1;
                    new_route.len_1 = 2;
                    new_route.seqList_1[0] = Seq(r, 0, start - 1);
                    new_route.seqList_1[1] = Seq(r, end + 1, length - 1);
                    new_route.len_2 = 3;
                    new_route.seqList_2[0] = Seq(-1, data.DC, data.DC);
                    new_route.seqList_2[1] = Seq(r, start, end);
                    new_route.seqList_2[2] = Seq(-1, data.DC, data.DC);
                    batch.add(new_route);
                }
            }
        }
    } else {
        for (int r1 = 0; r1 < route_count; ++r1) {
            for (int r2 = r1 + 1; r2 < route_count; ++r2) {
                const int len1 = static_cast<int>(s.get(r1).node_list.size());
                const int len2 = static_cast<int>(s.get(r2).node_list.size());
                if (opt == "2opt*") {
                    for (int pos1 = 1; pos1 < len1; ++pos1) {
                        for (int pos2 = 1; pos2 < len2; ++pos2) {
                            if ((pos1 == 1 && pos2 == 1) ||
                                (pos1 == len1 - 1 && pos2 == len2 - 1)) continue;
                            Move candidate;
                            candidate.r_indice[0] = r1;
                            candidate.r_indice[1] = r2;
                            candidate.len_1 = 2;
                            candidate.seqList_1[0] = Seq(r1, 0, pos1 - 1);
                            candidate.seqList_1[1] = Seq(r2, pos2, len2 - 1);
                            candidate.len_2 = 2;
                            candidate.seqList_2[0] = Seq(r2, 0, pos2 - 1);
                            candidate.seqList_2[1] = Seq(r1, pos1, len1 - 1);
                            batch.add(candidate);
                        }
                    }
                } else if (opt == "oropt_double") {
                    for (int direction = 0; direction < 2; ++direction) {
                        const int source = direction == 0 ? r1 : r2;
                        const int target = direction == 0 ? r2 : r1;
                        const int source_len = direction == 0 ? len1 : len2;
                        const int target_len = direction == 0 ? len2 : len1;
                        for (int start = 1; start < source_len - 1; ++start) {
                            for (int seq_len = 1; seq_len <= data.or_opt_len; ++seq_len) {
                                const int end = start + seq_len - 1;
                                if (end >= source_len - 1) continue;
                                for (int pos = 1; pos < target_len; ++pos) {
                                    Move candidate;
                                    candidate.r_indice[0] = source;
                                    candidate.r_indice[1] = target;
                                    candidate.len_1 = 2;
                                    candidate.seqList_1[0] = Seq(source, 0, start - 1);
                                    candidate.seqList_1[1] = Seq(source, end + 1, source_len - 1);
                                    candidate.len_2 = 3;
                                    candidate.seqList_2[0] = Seq(target, 0, pos - 1);
                                    candidate.seqList_2[1] = Seq(source, start, end);
                                    candidate.seqList_2[2] = Seq(target, pos, target_len - 1);
                                    batch.add(candidate);
                                }
                            }
                        }
                    }
                } else if (opt == "2exchange") {
                    for (int start1 = 1; start1 < len1 - 1; ++start1) {
                        for (int sl1 = 1; sl1 <= data.ex_len; ++sl1) {
                            const int end1 = start1 + sl1 - 1;
                            if (end1 >= len1 - 1) continue;
                            for (int start2 = 1; start2 < len2 - 1; ++start2) {
                                for (int sl2 = 1; sl2 <= data.ex_len; ++sl2) {
                                    const int end2 = start2 + sl2 - 1;
                                    if (end2 >= len2 - 1) continue;
                                    Move candidate;
                                    candidate.r_indice[0] = r1;
                                    candidate.r_indice[1] = r2;
                                    candidate.len_1 = 3;
                                    candidate.seqList_1[0] = Seq(r1, 0, start1 - 1);
                                    candidate.seqList_1[1] = Seq(r2, start2, end2);
                                    candidate.seqList_1[2] = Seq(r1, end1 + 1, len1 - 1);
                                    candidate.len_2 = 3;
                                    candidate.seqList_2[0] = Seq(r2, 0, start2 - 1);
                                    candidate.seqList_2[1] = Seq(r1, start1, end1);
                                    candidate.seqList_2[2] = Seq(r2, end2 + 1, len2 - 1);
                                    batch.add(candidate);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    batch.flush();
}

static void refresh_local_move_list(
    Solution& s,
    const Data& data,
    BaseComputeBackend* backend,
    std::vector<Move>& move_list
) {
    const int length = s.len();
    for (size_t i = 0; i < move_list.size(); ++i) {
        move_list[i].delta_cost = std::numeric_limits<double>::infinity();
        const std::string& opt = data.small_opts[i];
        if (data.execution_policy == "adaptive") {
            collect_adaptive_operator_candidates(opt, s, data, backend, move_list[i]);
        } else if (opt == "2opt" || opt == "oropt_single") {
            for (int r = 0; r < length; ++r) snippet(r, -1, opt, s, data, move_list[i], backend);
        } else {
            for (int r1 = 0; r1 < length; ++r1) {
                for (int r2 = r1 + 1; r2 < length; ++r2) {
                    snippet(r1, r2, opt, s, data, move_list[i], backend);
                }
            }
        }
    }
}

void find_local_optima(Solution& s, const Data& data, BaseComputeBackend* backend) {
    if (data.skip_finding_lo) return;
    ScopedProfileTimer timer(data.profile, profile_registry().local_search);

    // Refresh cached route costs before the first LS pass.
    s.update(data);
    s.cal_cost(data);

    std::vector<Move> move_list(data.small_opts.size());
    int length = s.len();
    refresh_local_move_list(s, data, backend, move_list);

    std::vector<int> tour_id_array;
    while (true) {
        int best_index = -1;
        double min_delta_cost = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < move_list.size(); ++i) {
            if (move_list[i].delta_cost - min_delta_cost < -0.001) {
                best_index = i;
                min_delta_cost = move_list[i].delta_cost;
            }
        }
        if (min_delta_cost < -0.001) {
            std::printf("Applying LS move: opt=%s, delta=%.4f, current_len=%d\n", data.small_opts[best_index].c_str(), min_delta_cost, length);
            apply_move(s, move_list[best_index], data);
            length = s.len();
            refresh_local_move_list(s, data, backend, move_list);
        } else {
            break;
        }
    }
}

void do_local_search(Solution& s, const Data& data, BaseComputeBackend* backend) {
    if (data.small_opts.empty()) return;
    if (data.elo == -1) return;

    find_local_optima(s, data, backend);
    s.cal_cost(data);
    if (data.elo == 0) return;

    // Multi-operator local search with perturbation
    std::vector<Solution> s_vector(1, s.clone());
    int no_improve = 0;
    LegacyMt19937 temp_rng(data.seed);

    while (no_improve < data.elo) {
        s_vector[0] = s.clone();
        perturb(s_vector, data, backend, temp_rng);
        find_local_optima(s_vector[0], data, backend);
        s_vector[0].cal_cost(data);

        if (s_vector[0].cost - s.cost < -0.001) {
            s.copy_from(s_vector[0]);
            no_improve = 0;
        } else {
            no_improve++;
        }
    }
}

void perturb(std::vector<Solution>& s_vector, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    related_removal(s_vector[0], data, rng);
    regret_insertion(s_vector[0], data, backend, rng);
}

struct InsertionCandidateOption {
    int route_idx = -1;
    int insert_pos = 1;
    double incremental_cost = std::numeric_limits<double>::infinity();
    double route_cost_after = std::numeric_limits<double>::infinity();
};

static bool insertion_option_less(
    const InsertionCandidateOption& a,
    const InsertionCandidateOption& b
) {
    if (std::abs(a.incremental_cost - b.incremental_cost) > 1e-9) {
        return a.incremental_cost < b.incremental_cost;
    }
    if (a.route_idx != b.route_idx) {
        return a.route_idx < b.route_idx;
    }
    return a.insert_pos < b.insert_pos;
}

static bool evaluate_insertion_option_cpu(
    const Solution& s,
    const Data& data,
    int customer,
    InsertionCandidateOption& option
) {
    if (option.route_idx == -1) {
        std::vector<int> candidate{data.DC, customer, data.DC};
        RouteEval eval = _evaluate_route_cpu(candidate, data);
        if (!eval.feasible) {
            return false;
        }
        option.insert_pos = 1;
        option.route_cost_after = eval.cost;
        option.incremental_cost = eval.cost;
        return true;
    }
    if (option.route_idx < 0 || option.route_idx >= s.len()) {
        return false;
    }

    const Route& route = s.get(option.route_idx);
    if (option.insert_pos < 1 ||
        option.insert_pos >= static_cast<int>(route.node_list.size())) {
        return false;
    }
    std::vector<int> candidate = route.node_list;
    candidate.insert(candidate.begin() + option.insert_pos, customer);
    RouteEval eval = _evaluate_route_cpu(candidate, data);
    if (!eval.feasible) {
        return false;
    }
    option.route_cost_after = eval.cost;
    option.incremental_cost = eval.cost - route_total_cost(route, data);
    return true;
}

static std::vector<InsertionCandidateOption> cpu_ranked_insertion_prefix(
    const Solution& s,
    const Data& data,
    int customer,
    const std::vector<InsertionCandidateOption>& options,
    size_t minimum_count
) {
    std::vector<InsertionCandidateOption> verified;
    size_t cursor = 0;
    while (cursor < options.size() && verified.size() < minimum_count) {
        const double group_limit = options[cursor].incremental_cost + INSERTION_TIE_EPSILON;
        do {
            InsertionCandidateOption option = options[cursor++];
            if (evaluate_insertion_option_cpu(s, data, customer, option)) {
                verified.push_back(option);
            }
        } while (cursor < options.size() && options[cursor].incremental_cost <= group_limit);
        std::stable_sort(verified.begin(), verified.end(), insertion_option_less);
    }
    return verified;
}

static std::vector<int> _find_unrouted_nodes(const Solution& s, const Data& data) {
    std::vector<int> record(data.customer_num + 1, 0);
    for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
        const Route& r = s.get(r_idx);
        for (int node : r.node_list) {
            if (node >= 0 && node <= data.customer_num) {
                record[node] = 1;
            }
        }
    }

    std::vector<int> unrouted;
    unrouted.reserve(data.customer_num);
    for (int i = 1; i <= data.customer_num; ++i) {
        if (i != data.DC && record[i] == 0) {
            unrouted.push_back(i);
        }
    }
    return unrouted;
}

static bool _apply_insertion_option(Solution& s, const Data& data, int customer, InsertionCandidateOption option) {
    if (!evaluate_insertion_option_cpu(s, data, customer, option)) {
        return false;
    }
    if (option.route_idx >= 0) {
        Route& r = s.get(option.route_idx);
        int pos = option.insert_pos;
        r.node_list.insert(r.node_list.begin() + pos, customer);
        r.update(data);
    } else {
        Route r_new(data);
        r_new.node_list.insert(r_new.node_list.begin() + 1, customer);
        r_new.update(data);
        s.append(r_new);
    }
    s.cal_cost(data);
    return true;
}

static std::vector<std::vector<InsertionCandidateOption>> _build_insertion_options(
    const Solution& s,
    const Data& data,
    BaseComputeBackend* backend,
    const std::vector<int>& unrouted,
    int context
) {
    std::vector<std::vector<InsertionCandidateOption>> options(unrouted.size());
    if (unrouted.empty()) {
        return options;
    }

    std::unique_ptr<CpuComputeBackend> cpu_backend;
    BaseComputeBackend* eval_backend = backend;
    if (eval_backend == nullptr) {
        cpu_backend = std::make_unique<CpuComputeBackend>(data);
        eval_backend = cpu_backend.get();
    }

    for (size_t node_idx = 0; node_idx < unrouted.size(); ++node_idx) {
        std::vector<int> new_route{data.DC, unrouted[node_idx], data.DC};
        RouteEval eval = _evaluate_route_cpu(new_route, data);
        if (eval.feasible) {
            InsertionCandidateOption opt;
            opt.route_idx = -1;
            opt.insert_pos = 1;
            opt.incremental_cost = eval.cost;
            opt.route_cost_after = eval.cost;
            options[node_idx].push_back(opt);
        }
    }

    if (s.len() > 0) {
        std::vector<std::vector<int>> routes;
        routes.reserve(s.len());
        std::vector<double> old_route_costs;
        old_route_costs.reserve(s.len());
        for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
            routes.push_back(s.get(r_idx).node_list);
            old_route_costs.push_back(route_total_cost(s.get(r_idx), data));
        }

        struct RequestMeta {
            int node_idx;
            int route_idx;
            int pos;
        };
        std::vector<RequestMeta> meta;
        std::vector<InsertionRequest> requests;
        int request_id = 0;
        for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
            const std::vector<int>& route = routes[r_idx];
            if (route.size() < 2) {
                continue;
            }
            for (size_t node_idx = 0; node_idx < unrouted.size(); ++node_idx) {
                int customer = unrouted[node_idx];
                for (int pos = 1; pos < static_cast<int>(route.size()); ++pos) {
                    if (data.pruning && !data.pm.empty()) {
                        int prev = route[pos - 1];
                        int next = route[pos];
                        if (!data.pm[prev][customer] || !data.pm[customer][next]) {
                            continue;
                        }
                    }
                    InsertionRequest req;
                    req.request_id = request_id++;
                    req.route_id = r_idx;
                    req.customer_id = customer;
                    req.insert_position = pos;
                    req.source_context = context;
                    req.context.operator_type = context == INSERTION_CONTEXT_INITIALIZATION
                        ? CANDIDATE_OPERATOR_INITIALIZATION
                        : (context == INSERTION_CONTEXT_LOCAL_SEARCH
                            ? CANDIDATE_OPERATOR_RELOCATE
                            : CANDIDATE_OPERATOR_REPAIR);
                    req.context.local_order = req.request_id;
                    requests.push_back(req);
                    meta.push_back(RequestMeta{static_cast<int>(node_idx), r_idx, pos});
                }
            }
        }

        if (!requests.empty()) {
            std::vector<InsertionScore> scores = evaluate_insertion_requests(
                data, eval_backend, routes, requests
            );
            ScopedProfileTimer selection_timer(data.profile, profile_registry().insertion_selection, static_cast<long long>(scores.size()));
            ScopedProfileTimer context_selection_timer(
                data.profile,
                insertion_selection_counter(context),
                static_cast<long long>(scores.size())
            );
            for (size_t i = 0; i < scores.size() && i < meta.size(); ++i) {
                const InsertionScore& score = scores[i];
                if (!score.feasible || score.request_id != requests[i].request_id) {
                    continue;
                }
                const RequestMeta& m = meta[i];
                InsertionCandidateOption opt;
                opt.route_idx = m.route_idx;
                opt.insert_pos = m.pos;
                opt.route_cost_after = score.fitness_after;
                opt.incremental_cost = score.fitness_after - old_route_costs[m.route_idx];
                options[m.node_idx].push_back(opt);
            }
        }
    }

    for (auto& node_options : options) {
        std::stable_sort(node_options.begin(), node_options.end(), insertion_option_less);
    }
    return options;
}

void related_removal(Solution& s, const Data& data, LegacyMt19937& rng) {
    if (data.rm_argrank.empty()) {
        random_removal(s, data, rng);
        return;
    }

    std::vector<int> present;
    present.reserve(data.customer_num);
    std::vector<int> present_flag(data.customer_num + 1, 0);
    for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
        const Route& r = s.get(r_idx);
        for (int node : r.node_list) {
            if (_is_customer(node, data) && present_flag[node] == 0) {
                present.push_back(node);
                present_flag[node] = 1;
            }
        }
    }
    if (present.empty()) {
        return;
    }

    int total_remove = static_cast<int>(std::round(data.customer_num * randdouble(data.removal_lower, data.removal_upper, rng)));
    total_remove = std::max(1, std::min(total_remove, static_cast<int>(present.size())));

    std::vector<int> flag(data.customer_num + 1, 0);
    std::vector<int> selected;
    selected.reserve(total_remove);

    int seed_node = present[randint(0, static_cast<int>(present.size()) - 1, rng)];
    flag[seed_node] = 1;
    selected.push_back(seed_node);

    while (static_cast<int>(selected.size()) < total_remove) {
        int ref = selected[randint(0, static_cast<int>(selected.size()) - 1, rng)];
        int first = -1;
        int second = -1;
        for (int cand : data.rm_argrank[ref]) {
            if (cand <= 0 || cand > data.customer_num || flag[cand] != 0 || present_flag[cand] == 0) {
                continue;
            }
            if (first == -1) {
                first = cand;
            } else {
                second = cand;
                break;
            }
        }

        int chosen = first;
        if (first == -1) {
            for (int node : present) {
                if (flag[node] == 0) {
                    chosen = node;
                    break;
                }
            }
        } else if (second != -1) {
            double denom = data.rm[ref][first] + data.rm[ref][second];
            double first_weight = (denom > 0.0 && std::isfinite(denom)) ? data.rm[ref][second] / denom : 1.0;
            chosen = (randdouble(0.0, 1.0, rng) < first_weight) ? first : second;
        }

        if (chosen == -1) {
            break;
        }
        flag[chosen] = 1;
        selected.push_back(chosen);
    }

    for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
        Route& r = s.get(r_idx);
        std::vector<int> next_nl;
        next_nl.reserve(r.node_list.size());
        for (int node : r.node_list) {
            if (node == data.DC || flag[node] == 0) {
                next_nl.push_back(node);
            }
        }
        r.node_list = next_nl;
    }
    s.update(data);
    s.cal_cost(data);
}

void random_removal(Solution& s, const Data& data, LegacyMt19937& rng) {
    int total_remove = std::round(data.customer_num * randdouble(data.removal_lower, data.removal_upper, rng));
    std::vector<int> customers;
    for (int i = 1; i <= data.customer_num; ++i) customers.push_back(i);
    legacy_shuffle(customers.begin(), customers.end(), rng);

    std::vector<int> flag(data.customer_num + 1, 0);
    for (int i = 0; i < total_remove && i < customers.size(); ++i) {
        flag[customers[i]] = 1;
    }

    for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
        Route& r = s.get(r_idx);
        std::vector<int> next_nl;
        for (int node : r.node_list) {
            if (node == data.DC || flag[node] == 0) {
                next_nl.push_back(node);
            }
        }
        r.node_list = next_nl;
    }
    s.update(data);
}

void greedy_insertion(Solution& s, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    (void)rng;
    s.update(data);
    s.cal_cost(data);

    while (true) {
        std::vector<int> unrouted = _find_unrouted_nodes(s, data);
        if (unrouted.empty()) {
            break;
        }

        std::vector<std::vector<InsertionCandidateOption>> options =
            _build_insertion_options(s, data, backend, unrouted, INSERTION_CONTEXT_REPAIR);
        int best_node_idx = -1;
        InsertionCandidateOption best_option;

        for (size_t i = 0; i < options.size(); ++i) {
            std::vector<InsertionCandidateOption> ranked = cpu_ranked_insertion_prefix(
                s, data, unrouted[i], options[i], 1
            );
            if (ranked.empty()) {
                continue;
            }
            if (ranked[0].incremental_cost < best_option.incremental_cost) {
                best_option = ranked[0];
                best_node_idx = static_cast<int>(i);
            }
        }

        if (best_node_idx == -1) {
            break;
        }
        if (!_apply_insertion_option(s, data, unrouted[best_node_idx], best_option)) {
            break;
        }
    }

    s.update(data);
    s.cal_cost(data);
}

void regret_insertion(Solution& s, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    (void)rng;
    s.update(data);
    s.cal_cost(data);

    while (true) {
        std::vector<int> unrouted = _find_unrouted_nodes(s, data);
        if (unrouted.empty()) {
            break;
        }

        std::vector<std::vector<InsertionCandidateOption>> options =
            _build_insertion_options(s, data, backend, unrouted, INSERTION_CONTEXT_REPAIR);
        int best_node_idx = -1;
        InsertionCandidateOption best_option;
        double best_regret = -std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < options.size(); ++i) {
            std::vector<InsertionCandidateOption> ranked = cpu_ranked_insertion_prefix(
                s, data, unrouted[i], options[i], 2
            );
            if (ranked.empty()) {
                continue;
            }
            double first = ranked[0].incremental_cost;
            double second = ranked.size() > 1 ? ranked[1].incremental_cost : first;
            double regret = second - first;
            if (regret > best_regret + 1e-9 ||
                (std::abs(regret - best_regret) <= 1e-9 && first < best_option.incremental_cost)) {
                best_regret = regret;
                best_option = ranked[0];
                best_node_idx = static_cast<int>(i);
            }
        }

        if (best_node_idx == -1) {
            break;
        }
        if (!_apply_insertion_option(s, data, unrouted[best_node_idx], best_option)) {
            break;
        }
    }
    s.cal_cost(data);
}

static void regret_insertion_legacy_unused(Solution& s, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    // Gặp các nốt chưa được sắp tuyến, chèn lại bằng regret logic
    std::vector<int> record(data.customer_num + 1, 0);
    for (int i = 0; i < s.len(); ++i) {
        for (int node : s.get(i).node_list) {
            record[node] = 1;
        }
    }

    std::vector<int> unrouted;
    for (int i = 1; i <= data.customer_num; ++i) {
        if (record[i] == 0) unrouted.push_back(i);
    }

    // Regret heuristic
    for (int c : unrouted) {
        int best_r_idx = -1;
        int best_pos_idx = -1;
        double best_cost = std::numeric_limits<double>::infinity();

        _best_insertion_position(s, data, backend, c, best_r_idx, best_pos_idx, best_cost);

        if (best_r_idx != -1) {
            s.get(best_r_idx).node_list.insert(s.get(best_r_idx).node_list.begin() + best_pos_idx, c);
            s.get(best_r_idx).update(data);
        } else {
            Route r_new(data);
            r_new.node_list.insert(r_new.node_list.begin() + 1, c);
            r_new.update(data);
            s.append(r_new);
        }
    }
    s.cal_cost(data);
}

static double _construction_tc(
    const std::vector<int>& route_nodes,
    int inserted_node,
    int pos,
    double unrouted_delivery,
    double unrouted_pickup,
    const Data& data
) {
    struct ConstructionWorkspace {
        std::vector<int> nodes;
        std::vector<double> load;
        std::vector<double> cd;
        std::vector<double> cp;
        std::vector<double> rd;
        std::vector<double> rp;
    };
    thread_local ConstructionWorkspace workspace;

    std::vector<int>& tmp = workspace.nodes;
    tmp.assign(route_nodes.begin(), route_nodes.end());
    tmp.insert(tmp.begin() + pos, inserted_node);
    const int new_len = static_cast<int>(tmp.size());
    const double capacity = data.vehicle.capacity;

    double route_delivery = 0.0;
    double route_pickup = 0.0;
    for (int i = 1; i < new_len - 1; ++i) {
        route_delivery += data.node[tmp[i]].delivery;
        route_pickup += data.node[tmp[i]].pickup;
    }

    workspace.load.assign(new_len, 0.0);
    workspace.cd.assign(new_len, 0.0);
    workspace.cp.assign(new_len, 0.0);
    workspace.rd.assign(new_len, 0.0);
    workspace.rp.assign(new_len, 0.0);
    std::vector<double>& load = workspace.load;
    std::vector<double>& cd = workspace.cd;
    std::vector<double>& cp = workspace.cp;
    std::vector<double>& rd = workspace.rd;
    std::vector<double>& rp = workspace.rp;

    load[0] = route_delivery;
    for (int i = 1; i < new_len; ++i) {
        int node = tmp[i];
        load[i] = load[i - 1] - data.node[node].delivery + data.node[node].pickup;
        cd[i] = cd[i - 1] + data.dist[tmp[i - 1]][node];
    }

    cp[new_len - 1] = 0.0;
    for (int i = new_len - 1; i > 0; --i) {
        cp[i - 1] = cp[i] + data.dist[tmp[i - 1]][tmp[i]];
    }

    rd[0] = capacity - route_delivery;
    if (new_len >= 2) {
        rp[new_len - 2] = capacity - route_pickup;
    }
    for (int i = 1; i < new_len - 1; ++i) {
        rd[i] = std::min(rd[i - 1], capacity - load[i]);
        rp[new_len - 2 - i] = std::min(rp[new_len - 1 - i], capacity - load[new_len - 2 - i]);
    }

    double rdt_u = 0.0;
    double rdt_d = 0.0;
    double rpt_u = 0.0;
    double rpt_d = 0.0;
    for (int i = 0; i < new_len - 1; ++i) {
        rdt_u += rd[i] * cd[i + 1];
        rdt_d += cd[i + 1];
        rpt_u += rp[i] * cp[i];
        rpt_d += cp[i];
    }

    double rdt = rdt_d > 0.0 ? rdt_u / rdt_d : 0.0;
    double rpt = rpt_d > 0.0 ? rpt_u / rpt_d : 0.0;
    double delivery_term = data.all_delivery > 0.0 ? (unrouted_delivery / data.all_delivery) * (1.0 - rdt / capacity) : 0.0;
    double pickup_term = data.all_pickup > 0.0 ? (unrouted_pickup / data.all_pickup) * (1.0 - rpt / capacity) : 0.0;
    return delivery_term + pickup_term;
}

static double _construction_criterion(
    const Route& r,
    const Data& data,
    const ConstructionConfig& config,
    int node,
    int pos,
    double unrouted_delivery,
    double unrouted_pickup
) {
    const std::vector<int>& nl = r.node_list;
    int pre = nl[pos - 1];
    int suc = nl[pos];
    double td = data.dist[pre][node] + data.dist[node][suc] - data.dist[pre][suc];
    if (config.insertion_mode == "td") {
        return td;
    }

    double tc = _construction_tc(nl, node, pos, unrouted_delivery, unrouted_pickup, data);
    double rs = data.dist[data.DC][node] + data.dist[node][data.DC];
    return td + config.lambda * tc * (2.0 * data.max_dist - data.min_dist) - config.gamma * rs;
}

static bool _score_construction_route(
    const Route& r,
    const Data& data,
    const ConstructionConfig& config,
    BaseComputeBackend* backend,
    const std::vector<int>& unrouted,
    double unrouted_delivery,
    double unrouted_pickup,
    int& selected_unrouted_idx,
    int& selected_pos
) {
    selected_unrouted_idx = -1;
    selected_pos = -1;
    if (unrouted.empty() || r.node_list.size() < 2) {
        return false;
    }

    std::unique_ptr<CpuComputeBackend> cpu_backend;
    BaseComputeBackend* eval_backend = backend;
    if (eval_backend == nullptr) {
        cpu_backend = std::make_unique<CpuComputeBackend>(data);
        eval_backend = cpu_backend.get();
    }

    std::vector<std::vector<int>> routes{r.node_list};
    std::vector<InsertionRequest> requests;
    std::vector<std::pair<int, int>> meta;
    int request_id = 0;
    for (size_t node_idx = 0; node_idx < unrouted.size(); ++node_idx) {
        int customer = unrouted[node_idx];
        for (int pos = 1; pos < static_cast<int>(r.node_list.size()); ++pos) {
            if (data.pruning && !data.pm.empty()) {
                int prev = r.node_list[pos - 1];
                int next = r.node_list[pos];
                if (!data.pm[prev][customer] || !data.pm[customer][next]) {
                    continue;
                }
            }
            InsertionRequest req;
            req.request_id = request_id++;
            req.route_id = 0;
            req.customer_id = customer;
            req.insert_position = pos;
            req.source_context = INSERTION_CONTEXT_INITIALIZATION;
            req.context.operator_type = CANDIDATE_OPERATOR_INITIALIZATION;
            req.context.local_order = req.request_id;
            requests.push_back(req);
            meta.emplace_back(static_cast<int>(node_idx), pos);
        }
    }

    if (requests.empty()) {
        return false;
    }

    std::vector<InsertionScore> scores = evaluate_insertion_requests(
        data, eval_backend, routes, requests
    );
    ScopedProfileTimer selection_timer(data.profile, profile_registry().insertion_selection, static_cast<long long>(scores.size()));
    ScopedProfileTimer context_selection_timer(
        data.profile,
        profile_registry().insertion_selection_initialization,
        static_cast<long long>(scores.size())
    );

    std::vector<double> metrics(scores.size(), std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < scores.size() && i < meta.size(); ++i) {
        const InsertionScore& score = scores[i];
        if (!score.feasible || score.request_id != requests[i].request_id) {
            continue;
        }
        int node_idx = meta[i].first;
        int pos = meta[i].second;
        metrics[i] = _construction_criterion(
            r, data, config, unrouted[node_idx], pos, unrouted_delivery, unrouted_pickup
        );
    }

    int selected_idx = select_cpu_verified_insertion(
        routes, requests, scores, std::move(metrics), data, false, 1e-9
    );
    if (selected_idx >= 0) {
        selected_unrouted_idx = meta[selected_idx].first;
        selected_pos = meta[selected_idx].second;
    }

    return selected_unrouted_idx != -1;
}

struct BatchedConstructionState {
    int solution_id = -1;
    ConstructionConfig config;
    Solution base;
    Solution branch;
    Solution best;
    std::vector<int> sample_pool;
    std::vector<int> unrouted;
    std::unique_ptr<Route> route;
    double unrouted_delivery = 0.0;
    double unrouted_pickup = 0.0;
    double best_cost = std::numeric_limits<double>::infinity();
    int sample_count = 0;
    int sample_index = 0;
    int initial_node = -1;
    bool branch_active = false;
    bool route_active = false;
    bool complete = false;
};

struct BatchedConstructionRequestMeta {
    int state_index = -1;
    int unrouted_index = -1;
    int insert_position = -1;
};

static void begin_batched_construction_branch(
    BatchedConstructionState& state,
    const Data& data,
    LegacyMt19937& rng
) {
    const int remaining_samples = static_cast<int>(state.sample_pool.size()) - state.sample_index;
    if (remaining_samples <= 0) {
        state.complete = true;
        return;
    }
    const int selected = randint(0, remaining_samples - 1, rng);
    state.initial_node = state.sample_pool[selected];
    std::swap(
        state.sample_pool[selected],
        state.sample_pool[state.sample_pool.size() - 1 - state.sample_index]
    );

    state.branch = state.base.clone();
    std::vector<int> present(data.customer_num + 1, 0);
    for (int route_idx = 0; route_idx < state.branch.len(); ++route_idx) {
        for (int node : state.branch.get(route_idx).node_list) {
            if (node >= 0 && node <= data.customer_num) {
                present[node] = 1;
            }
        }
    }
    state.unrouted.clear();
    state.unrouted.reserve(data.customer_num);
    state.unrouted_delivery = data.all_delivery;
    state.unrouted_pickup = data.all_pickup;
    for (int customer = 1; customer <= data.customer_num; ++customer) {
        if (customer == data.DC) {
            continue;
        }
        if (present[customer] == 0) {
            state.unrouted.push_back(customer);
        } else {
            state.unrouted_delivery -= data.node[customer].delivery;
            state.unrouted_pickup -= data.node[customer].pickup;
        }
    }
    state.route.reset();
    state.route_active = false;
    state.branch_active = true;
}

static void finish_batched_construction_route(
    BatchedConstructionState& state,
    const Data& data
) {
    if (state.route != nullptr) {
        state.route->update(data);
        if (!state.route->isempty()) {
            state.branch.append(*state.route);
        }
    }
    state.route.reset();
    state.route_active = false;
}

static void finish_batched_construction_branch(
    BatchedConstructionState& state,
    const Data& data
) {
    state.branch.cal_cost(data);
    if (state.branch.cost < state.best_cost - 0.001) {
        state.best = state.branch.clone();
        state.best_cost = state.branch.cost;
    }
    state.sample_index++;
    state.branch_active = false;
    if (state.sample_index >= state.sample_count) {
        state.complete = true;
    }
}

static bool prepare_batched_construction_route(
    BatchedConstructionState& state,
    const Data& data,
    LegacyMt19937& rng
) {
    while (!state.complete) {
        if (!state.branch_active) {
            begin_batched_construction_branch(state, data, rng);
            if (state.complete) {
                return false;
            }
        }
        if (state.route_active) {
            if (!state.unrouted.empty()) {
                return true;
            }
            finish_batched_construction_route(state, data);
            continue;
        }
        if (state.unrouted.empty()) {
            finish_batched_construction_branch(state, data);
            continue;
        }

        state.route = std::make_unique<Route>(data);
        int selected = -1;
        if (static_cast<int>(state.unrouted.size()) == data.customer_num) {
            for (int i = 0; i < static_cast<int>(state.unrouted.size()); ++i) {
                if (state.unrouted[i] == state.initial_node) {
                    selected = i;
                    break;
                }
            }
        }
        if (selected == -1) {
            selected = randint(0, static_cast<int>(state.unrouted.size()) - 1, rng);
        }
        const int first_node = state.unrouted[selected];
        state.route->node_list.insert(state.route->node_list.begin() + 1, first_node);
        state.unrouted_delivery -= data.node[first_node].delivery;
        state.unrouted_pickup -= data.node[first_node].pickup;
        state.unrouted.erase(state.unrouted.begin() + selected);
        state.route_active = true;
    }
    return false;
}

void new_route_insertion_batched(
    std::vector<Solution>& solutions,
    const Data& data,
    const std::vector<ConstructionConfig>& configs,
    BaseComputeBackend* backend,
    std::vector<LegacyMt19937>& rngs
) {
    if (solutions.size() != configs.size() || solutions.size() != rngs.size()) {
        throw std::invalid_argument("Batched construction inputs must have equal sizes");
    }
    if (solutions.empty()) {
        return;
    }

    std::vector<BatchedConstructionState> states;
    states.reserve(solutions.size());
    for (size_t i = 0; i < solutions.size(); ++i) {
        states.emplace_back();
        BatchedConstructionState& state = states.back();
        state.solution_id = static_cast<int>(i);
        state.config = configs[i];
        state.base = solutions[i].clone();
        state.best = state.base.clone();
        state.sample_pool = _find_unrouted_nodes(state.base, data);
        if (state.sample_pool.empty()) {
            state.best_cost = state.base.cal_cost(data);
            state.complete = true;
            continue;
        }
        const int ksize = state.config.ksize > 0 ? state.config.ksize : data.customer_num;
        state.sample_count = std::min(ksize, static_cast<int>(state.sample_pool.size()));
    }

    while (true) {
        int next_request_id = 0;
        bool all_complete = true;
        std::vector<std::vector<int>> routes;
        std::vector<InsertionRequest> requests;
        std::vector<BatchedConstructionRequestMeta> meta;
        std::vector<std::pair<size_t, size_t>> state_ranges(states.size(), {0, 0});

        for (size_t state_idx = 0; state_idx < states.size(); ++state_idx) {
            BatchedConstructionState& state = states[state_idx];
            while (prepare_batched_construction_route(state, data, rngs[state_idx])) {
                all_complete = false;
                const int route_id = static_cast<int>(routes.size());
                const size_t request_begin = requests.size();
                int local_order = 0;
                for (size_t node_idx = 0; node_idx < state.unrouted.size(); ++node_idx) {
                    const int customer = state.unrouted[node_idx];
                    for (int pos = 1; pos < static_cast<int>(state.route->node_list.size()); ++pos) {
                        if (data.pruning && !data.pm.empty()) {
                            const int prev = state.route->node_list[pos - 1];
                            const int next = state.route->node_list[pos];
                            if (!data.pm[prev][customer] || !data.pm[customer][next]) {
                                continue;
                            }
                        }
                        InsertionRequest request;
                        request.request_id = next_request_id++;
                        request.route_id = route_id;
                        request.customer_id = customer;
                        request.insert_position = pos;
                        request.source_context = INSERTION_CONTEXT_INITIALIZATION;
                        request.context.solution_id = state.solution_id;
                        request.context.operator_type = CANDIDATE_OPERATOR_INITIALIZATION;
                        request.context.stage_id = state.sample_index;
                        request.context.local_order = local_order++;
                        request.context.route_version = static_cast<int>(state.route->node_list.size());
                        requests.push_back(request);
                        meta.push_back(BatchedConstructionRequestMeta{
                            static_cast<int>(state_idx), static_cast<int>(node_idx), pos
                        });
                    }
                }
                if (requests.size() == request_begin) {
                    finish_batched_construction_route(state, data);
                    continue;
                }
                routes.push_back(state.route->node_list);
                state_ranges[state_idx] = {request_begin, requests.size()};
                break;
            }
            if (!state.complete) {
                all_complete = false;
            }
        }

        if (all_complete) {
            break;
        }
        if (requests.empty()) {
            throw std::runtime_error("Batched construction made no progress");
        }

        std::vector<InsertionScore> scores = evaluate_insertion_requests(
            data, backend, routes, requests
        );
        ScopedProfileTimer selection_timer(
            data.profile,
            profile_registry().insertion_selection,
            static_cast<long long>(requests.size())
        );
        ScopedProfileTimer context_selection_timer(
            data.profile,
            profile_registry().insertion_selection_initialization,
            static_cast<long long>(requests.size())
        );

        for (size_t state_idx = 0; state_idx < states.size(); ++state_idx) {
            const size_t begin = state_ranges[state_idx].first;
            const size_t end = state_ranges[state_idx].second;
            if (begin == end) {
                continue;
            }
            BatchedConstructionState& state = states[state_idx];
            std::vector<std::vector<int>> local_routes{state.route->node_list};
            std::vector<InsertionRequest> local_requests;
            std::vector<InsertionScore> local_scores;
            std::vector<double> metrics;
            local_requests.reserve(end - begin);
            local_scores.reserve(end - begin);
            metrics.reserve(end - begin);
            for (size_t request_idx = begin; request_idx < end; ++request_idx) {
                InsertionRequest request = requests[request_idx];
                request.route_id = 0;
                local_requests.push_back(request);
                if (request_idx < scores.size()) {
                    local_scores.push_back(scores[request_idx]);
                } else {
                    InsertionScore missing;
                    missing.request_id = request.request_id;
                    local_scores.push_back(missing);
                }
                const BatchedConstructionRequestMeta& request_meta = meta[request_idx];
                metrics.push_back(_construction_criterion(
                    *state.route,
                    data,
                    state.config,
                    state.unrouted[request_meta.unrouted_index],
                    request_meta.insert_position,
                    state.unrouted_delivery,
                    state.unrouted_pickup
                ));
            }

            const int selected = select_cpu_verified_insertion(
                local_routes,
                local_requests,
                local_scores,
                std::move(metrics),
                data,
                false,
                1e-9
            );
            if (selected < 0) {
                finish_batched_construction_route(state, data);
                continue;
            }

            const BatchedConstructionRequestMeta& selected_meta = meta[begin + selected];
            const int customer = state.unrouted[selected_meta.unrouted_index];
            state.route->node_list.insert(
                state.route->node_list.begin() + selected_meta.insert_position,
                customer
            );
            state.unrouted_delivery -= data.node[customer].delivery;
            state.unrouted_pickup -= data.node[customer].pickup;
            state.unrouted.erase(state.unrouted.begin() + selected_meta.unrouted_index);
        }
    }

    for (size_t i = 0; i < solutions.size(); ++i) {
        solutions[i].copy_from(states[i].best);
    }
}

void new_route_insertion(
    Solution& s,
    const Data& data,
    const ConstructionConfig& config,
    BaseComputeBackend* backend,
    LegacyMt19937& rng,
    int initial_node
) {
    if (initial_node == -1) {
        std::vector<int> unrouted = _find_unrouted_nodes(s, data);
        if (unrouted.empty()) {
            return;
        }

        int ksize = config.ksize > 0 ? config.ksize : data.customer_num;
        if (ksize == 1) {
            int selected = randint(0, static_cast<int>(unrouted.size()) - 1, rng);
            new_route_insertion(s, data, config, backend, rng, unrouted[selected]);
            return;
        }

        Solution best_s = s.clone();
        double best_cost = std::numeric_limits<double>::infinity();
        int sample_count = std::min(ksize, static_cast<int>(unrouted.size()));
        for (int i = 0; i < sample_count; ++i) {
            int selected = randint(0, static_cast<int>(unrouted.size()) - 1 - i, rng);
            int node = unrouted[selected];
            Solution tmp_s = s.clone();
            new_route_insertion(tmp_s, data, config, backend, rng, node);
            if (tmp_s.cost < best_cost - 0.001) {
                best_s = tmp_s.clone();
                best_cost = tmp_s.cost;
            }
            std::swap(unrouted[selected], unrouted[unrouted.size() - 1 - i]);
        }
        s.copy_from(best_s);
        return;
    }

    std::vector<int> record(data.customer_num + 1, 0);
    for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
        const Route& r = s.get(r_idx);
        for (int node : r.node_list) {
            if (node >= 0 && node <= data.customer_num) {
                record[node] = 1;
            }
        }
    }

    std::vector<int> unrouted;
    unrouted.reserve(data.customer_num);
    double unrouted_delivery = data.all_delivery;
    double unrouted_pickup = data.all_pickup;
    for (int i = 1; i <= data.customer_num; ++i) {
        if (i == data.DC) {
            continue;
        }
        if (record[i] == 0) {
            unrouted.push_back(i);
        } else {
            unrouted_delivery -= data.node[i].delivery;
            unrouted_pickup -= data.node[i].pickup;
        }
    }

    while (!unrouted.empty()) {
        Route r(data);
        int selected = -1;
        if (static_cast<int>(unrouted.size()) == data.customer_num) {
            for (int i = 0; i < static_cast<int>(unrouted.size()); ++i) {
                if (unrouted[i] == initial_node) {
                    selected = i;
                    break;
                }
            }
        }
        if (selected == -1) {
            selected = randint(0, static_cast<int>(unrouted.size()) - 1, rng);
        }

        int first_node = unrouted[selected];
        r.node_list.insert(r.node_list.begin() + 1, first_node);
        unrouted_delivery -= data.node[first_node].delivery;
        unrouted_pickup -= data.node[first_node].pickup;
        unrouted.erase(unrouted.begin() + selected);

        int node_idx = -1;
        int pos = -1;
        while (_score_construction_route(
            r, data, config, backend, unrouted, unrouted_delivery, unrouted_pickup, node_idx, pos
        )) {
            int node = unrouted[node_idx];
            r.node_list.insert(r.node_list.begin() + pos, node);
            unrouted_delivery -= data.node[node].delivery;
            unrouted_pickup -= data.node[node].pickup;
            unrouted.erase(unrouted.begin() + node_idx);
        }

        r.update(data);
        if (!r.isempty()) {
            s.append(r);
        }
    }
    s.cal_cost(data);
}

void new_route_insertion(
    Solution& s,
    const Data& data,
    BaseComputeBackend* backend,
    LegacyMt19937& rng,
    int initial_node
) {
    ConstructionConfig config;
    config.ksize = data.ksize;
    config.insertion_mode = data.n_insert;
    config.lambda = data.lambda_gamma.first;
    config.gamma = data.lambda_gamma.second;
    new_route_insertion(s, data, config, backend, rng, initial_node);
}

static void new_route_insertion_legacy_unused(Solution& s, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng, int initial_node) {
    // Khởi tạo RCRS chèn dần khách hàng
    std::vector<int> record(data.customer_num + 1, 0);
    for (int i = 0; i < s.len(); ++i) {
        for (int node : s.get(i).node_list) {
            record[node] = 1;
        }
    }

    std::vector<int> unrouted;
    for (int i = 1; i <= data.customer_num; ++i) {
        if (i != data.DC && record[i] == 0) {
            unrouted.push_back(i);
        }
    }

    if (unrouted.empty()) return;

    // RCRS insertion loop
    while (!unrouted.empty()) {
        int c_idx = randint(0, unrouted.size() - 1, rng);
        int c = unrouted[c_idx];
        unrouted.erase(unrouted.begin() + c_idx);

        int best_r_idx = -1;
        int best_pos_idx = -1;
        double best_cost = std::numeric_limits<double>::infinity();

        _best_insertion_position(s, data, backend, c, best_r_idx, best_pos_idx, best_cost);

        if (best_r_idx != -1) {
            s.get(best_r_idx).node_list.insert(s.get(best_r_idx).node_list.begin() + best_pos_idx, c);
            s.get(best_r_idx).update(data);
        } else {
            Route r_new(data);
            r_new.node_list.insert(r_new.node_list.begin() + 1, c);
            r_new.update(data);
            s.append(r_new);
        }
    }
    s.cal_cost(data);
}

// ─────────────────────────────────────────────────────────────────────────────
// RCRS score helper
// ─────────────────────────────────────────────────────────────────────────────
// Returns a composite insertion score for placing customer `c` at position
// `pos` inside route `r` (0-indexed into node_list, 1-based insertion).
//
//   score = w_td * ΔTD  +  w_rc * RC_penalty  +  w_rs * RS_penalty
//
// ΔTD  : extra travel distance caused by the insertion.
// RC   : residual capacity penalty = max(0, C_H_new - capacity * rc_threshold)
//        This penalises insertions that exhaust route capacity too early,
//        preserving slack (buffer) for later customers (RCRS key idea).
// RS   : radial surcharge = angle deviation of the detour from the
//        depot–customer axis, discouraging route cross-overs.
//
// All three weights are tunable; defaults reflect the relative importance
// described in the research report.
// ─────────────────────────────────────────────────────────────────────────────
static double _rcrs_score(
    const Route& r,
    const Data&  data,
    int          c,          // customer to insert
    int          pos,        // insertion position (1 … |node_list|-1)
    double       w_td   = 1.0,
    double       w_rc   = 0.5,
    double       w_rs   = 0.3,
    double       rc_thr = 0.70   // fraction of capacity considered "safe"
) {
    const auto& nl    = r.node_list;
    int         prev  = nl[pos - 1];
    int         next  = nl[pos];

    // ── ΔTD ──────────────────────────────────────────────────────────────────
    double delta_td = std::max(0.0, data.dist[prev][c] + data.dist[c][next] - data.dist[prev][next]);

    // ── RC penalty ───────────────────────────────────────────────────────────
    // Route self attribute already reflects the current route WITHOUT c.
    // After inserting c, C_H (peak load) rises by at least the larger of
    // delivery and pickup of c.  We approximate conservatively.
    double C_H_new = r.self.C_H
                   + std::max(data.node[c].delivery, data.node[c].pickup);
    double capacity    = data.vehicle.capacity;
    double rc_penalty  = std::max(0.0, C_H_new - capacity * rc_thr);

    // ── RS (Radial Surcharge) ─────────────────────────────────────────────────
    // Angle between vector (DC → prev) and vector (prev → c).
    // A large deviation means the vehicle is going away from the depot axis.
    double dx_prev = data.dist[data.DC][prev];   // proxy: use distances
    double dx_c    = data.dist[data.DC][c];
    // Simple geometric proxy: |dist(DC, prev) + dist(prev, c) - dist(DC, c)|
    // penalises triangular detours that create crossing routes.
    double rs_penalty  = std::abs(dx_prev + data.dist[prev][c] - dx_c);

    return w_td * delta_td + w_rc * rc_penalty + w_rs * rs_penalty;
}

// ─────────────────────────────────────────────────────────────────────────────
// RCRS-GRASP hybrid initialisation  (Algorithm 2 replacement)
// ─────────────────────────────────────────────────────────────────────────────
// For each individual in the population:
//   1. Build an empty solution.
//   2. Iteratively pick the next customer to route using a GRASP RCL:
//      a. Compute the best feasible RCRS-score over ALL routes for every
//         remaining unrouted customer.
//      b. Build a Restricted Candidate List (RCL) containing all customers
//         whose best score ≤ score_min * (1 + alpha).
//      c. Choose one customer uniformly at random from the RCL.
//      d. Insert it at the best RCRS position for that customer.
//   3. Customers with no feasible insertion position open a new route.
//
// This guarantees:
//   • 100% feasibility (capacity + TW enforced at every step via _chk_route_list).
//   • Structural diversity: different RCL draws produce distinct topologies.
//   • No Repair Procedure (Algorithm 10) is ever needed.
//   • Thread-safe: each call operates on its own rng / solution copy.
//
// `alpha`  ∈ [0, 1] controls greediness:
//   alpha = 0  →  pure greedy (deterministic RCRS, same as new_route_insertion)
//   alpha = 1  →  full random (all unrouted customers in RCL)
// ─────────────────────────────────────────────────────────────────────────────
Solution rcrs_grasp_initialization(const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng, double alpha) {
    Solution s;

    // ── Customers to route ────────────────────────────────────────────────────
    std::vector<int> unrouted;
    for (int i = 1; i <= data.customer_num; ++i)
        if (i != data.DC) unrouted.push_back(i);

    // Shuffle to eliminate systematic ordering bias across individuals
    legacy_shuffle(unrouted.begin(), unrouted.end(), rng);

    while (!unrouted.empty()) {
        // ── Step A: find the best RCRS score for every unrouted customer ────
        struct CandInfo {
            int    customer;
            int    r_idx;
            int    pos;
            double score;
        };
        std::vector<CandInfo> best_per_customer;
        best_per_customer.reserve(unrouted.size());

        double global_best_score = std::numeric_limits<double>::infinity();

        for (int c : unrouted) {
            CandInfo best{ c, -1, -1, std::numeric_limits<double>::infinity() };

            for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
                const Route& r  = s.get(r_idx);
                const auto&  nl = r.node_list;
                for (int pos = 1; pos < (int)nl.size(); ++pos) {
                    std::vector<int> cand = nl;
                    cand.insert(cand.begin() + pos, c);
                    if (!_chk_route_list(cand, data, backend)) continue;

                    double score = _rcrs_score(r, data, c, pos);
                    if (score < best.score) {
                        best.score = score;
                        best.r_idx = r_idx;
                        best.pos   = pos;
                    }
                }
            }
            // If no existing route can absorb c, its "score" is 0 (forced new route)
            // We handle this case below; don't include in RCL to avoid bias.
            if (best.r_idx != -1 && best.score < global_best_score)
                global_best_score = best.score;

            best_per_customer.push_back(best);
        }

        // ── Step B: build RCL ─────────────────────────────────────────────────
        // Customers with no feasible insertion (r_idx == -1) must open a new
        // route; they are separated out and handled after the RCL draw.
        double threshold = global_best_score * (1.0 + alpha);
        std::vector<int> rcl_indices;          // indices into best_per_customer
        std::vector<int> forced_new_indices;   // must open a new route

        for (int i = 0; i < (int)best_per_customer.size(); ++i) {
            if (best_per_customer[i].r_idx == -1)
                forced_new_indices.push_back(i);
            else if (best_per_customer[i].score <= threshold + 1e-9)
                rcl_indices.push_back(i);
        }

        // If no RCL candidate (all need new routes), force one at a time
        if (rcl_indices.empty()) {
            if (forced_new_indices.empty()) break; // safety
            // Pick a random forced customer and open a route for it
            int pick_forced = randint(0, (int)forced_new_indices.size() - 1, rng);
            int ci = forced_new_indices[pick_forced];
            int c  = best_per_customer[ci].customer;
            Route r_new(data);
            r_new.node_list.insert(r_new.node_list.begin() + 1, c);
            r_new.update(data);
            s.append(r_new);
            // Remove from unrouted
            unrouted.erase(std::find(unrouted.begin(), unrouted.end(), c));
            continue;
        }

        // ── Step C: random draw from RCL ──────────────────────────────────────
        int pick = randint(0, (int)rcl_indices.size() - 1, rng);
        const CandInfo& chosen = best_per_customer[rcl_indices[pick]];

        // ── Step D: insert at best RCRS position for chosen customer ─────────
        s.get(chosen.r_idx).node_list.insert(
            s.get(chosen.r_idx).node_list.begin() + chosen.pos,
            chosen.customer);
        s.get(chosen.r_idx).update(data);

        // Remove from unrouted
        unrouted.erase(std::find(unrouted.begin(), unrouted.end(), chosen.customer));

        // ── Also flush customers that now MUST open a new route immediately ─
        // (Avoids them blocking future RCL draws when population of routes is small)
        // We defer this to the next iteration naturally.
    }

    s.cal_cost(data);
    return s;
}

Solution _sa_initialization(const Solution& s_0, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    Solution s = s_0.clone();
    s.update(data);
    s.cal_cost(data);

    Solution s_best = s.clone();
    double best_cost = s_best.cost;

    double t0 = 100.0;
    double alpha = 0.95;
    double tmin = 0.1;
    int itermax = 100;

    double t = t0;
    while (t > tmin) {
        for (int iter = 0; iter < itermax; ++iter) {
            int move_type = randint(1, 5, rng);

            if (move_type >= 1 && move_type <= 3) {
                if (s.len() == 0) continue;
                int r_idx = randint(0, s.len() - 1, rng);
                Route& route = s.get(r_idx);
                std::vector<int> nl = route.node_list;
                if (nl.size() < 4) continue;

                if (move_type == 1) { // Swap
                    int idx1 = randint(1, nl.size() - 2, rng);
                    int idx2 = randint(1, nl.size() - 2, rng);
                    while (idx1 == idx2) idx2 = randint(1, nl.size() - 2, rng);
                    std::swap(nl[idx1], nl[idx2]);
                } else if (move_type == 2) { // Insert
                    int idx1 = randint(1, nl.size() - 2, rng);
                    int node = nl[idx1];
                    nl.erase(nl.begin() + idx1);
                    int idx2 = randint(1, nl.size() - 1, rng);
                    nl.insert(nl.begin() + idx2, node);
                } else if (move_type == 3) { // Reverse
                    int idx1 = randint(1, nl.size() - 2, rng);
                    int idx2 = randint(1, nl.size() - 2, rng);
                    if (idx1 > idx2) std::swap(idx1, idx2);
                    std::reverse(nl.begin() + idx1, nl.begin() + idx2 + 1);
                }

                if (_chk_route_list(nl, data, backend)) {
                    double distance = 0.0;
                    for (size_t idx = 1; idx < nl.size(); ++idx) {
                        distance += data.dist[nl[idx - 1]][nl[idx]];
                    }
                    double r_cost = route_total_cost_from_distance(data, distance, true);
                    double old_r_cost = route.cal_cost(data);
                    double new_cost = s.cost - old_r_cost + r_cost;
                    double delta = new_cost - s.cost;

                    if (delta < 0.0 || randdouble(0.0, 1.0, rng) < std::exp(-delta / (1e-6 + t * std::abs(s.cost)))) {
                        route.node_list = nl;
                        route.update(data);
                        s.cal_cost(data);
                        if (s.cost < best_cost) {
                            best_cost = s.cost;
                            s_best = s.clone();
                        }
                    }
                }
            } else { // Move types 4 and 5
                if (s.len() < 2) continue;
                int r_idx1 = randint(0, s.len() - 1, rng);
                int r_idx2 = randint(0, s.len() - 1, rng);
                while (r_idx1 == r_idx2) r_idx2 = randint(0, s.len() - 1, rng);

                Route& route1 = s.get(r_idx1);
                Route& route2 = s.get(r_idx2);
                std::vector<int> nl1 = route1.node_list;
                std::vector<int> nl2 = route2.node_list;

                if (move_type == 4) { // Pd-Shift
                    if (nl1.size() < 3) continue;
                    int idx1 = randint(1, nl1.size() - 2, rng);
                    int node = nl1[idx1];
                    nl1.erase(nl1.begin() + idx1);
                    int idx2 = randint(1, nl2.size() - 1, rng);
                    nl2.insert(nl2.begin() + idx2, node);
                } else if (move_type == 5) { // Pd-Exchange
                    if (nl1.size() < 3 || nl2.size() < 3) continue;
                    int idx1 = randint(1, nl1.size() - 2, rng);
                    int idx2 = randint(1, nl2.size() - 2, rng);
                    std::swap(nl1[idx1], nl2[idx2]);
                }

                if (_chk_route_list(nl1, data, backend) && _chk_route_list(nl2, data, backend)) {
                    double dist1 = 0.0;
                    for (size_t idx = 1; idx < nl1.size(); ++idx) dist1 += data.dist[nl1[idx - 1]][nl1[idx]];
                    double r_cost1 = route_total_cost_from_distance(data, dist1, true);

                    double dist2 = 0.0;
                    for (size_t idx = 1; idx < nl2.size(); ++idx) dist2 += data.dist[nl2[idx - 1]][nl2[idx]];
                    double r_cost2 = route_total_cost_from_distance(data, dist2, true);

                    double old_r_cost1 = route1.cal_cost(data);
                    double old_r_cost2 = route2.cal_cost(data);
                    double new_cost = s.cost - (old_r_cost1 + old_r_cost2) + (r_cost1 + r_cost2);
                    double delta = new_cost - s.cost;

                    if (delta < 0.0 || randdouble(0.0, 1.0, rng) < std::exp(-delta / (1e-6 + t * std::abs(s.cost)))) {
                        route1.node_list = nl1;
                        route2.node_list = nl2;
                        s.update(data);
                        s.cal_cost(data);
                        if (s.cost < best_cost) {
                            best_cost = s.cost;
                            s_best = s.clone();
                        }
                    }
                }
            }
        }
        t = alpha * t;
    }

    return s_best;
}

static Route _make_route_with_customer(int customer, const Data& data) {
    Route r(data);
    r.node_list.insert(r.node_list.begin() + 1, customer);
    r.update(data);
    r.cal_cost(data);
    return r;
}

static bool _check_route_capacity_only(const std::vector<int>& nl, const Data& data) {
    if (nl.size() <= 2) {
        return true;
    }
    double load = 0.0;
    for (int node : nl) {
        if (node >= 0 && node <= data.customer_num) {
            load += data.node[node].delivery;
        }
    }
    if (load > data.vehicle.capacity) {
        return false;
    }
    for (size_t i = 1; i < nl.size(); ++i) {
        int node = nl[i];
        if (node < 0 || node > data.customer_num) {
            return false;
        }
        load = load - data.node[node].delivery + data.node[node].pickup;
        if (load < 0.0 || load > data.vehicle.capacity) {
            return false;
        }
    }
    return true;
}

static void _route_arrival_violations(
    const std::vector<int>& nl,
    const Data& data,
    std::vector<double>& arrival_times,
    std::vector<int>& violations
) {
    arrival_times.assign(nl.size(), 0.0);
    violations.clear();
    if (nl.size() <= 2) {
        return;
    }

    double time_val = data.start_time;
    arrival_times[0] = time_val;
    int prev = nl[0];
    for (size_t i = 1; i < nl.size(); ++i) {
        int node = nl[i];
        time_val += data.time[prev][node];
        arrival_times[i] = time_val;
        if (node != data.DC && time_val > data.node[node].end) {
            violations.push_back(node);
        }
        time_val = std::max(time_val, data.node[node].start) + data.node[node].s_time;
        prev = node;
    }
}

static bool _insert_customer_best_position(
    Solution& s,
    const Data& data,
    BaseComputeBackend* backend,
    int customer,
    int excluded_route = -1,
    bool minimize_arrival = false
) {
    std::unique_ptr<CpuComputeBackend> cpu_backend;
    BaseComputeBackend* eval_backend = backend;
    if (eval_backend == nullptr) {
        cpu_backend = std::make_unique<CpuComputeBackend>(data);
        eval_backend = cpu_backend.get();
    }

    std::vector<std::vector<int>> routes;
    std::vector<int> route_map;
    for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
        if (r_idx == excluded_route) {
            continue;
        }
        routes.push_back(s.get(r_idx).node_list);
        route_map.push_back(r_idx);
    }

    std::vector<InsertionRequest> requests;
    int request_id = 0;
    for (int local_r = 0; local_r < static_cast<int>(routes.size()); ++local_r) {
        const std::vector<int>& route = routes[local_r];
        for (int pos = 1; pos < static_cast<int>(route.size()); ++pos) {
            InsertionRequest req;
            req.request_id = request_id++;
            req.route_id = local_r;
            req.customer_id = customer;
            req.insert_position = pos;
            req.source_context = INSERTION_CONTEXT_REPAIR;
            req.context.operator_type = CANDIDATE_OPERATOR_REPAIR;
            req.context.local_order = req.request_id;
            requests.push_back(req);
        }
    }

    int best_route = -1;
    int best_pos = -1;
    if (!requests.empty()) {
        std::vector<InsertionScore> scores = evaluate_insertion_requests(
            data, eval_backend, routes, requests
        );
        ScopedProfileTimer selection_timer(data.profile, profile_registry().insertion_selection, static_cast<long long>(scores.size()));
        ScopedProfileTimer context_selection_timer(
            data.profile,
            insertion_selection_counter(INSERTION_CONTEXT_REPAIR),
            static_cast<long long>(scores.size())
        );
        std::vector<double> metrics(scores.size(), std::numeric_limits<double>::infinity());
        for (size_t i = 0; i < scores.size() && i < requests.size(); ++i) {
            if (!scores[i].feasible || scores[i].request_id != requests[i].request_id) {
                continue;
            }
            metrics[i] = scores[i].fitness_after;
            if (minimize_arrival) {
                std::vector<int> candidate = routes[requests[i].route_id];
                candidate.insert(candidate.begin() + requests[i].insert_position, customer);
                std::vector<double> arrivals;
                std::vector<int> violations;
                _route_arrival_violations(candidate, data, arrivals, violations);
                metrics[i] = requests[i].insert_position < static_cast<int>(arrivals.size())
                    ? arrivals[requests[i].insert_position]
                    : scores[i].fitness_after;
            }
        }
        int best_idx = select_cpu_verified_insertion(
            routes,
            requests,
            scores,
            std::move(metrics),
            data,
            !minimize_arrival,
            0.0
        );
        if (best_idx >= 0) {
            best_route = route_map[requests[best_idx].route_id];
            best_pos = requests[best_idx].insert_position;
        }
    }

    if (best_route != -1) {
        Route& r = s.get(best_route);
        r.node_list.insert(r.node_list.begin() + best_pos, customer);
        r.update(data);
        s.cal_cost(data);
        return true;
    }

    Route r_new = _make_route_with_customer(customer, data);
    if (_evaluate_route_cpu(r_new.node_list, data).feasible) {
        s.append(r_new);
        s.cal_cost(data);
        return true;
    }
    return false;
}

Solution feasible_or_repair_algorithm_10(const Solution& s, const Data& data, BaseComputeBackend* backend, LegacyMt19937& rng) {
    (void)rng;
    ScopedProfileTimer timer(data.profile, profile_registry().repair);
    Solution s_prime = s.clone();
    s_prime.update(data);
    s_prime.cal_cost(data);

    if (s_prime.check(data, false)) return s_prime;

    std::vector<std::vector<std::pair<int, int>>> occurrences(data.customer_num + 1);
    for (int r_idx = 0; r_idx < s_prime.len(); ++r_idx) {
        const Route& route = s_prime.get(r_idx);
        for (int p_idx = 0; p_idx < static_cast<int>(route.node_list.size()); ++p_idx) {
            int node = route.node_list[p_idx];
            if (_is_customer(node, data)) {
                occurrences[node].push_back({r_idx, p_idx});
            }
        }
    }

    for (int customer = 1; customer <= data.customer_num; ++customer) {
        const auto& occs = occurrences[customer];
        if (static_cast<int>(occs.size()) <= 1) {
            continue;
        }

        int keep_index = 0;
        double best_cost = std::numeric_limits<double>::infinity();
        for (int occ_idx = 0; occ_idx < static_cast<int>(occs.size()); ++occ_idx) {
            Solution candidate = s_prime.clone();
            for (int other_idx = 0; other_idx < static_cast<int>(occs.size()); ++other_idx) {
                if (other_idx == occ_idx) {
                    continue;
                }
                int r_i = occs[other_idx].first;
                int p_i = occs[other_idx].second;
                if (r_i < candidate.len() && p_i < static_cast<int>(candidate.get(r_i).node_list.size())) {
                    candidate.get(r_i).node_list[p_i] = -1;
                }
            }
            for (int r_idx = 0; r_idx < candidate.len(); ++r_idx) {
                Route& route = candidate.get(r_idx);
                route.node_list.erase(std::remove(route.node_list.begin(), route.node_list.end(), -1), route.node_list.end());
            }
            candidate.update(data);
            candidate.cal_cost(data);
            if (candidate.cost < best_cost) {
                best_cost = candidate.cost;
                keep_index = occ_idx;
            }
        }

        for (int other_idx = 0; other_idx < static_cast<int>(occs.size()); ++other_idx) {
            if (other_idx == keep_index) {
                continue;
            }
            int r_i = occs[other_idx].first;
            int p_i = occs[other_idx].second;
            if (r_i < s_prime.len() && p_i < static_cast<int>(s_prime.get(r_i).node_list.size())) {
                s_prime.get(r_i).node_list[p_i] = -1;
            }
        }
    }

    for (int r_idx = 0; r_idx < s_prime.len(); ++r_idx) {
        Route& route = s_prime.get(r_idx);
        route.node_list.erase(std::remove(route.node_list.begin(), route.node_list.end(), -1), route.node_list.end());
    }
    s_prime.update(data);
    s_prime.cal_cost(data);

    std::vector<int> visited(data.customer_num + 1, 0);
    for (int r_idx = 0; r_idx < s_prime.len(); ++r_idx) {
        const Route& route = s_prime.get(r_idx);
        for (int node : route.node_list) {
            if (_is_customer(node, data)) {
                visited[node] = 1;
            }
        }
    }

    for (int customer = 1; customer <= data.customer_num; ++customer) {
        if (customer != data.DC && visited[customer] == 0) {
            _insert_customer_best_position(s_prime, data, backend, customer);
        }
    }
    s_prime.update(data);
    s_prime.cal_cost(data);

    for (int r_idx = 0; r_idx < s_prime.len(); ++r_idx) {
        while (r_idx < s_prime.len() && !_check_route_capacity_only(s_prime.get(r_idx).node_list, data)) {
            Route& route = s_prime.get(r_idx);
            std::vector<int> route_customers;
            for (int node : route.node_list) {
                if (_is_customer(node, data)) {
                    route_customers.push_back(node);
                }
            }
            if (route_customers.empty()) {
                break;
            }
            int customer = route_customers[0];
            double worst_balance = -1.0;
            for (int node : route_customers) {
                double balance = std::abs(data.node[node].delivery - data.node[node].pickup);
                if (balance > worst_balance) {
                    worst_balance = balance;
                    customer = node;
                }
            }
            auto it = std::find(route.node_list.begin(), route.node_list.end(), customer);
            if (it == route.node_list.end()) {
                break;
            }
            route.node_list.erase(it);
            route.update(data);
            s_prime.cal_cost(data);
            _insert_customer_best_position(s_prime, data, backend, customer, r_idx);
            s_prime.update(data);
            s_prime.cal_cost(data);
        }
    }

    for (int r_idx = 0; r_idx < s_prime.len(); ++r_idx) {
        while (r_idx < s_prime.len()) {
            std::vector<double> arrivals;
            std::vector<int> violations;
            _route_arrival_violations(s_prime.get(r_idx).node_list, data, arrivals, violations);
            if (violations.empty()) {
                break;
            }
            int customer = violations[0];
            Route& route = s_prime.get(r_idx);
            auto it = std::find(route.node_list.begin(), route.node_list.end(), customer);
            if (it == route.node_list.end()) {
                break;
            }
            route.node_list.erase(it);
            route.update(data);
            s_prime.cal_cost(data);
            _insert_customer_best_position(s_prime, data, backend, customer, -1, true);
            s_prime.update(data);
            s_prime.cal_cost(data);
        }
    }

    for (int r_idx = 0; r_idx < s_prime.len(); ++r_idx) {
        Route& r = s_prime.get(r_idx);
        bool improved = true;
        while (improved) {
            improved = false;
            int length = r.node_list.size();
            if (length < 4) {
                break;
            }

            std::vector<std::vector<int>> candidates;
            std::vector<std::pair<int, int>> meta;
            candidates.reserve((length * (length - 3)) / 2);

            for (int i = 1; i < length - 2; ++i) {
                for (int j = i + 1; j < length - 1; ++j) {
                    std::vector<int> new_nl = r.node_list;
                    std::reverse(new_nl.begin() + i, new_nl.begin() + j + 1);
                    candidates.push_back(std::move(new_nl));
                    meta.emplace_back(i, j);
                }
            }

            if (candidates.empty()) {
                break;
            }

            double current_cost = route_total_cost(r, data);
            std::vector<RouteEval> evals = evaluate_route_batch(candidates, data, backend);

            int best_idx = -1;
            double best_cost = current_cost;
            for (size_t idx = 0; idx < evals.size(); ++idx) {
                if (!evals[idx].feasible) {
                    continue;
                }
                if (evals[idx].cost < best_cost) {
                    best_cost = evals[idx].cost;
                    best_idx = static_cast<int>(idx);
                }
            }

            if (best_idx != -1) {
                const int i = meta[best_idx].first;
                const int j = meta[best_idx].second;
                std::reverse(r.node_list.begin() + i, r.node_list.begin() + j + 1);
                r.update(data);
                r.cal_cost(data); // Update transcost to avoid stale cost in next iteration
                improved = true;
            }
        }
    }
    s_prime.update(data);
    s_prime.cal_cost(data);

    if (s_prime.check(data, false)) {
        return s_prime;
    }
    return s;
}
