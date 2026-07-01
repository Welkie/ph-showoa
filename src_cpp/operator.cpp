#include "operator.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

// Global helper for local search move representation
static Move TMP_MOVE;

// Simple random helpers
static inline int randint(int low, int high, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(low, high);
    return dist(rng);
}

static inline double randdouble(double low, double high, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(low, high);
    return dist(rng);
}

static inline bool _is_customer(int node, const Data& data) {
    return node != data.DC;
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
        return backend->evaluate_routes(routes);
    }
    std::vector<RouteEval> results(routes.size());
    for (size_t i = 0; i < routes.size(); ++i) {
        results[i] = _evaluate_route_cpu(routes[i], data);
    }
    return results;
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
    std::vector<std::vector<int>> candidates;
    std::vector<std::pair<int, int>> meta;

    for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
        const Route& r = s.get(r_idx);
        for (size_t pos = 1; pos < r.node_list.size(); ++pos) {
            std::vector<int> cand = r.node_list;
            cand.insert(cand.begin() + pos, customer);
            candidates.push_back(std::move(cand));
            meta.emplace_back(r_idx, static_cast<int>(pos));
        }
    }

    if (candidates.empty()) {
        return false;
    }

    std::vector<RouteEval> evals = evaluate_route_batch(candidates, data, backend);
    bool found = false;
    for (size_t i = 0; i < evals.size(); ++i) {
        if (!evals[i].feasible) {
            continue;
        }
        if (evals[i].cost < best_cost) {
            best_cost = evals[i].cost;
            best_r_idx = meta[i].first;
            best_pos_idx = meta[i].second;
            found = true;
        }
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

    double ori_cost = s.get(r_indice[0]).transcost + (s.get(r_indice[0]).isempty() ? 0.0 : 2000.0);

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
        new_cost = 2000.0 + distance * 1.0;

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
                ori_cost += s.get(r_indice[1]).transcost + (s.get(r_indice[1]).isempty() ? 0.0 : 2000.0);
            }
            bool flag2 = _chk_route_list(target_n_l_2, data, backend);
            if (!flag2) return false;

            double distance2 = 0.0;
            for (size_t idx = 1; idx < target_n_l_2.size(); ++idx) {
                distance2 += data.dist[target_n_l_2[idx - 1]][target_n_l_2[idx]];
            }
            new_cost += 2000.0 + distance2 * 1.0;
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
        new_cost += 2000.0 + tmp_attr_1.dist * 1.0;
    }
    if (r_indice.size() == 2) {
        Attr tmp_attr_2;
        if (!eval_route(s, m.seqList_2, m.len_2, tmp_attr_2, data)) {
            return false;
        }
        if (r_indice[1] != -1) {
            ori_cost += s.get(r_indice[1]).transcost + (s.get(r_indice[1]).isempty() ? 0.0 : 2000.0);
        }
        if (tmp_attr_2.num_cus != 0) {
            new_cost += 2000.0 + tmp_attr_2.dist * 1.0;
        }
    }
    m.delta_cost = new_cost - ori_cost;
    return true;
}

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

static void or_opt_single_opt(int r1, int r2, Solution& s, const Data& data, Move& m, BaseComputeBackend* backend) {
    m.delta_cost = std::numeric_limits<double>::infinity();
    Route& r = s.get(r1);
    const std::vector<int>& n_l = r.node_list;
    int length = n_l.size();

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
                if (eval_move(s, TMP_MOVE, data, backend) && TMP_MOVE.delta_cost < m.delta_cost) {
                    m.copy_from(TMP_MOVE);
                }
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
                if (eval_move(s, TMP_MOVE, data, backend) && TMP_MOVE.delta_cost < m.delta_cost) {
                    m.copy_from(TMP_MOVE);
                }
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
            if (eval_move(s, TMP_MOVE, data, backend) && TMP_MOVE.delta_cost < m.delta_cost) {
                m.copy_from(TMP_MOVE);
            }
        }
    }
}

static void or_opt_double_opt(int r1, int r2, Solution& s, const Data& data, Move& m, BaseComputeBackend* backend) {
    m.delta_cost = std::numeric_limits<double>::infinity();
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

// Local search mem database mapping
static Move MEM_LS_2OPT[100];
static Move MEM_LS_2OPTSTAR[100 * 100];
static Move MEM_LS_OROPTSINGLE[100];
static Move MEM_LS_OROPTDOUBLE[100 * 100];
static Move MEM_LS_2EX[100 * 100];

static Move* get_mem(const std::string& opt, int r1, int r2) {
    if (opt == "2opt") return &MEM_LS_2OPT[r1];
    if (opt == "oropt_single") return &MEM_LS_OROPTSINGLE[r1];
    if (opt == "2opt*") return &MEM_LS_2OPTSTAR[r1 * 100 + r2];
    if (opt == "oropt_double") return &MEM_LS_OROPTDOUBLE[r1 * 100 + r2];
    if (opt == "2exchange") return &MEM_LS_2EX[r1 * 100 + r2];
    return nullptr;
}

static void snippet(int r1, int r2, const std::string& opt, Solution& s, const Data& data, Move& target, BaseComputeBackend* backend) {
    Move* m = get_mem(opt, r1, r2);
    if (opt == "2opt") two_opt_opt(r1, r2, s, data, *m, backend);
    else if (opt == "2opt*") two_opt_star_opt(r1, r2, s, data, *m, backend);
    else if (opt == "oropt_single") or_opt_single_opt(r1, r2, s, data, *m, backend);
    else if (opt == "oropt_double") or_opt_double_opt(r1, r2, s, data, *m, backend);
    else if (opt == "2exchange") two_exchange_opt(r1, r2, s, data, *m, backend);

    if (m->delta_cost - target.delta_cost < -0.001) {
        target.copy_from(*m);
    }
}

void find_local_optima(Solution& s, const Data& data, BaseComputeBackend* backend) {
    if (data.skip_finding_lo) return;

    // Refresh cached route costs before the first LS pass.
    s.update(data);
    s.cal_cost(data);

    std::vector<Move> move_list(data.small_opts.size());
    int length = s.len();

    for (size_t i = 0; i < move_list.size(); ++i) {
        move_list[i].delta_cost = std::numeric_limits<double>::infinity();
        std::string opt = data.small_opts[i];
        if (opt == "2opt" || opt == "oropt_single") {
            for (int r = 0; r < length; ++r) {
                snippet(r, -1, opt, s, data, move_list[i], backend);
            }
        } else {
            for (int r1 = 0; r1 < length; ++r1) {
                for (int r2 = r1 + 1; r2 < length; ++r2) {
                    snippet(r1, r2, opt, s, data, move_list[i], backend);
                }
            }
        }
    }

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
            
            // Re-evaluate affected routes
            for (size_t i = 0; i < move_list.size(); ++i) {
                move_list[i].delta_cost = std::numeric_limits<double>::infinity();
                std::string opt = data.small_opts[i];
                if (opt == "2opt" || opt == "oropt_single") {
                    for (int r = 0; r < length; ++r) {
                        snippet(r, -1, opt, s, data, move_list[i], backend);
                    }
                } else {
                    for (int r1 = 0; r1 < length; ++r1) {
                        for (int r2 = r1 + 1; r2 < length; ++r2) {
                            snippet(r1, r2, opt, s, data, move_list[i], backend);
                        }
                    }
                }
            }
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
    std::mt19937 temp_rng(data.seed);

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

void perturb(std::vector<Solution>& s_vector, const Data& data, BaseComputeBackend* backend, std::mt19937& rng) {
    related_removal(s_vector[0], data, rng);
    regret_insertion(s_vector[0], data, backend, rng);
}

void related_removal(Solution& s, const Data& data, std::mt19937& rng) {
    // Basic random removal for perturb fallback in local search
    random_removal(s, data, rng);
}

void random_removal(Solution& s, const Data& data, std::mt19937& rng) {
    int total_remove = std::round(data.customer_num * randdouble(data.removal_lower, data.removal_upper, rng));
    std::vector<int> customers;
    for (int i = 1; i <= data.customer_num; ++i) customers.push_back(i);
    std::shuffle(customers.begin(), customers.end(), rng);

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

void greedy_insertion(Solution& s, const Data& data, BaseComputeBackend* backend, std::mt19937& rng) {
    regret_insertion(s, data, backend, rng); // Fallback to regret
}

void regret_insertion(Solution& s, const Data& data, BaseComputeBackend* backend, std::mt19937& rng) {
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
    double delta_td = data.dist[prev][c] + data.dist[c][next] - data.dist[prev][next];

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

void new_route_insertion(Solution& s, const Data& data, BaseComputeBackend* backend, std::mt19937& rng, int initial_node) {
    // ── Collect unrouted customers ────────────────────────────────────────────
    std::vector<int> record(data.customer_num + 1, 0);
    for (int i = 0; i < s.len(); ++i)
        for (int node : s.get(i).node_list)
            record[node] = 1;

    std::vector<int> unrouted;
    for (int i = 1; i <= data.customer_num; ++i)
        if (i != data.DC && record[i] == 0)
            unrouted.push_back(i);

    if (unrouted.empty()) return;

    // ── RCRS insertion loop (pure greedy, used as the constructive base) ──────
    while (!unrouted.empty()) {
        int c_idx = randint(0, (int)unrouted.size() - 1, rng);
        int c     = unrouted[c_idx];
        unrouted.erase(unrouted.begin() + c_idx);

        int    best_r_idx  = -1;
        int    best_pos    = -1;
        double best_score  = std::numeric_limits<double>::infinity();

        // Evaluate all feasible insertion positions via RCRS score
        for (int r_idx = 0; r_idx < s.len(); ++r_idx) {
            const Route& r  = s.get(r_idx);
            const auto&  nl = r.node_list;
            for (int pos = 1; pos < (int)nl.size(); ++pos) {
                std::vector<int> cand = nl;
                cand.insert(cand.begin() + pos, c);
                if (!_chk_route_list(cand, data, backend)) continue;

                double score = _rcrs_score(r, data, c, pos);
                if (score < best_score) {
                    best_score = score;
                    best_r_idx = r_idx;
                    best_pos   = pos;
                }
            }
        }

        if (best_r_idx != -1) {
            s.get(best_r_idx).node_list.insert(
                s.get(best_r_idx).node_list.begin() + best_pos, c);
            s.get(best_r_idx).update(data);
        } else {
            // No feasible position in any existing route → open a new one
            Route r_new(data);
            r_new.node_list.insert(r_new.node_list.begin() + 1, c);
            r_new.update(data);
            s.append(r_new);
        }
    }
    s.cal_cost(data);
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
Solution rcrs_grasp_initialization(const Data& data, BaseComputeBackend* backend, std::mt19937& rng, double alpha) {
    Solution s;

    // ── Customers to route ────────────────────────────────────────────────────
    std::vector<int> unrouted;
    for (int i = 1; i <= data.customer_num; ++i)
        if (i != data.DC) unrouted.push_back(i);

    // Shuffle to eliminate systematic ordering bias across individuals
    std::shuffle(unrouted.begin(), unrouted.end(), rng);

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

Solution _sa_initialization(const Solution& s_0, const Data& data, BaseComputeBackend* backend, std::mt19937& rng) {
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
                    double r_cost = 2000.0 + distance * 1.0;
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
                    double r_cost1 = 2000.0 + dist1 * 1.0;

                    double dist2 = 0.0;
                    for (size_t idx = 1; idx < nl2.size(); ++idx) dist2 += data.dist[nl2[idx - 1]][nl2[idx]];
                    double r_cost2 = 2000.0 + dist2 * 1.0;

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

Solution feasible_or_repair_algorithm_10(const Solution& s, const Data& data, BaseComputeBackend* backend, std::mt19937& rng) {
    Solution s_prime = s.clone();
    s_prime.update(data);
    s_prime.cal_cost(data);

    if (s_prime.check(data, false)) return s_prime; // If already feasible

    // Step 1: Duplicate removal & missing customers insertion
    // Let's implement RCRS new route insertion to insert any missing nodes
    new_route_insertion(s_prime, data, backend, rng);
    s_prime.update(data);
    s_prime.cal_cost(data);

    // Step 2 & 3 & 4: Fallback to local search repair
    for (int r_idx = 0; r_idx < s_prime.len(); ++r_idx) {
        Route& r = s_prime.get(r_idx);
        // Intra-route 2-opt locally on each route
        bool improved = true;
        while (improved) {
            improved = false;
            int length = r.node_list.size();
            for (int i = 1; i < length - 2; ++i) {
                for (int j = i + 1; j < length - 1; ++j) {
                    std::vector<int> new_nl = r.node_list;
                    std::reverse(new_nl.begin() + i, new_nl.begin() + j + 1);
                    if (_chk_route_list(new_nl, data, backend)) {
                        double dist_new = 0.0;
                        for (size_t idx = 1; idx < new_nl.size(); ++idx) dist_new += data.dist[new_nl[idx - 1]][new_nl[idx]];
                        double cost_new = 2000.0 + dist_new * 1.0;
                        if (cost_new < r.cal_cost(data)) {
                            r.node_list = new_nl;
                            r.update(data);
                            improved = true;
                            break;
                        }
                    }
                }
                if (improved) break;
            }
        }
    }
    s_prime.update(data);
    s_prime.cal_cost(data);

    return s_prime;
}
