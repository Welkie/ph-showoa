#pragma once

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <set>
#include "data.h"

// Define INFEASIBLE constant
#define INFEASIBLE_VAL -1

struct Attr {
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

    Attr copy() const {
        return *this;
    }
};

inline void attr_for_one_node(const Data& data, int node, Attr& a) {
    a.s = node;
    a.e = node;
    a.dist = 0.0;

    if (node == data.DC) {
        a.num_cus = 0;
        a.T_D = 0.0;
        a.T_E = data.start_time;
        a.T_L = data.end_time;
        a.C_E = 0.0;
        a.C_L = 0.0;
        a.C_H = 0.0;
    } else {
        a.num_cus = 1;
        a.T_D = data.node[node].s_time;
        a.T_E = data.node[node].start;
        a.T_L = data.node[node].end;
        a.C_E = data.node[node].delivery;
        a.C_L = data.node[node].pickup;
        a.C_H = std::max(a.C_E, a.C_L);
    }
}

inline void connect_into(const Attr& tmp_a, const Attr& tmp_b, Attr& merged_attr, double dist_ij, double t_ij) {
    merged_attr.num_cus = tmp_a.num_cus + tmp_b.num_cus;
    merged_attr.dist = tmp_a.dist + dist_ij + tmp_b.dist;

    double delta = tmp_a.T_D + t_ij;
    double delta_wt = std::max(tmp_b.T_E - delta - tmp_a.T_L, 0.0);
    merged_attr.T_D = tmp_a.T_D + tmp_b.T_D + t_ij + delta_wt;
    merged_attr.T_E = std::max(tmp_b.T_E - delta, tmp_a.T_E) - delta_wt;
    merged_attr.T_L = std::min(tmp_b.T_L - delta, tmp_a.T_L);

    merged_attr.C_E = tmp_a.C_E + tmp_b.C_E;
    merged_attr.C_H = std::max(tmp_a.C_H + tmp_b.C_E, tmp_a.C_L + tmp_b.C_H);
    merged_attr.C_L = tmp_a.C_L + tmp_b.C_L;

    merged_attr.s = tmp_a.s;
    merged_attr.e = tmp_b.e;
}

inline void connect_inplace(Attr& merged_attr, const Attr& tmp_b, double dist_ij, double t_ij) {
    merged_attr.num_cus = merged_attr.num_cus + tmp_b.num_cus;
    merged_attr.dist = merged_attr.dist + dist_ij + tmp_b.dist;

    double delta = merged_attr.T_D + t_ij;
    double delta_wt = std::max(tmp_b.T_E - delta - merged_attr.T_L, 0.0);
    merged_attr.T_D = merged_attr.T_D + tmp_b.T_D + t_ij + delta_wt;
    merged_attr.T_E = std::max(tmp_b.T_E - delta, merged_attr.T_E) - delta_wt;
    merged_attr.T_L = std::min(tmp_b.T_L - delta, merged_attr.T_L);

    double old_c_e = merged_attr.C_E;
    double old_c_h = merged_attr.C_H;
    double old_c_l = merged_attr.C_L;
    merged_attr.C_E = old_c_e + tmp_b.C_E;
    merged_attr.C_H = std::max(old_c_h + tmp_b.C_E, old_c_l + tmp_b.C_H);
    merged_attr.C_L = old_c_l + tmp_b.C_L;

    merged_attr.e = tmp_b.e;
}

inline bool equal_attr(const Attr& a, const Attr& b) {
    return (
        a.num_cus == b.num_cus
        && a.dist == b.dist
        && a.s == b.s
        && a.e == b.e
        && a.T_D == b.T_D
        && a.T_E == b.T_E
        && a.T_L == b.T_L
        && a.C_E == b.C_E
        && a.C_H == b.C_H
        && a.C_L == b.C_L
    );
}

class Route {
public:
    std::vector<int> node_list;
    double dep_time = 0.0;
    double ret_time = 0.0;
    double transcost = 0.0;
    std::vector<Attr> attr;
    Attr self;

    Route(const Data& data) {
        node_list.push_back(data.DC);
        node_list.push_back(data.DC);
        update(data);
    }

    Route clone() const {
        Route new_route = *this;
        return new_route;
    }

    Attr& gat(int i, int j) {
        int nl_len = node_list.size();
        return attr[i * nl_len + j];
    }

    const Attr& gat(int i, int j) const {
        int nl_len = node_list.size();
        return attr[i * nl_len + j];
    }

    void cal_attr(const Data& data) {
        int nl_len = node_list.size();
        int end_index = nl_len - 1;
        attr.assign(nl_len * nl_len, Attr());

        for (int i = 0; i <= end_index; ++i) {
            attr_for_one_node(data, node_list[i], gat(i, i));
        }

        for (int i = 0; i < end_index; ++i) {
            for (int j = i + 1; j <= end_index; ++j) {
                connect_into(
                    gat(i, j - 1),
                    gat(j, j),
                    gat(i, j),
                    data.dist[node_list[j - 1]][node_list[j]],
                    data.time[node_list[j - 1]][node_list[j]]
                );
            }
        }
        self = gat(0, end_index);

        for (int i = end_index - 1; i > 0; --i) {
            bool feasible = true;
            for (int j = i - 1; j > 0; --j) {
                if ((i - j + 1) > 2) {
                    break;
                }
                if (!feasible) {
                    gat(i, j).num_cus = INFEASIBLE_VAL;
                    continue;
                }
                if (gat(i, j + 1).T_E + gat(i, j + 1).T_D + data.time[node_list[j + 1]][node_list[j]] - gat(j, j).T_L > 0) {
                    gat(i, j).num_cus = INFEASIBLE_VAL;
                    feasible = false;
                } else {
                    connect_into(
                        gat(i, j + 1),
                        gat(j, j),
                        gat(i, j),
                        data.dist[node_list[j + 1]][node_list[j]],
                        data.time[node_list[j + 1]][node_list[j]]
                    );
                }
            }
        }
    }

    void update(const Data& data) {
        cal_attr(data);
        dep_time = self.T_E;
        ret_time = dep_time + self.T_D;
    }

    void set_node_list(const std::vector<int>& nl) {
        node_list = nl;
    }

    double cal_cost(const Data& data) {
        // FITNESS_DISTANCE_WEIGHT = 1.0, FITNESS_VEHICLE_WEIGHT = 2000.0
        transcost = self.dist * 1.0; 
        double dispatchcost = 0.0;
        if (!isempty()) {
            dispatchcost = 2000.0;
        }
        return transcost + dispatchcost;
    }

    bool isempty() const {
        return self.num_cus == 0;
    }

    bool check(const Data& data, std::vector<int>& nodes, bool& st_re_DC, bool& smaller_ca, bool& earlier_tw, double& cost) const {
        int length = node_list.size();
        st_re_DC = true;
        smaller_ca = true;
        earlier_tw = true;
        cost = 0.0;

        if (node_list[0] != data.DC || node_list[length - 1] != data.DC) {
            st_re_DC = false;
            return false;
        }

        double capacity = data.vehicle.capacity;
        double distance = 0.0;
        double time_val = data.start_time;
        double load = 0.0;

        for (int i = 1; i < length - 1; ++i) {
            nodes.push_back(node_list[i]);
            load += data.node[node_list[i]].delivery;
        }

        if (load > capacity) {
            smaller_ca = false;
            return false;
        }

        int pre_node = node_list[0];
        for (int i = 1; i < length; ++i) {
            int node = node_list[i];
            load = load - data.node[node].delivery + data.node[node].pickup;
            if (load > capacity) {
                smaller_ca = false;
                return false;
            }
            time_val += data.time[pre_node][node];
            if (time_val > data.node[node].end) {
                earlier_tw = false;
                return false;
            }
            time_val = std::max(time_val, data.node[node].start) + data.node[node].s_time;
            distance += data.dist[pre_node][node];
            pre_node = node;
        }

        cost = 2000.0 + distance * 1.0;
        return true;
    }
};

class Solution {
public:
    std::vector<Route> route_list;
    double cost = 0.0;

    Solution() {}
    Solution(const Data& data) {}

    Solution clone() const {
        Solution new_solution = *this;
        return new_solution;
    }

    void copy_from(const Solution& other) {
        route_list = other.route_list;
        cost = other.cost;
    }

    void reserve(const Data& data) {}

    void append(const Route& r) {
        route_list.push_back(r.clone());
    }

    void delete_route(int index) {
        route_list.erase(route_list.begin() + index);
    }

    Route& get(int index) {
        return route_list[index];
    }

    const Route& get(int index) const {
        return route_list[index];
    }

    Route& get_tail() {
        return route_list.back();
    }

    int len() const {
        return route_list.size();
    }

    void update(const Data& data) {
        int length = len();
        int index = 0;
        while (index < length) {
            Route& route = get(index);
            route.update(data);
            if (route.isempty()) {
                delete_route(index);
                length--;
            } else {
                index++;
            }
        }
    }

    void local_update(std::vector<int>& route_indice) {
        int length = len();
        int empty_id = -1;
        bool last_id_in = false;
        for (int item : route_indice) {
            if (get(item).isempty()) {
                empty_id = item;
            }
            if (item == length - 1) {
                last_id_in = true;
            }
        }
        if (empty_id != -1) {
            if (empty_id == length - 1) {
                route_list.pop_back();
            } else {
                route_list[empty_id] = route_list.back();
                route_list.pop_back();
                if (!last_id_in) {
                    route_indice.push_back(length - 1);
                }
            }
        }
    }

    void clear(const Data& data) {
        route_list.clear();
        cost = 0.0;
    }

    double cal_cost(const Data& data) {
        cost = 0.0;
        for (Route& route : route_list) {
            cost += route.cal_cost(data);
        }
        return cost;
    }

    std::string build_output_str() const {
        std::stringstream ss;
        ss << "Details of the solution:\n";
        int length = len();
        for (int i = 0; i < length; ++i) {
            const std::vector<int>& nl = route_list[i].node_list;
            ss << "route " << i << ", node_num " << nl.size() << ", cost " << route_list[i].transcost << ", nodes:";
            for (int node : nl) {
                ss << " " << node;
            }
            ss << "\n";
        }
        ss << "vehicle (route) number: " << length << "\n";
        ss.precision(6);
        ss << "Total cost: " << std::fixed << cost << "\n";
        return ss.str();
    }

    void output(const Data& data) const {
        std::string output_s = build_output_str();
        if (data.output_path.empty()) {
            std::cout << output_s;
        } else {
            std::ofstream out(data.output_path);
            if (out.is_open()) {
                out << output_s;
            }
        }
    }

    bool check(const Data& data, bool verbose = true) const {
        double total_cost = 0.0;
        int length = len();
        std::set<int> record;
        for (int i = 0; i < length; ++i) {
            const Route& route = get(i);
            std::vector<int> nodes;
            bool st_re_DC, smaller_ca, earlier_tw;
            double cost_val;
            route.check(data, nodes, st_re_DC, smaller_ca, earlier_tw, cost_val);
            if (!st_re_DC) {
                if (verbose) std::cout << "Not starting/ending at DC" << std::endl;
                return false;
            }
            if (!smaller_ca) {
                if (verbose) std::cout << "Capacity violation" << std::endl;
                return false;
            }
            if (!earlier_tw) {
                if (verbose) std::cout << "Time window violation" << std::endl;
                return false;
            }
            total_cost += cost_val;
            for (int node : nodes) {
                if (record.find(node) == record.end()) {
                    record.insert(node);
                } else {
                    if (verbose) std::cout << "Duplicate node: " << node << std::endl;
                    return false;
                }
            }
        }
        for (int i = 0; i <= data.customer_num; ++i) {
            if (i == data.DC) continue;
            if (record.find(i) == record.end()) {
                if (verbose) std::cout << "Missing customer: " << i << std::endl;
                return false;
            }
        }

        if (verbose) {
            std::cout << "This cost " << cost << ", check total cost " << total_cost 
                      << ", diff " << (total_cost - cost) << std::endl;
        }
        return true;
    }
};
