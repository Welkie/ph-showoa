from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import INFEASIBLE, FITNESS_VEHICLE_WEIGHT, FITNESS_DISTANCE_WEIGHT


@dataclass
class Attr:
    num_cus: int = 0
    dist: float = 0.0
    s: int = 0
    e: int = 0
    T_D: float = 0.0
    T_E: float = 0.0
    T_L: float = 0.0
    C_E: float = 0.0
    C_H: float = 0.0
    C_L: float = 0.0

    def copy(self) -> "Attr":
        return Attr(
            num_cus=self.num_cus,
            dist=self.dist,
            s=self.s,
            e=self.e,
            T_D=self.T_D,
            T_E=self.T_E,
            T_L=self.T_L,
            C_E=self.C_E,
            C_H=self.C_H,
            C_L=self.C_L,
        )


def attr_for_one_node(data, node: int, a: Optional[Attr] = None) -> Attr:
    if a is None:
        a = Attr()
    a.s = node
    a.e = node
    a.dist = 0.0

    if node == data.DC:
        a.num_cus = 0
        a.T_D = 0.0
        a.T_E = data.start_time
        a.T_L = data.end_time
        a.C_E = 0.0
        a.C_L = 0.0
        a.C_H = 0.0
    else:
        a.num_cus = 1
        a.T_D = data.node[node].s_time
        a.T_E = data.node[node].start
        a.T_L = data.node[node].end
        a.C_E = data.node[node].delivery
        a.C_L = data.node[node].pickup
        a.C_H = max(a.C_E, a.C_L)
    return a


def connect_attrs(tmp_a: Attr, tmp_b: Attr, dist_ij: float, t_ij: float) -> Attr:
    merged = Attr()
    connect_into(tmp_a, tmp_b, merged, dist_ij, t_ij)
    return merged


def connect_into(tmp_a: Attr, tmp_b: Attr, merged_attr: Attr, dist_ij: float, t_ij: float) -> None:
    merged_attr.num_cus = tmp_a.num_cus + tmp_b.num_cus
    merged_attr.dist = tmp_a.dist + dist_ij + tmp_b.dist

    delta = tmp_a.T_D + t_ij
    delta_wt = max(tmp_b.T_E - delta - tmp_a.T_L, 0.0)
    merged_attr.T_D = tmp_a.T_D + tmp_b.T_D + t_ij + delta_wt
    merged_attr.T_E = max(tmp_b.T_E - delta, tmp_a.T_E) - delta_wt
    merged_attr.T_L = min(tmp_b.T_L - delta, tmp_a.T_L)

    merged_attr.C_E = tmp_a.C_E + tmp_b.C_E
    merged_attr.C_H = max(tmp_a.C_H + tmp_b.C_E, tmp_a.C_L + tmp_b.C_H)
    merged_attr.C_L = tmp_a.C_L + tmp_b.C_L

    merged_attr.s = tmp_a.s
    merged_attr.e = tmp_b.e


def connect_inplace(merged_attr: Attr, tmp_b: Attr, dist_ij: float, t_ij: float) -> None:
    merged_attr.num_cus = merged_attr.num_cus + tmp_b.num_cus
    merged_attr.dist = merged_attr.dist + dist_ij + tmp_b.dist

    delta = merged_attr.T_D + t_ij
    delta_wt = max(tmp_b.T_E - delta - merged_attr.T_L, 0.0)
    merged_attr.T_D = merged_attr.T_D + tmp_b.T_D + t_ij + delta_wt
    merged_attr.T_E = max(tmp_b.T_E - delta, merged_attr.T_E) - delta_wt
    merged_attr.T_L = min(tmp_b.T_L - delta, merged_attr.T_L)

    old_c_e = merged_attr.C_E
    old_c_h = merged_attr.C_H
    old_c_l = merged_attr.C_L
    merged_attr.C_E = old_c_e + tmp_b.C_E
    merged_attr.C_H = max(old_c_h + tmp_b.C_E, old_c_l + tmp_b.C_H)
    merged_attr.C_L = old_c_l + tmp_b.C_L

    merged_attr.e = tmp_b.e


def make_tmp_nl(data) -> List[int]:
    return [data.DC, data.DC]


class Route:
    def __init__(self, data=None) -> None:
        self.node_list: List[int] = []
        self.dep_time = 0.0
        self.ret_time = 0.0
        self.transcost = 0.0
        self.attr: List[Attr] = []
        self.self = Attr()
        if data is not None:
            self.node_list = [data.DC, data.DC]
            self.update(data)

    def clone(self) -> "Route":
        new_route = Route()
        new_route.node_list = list(self.node_list)
        new_route.dep_time = self.dep_time
        new_route.ret_time = self.ret_time
        new_route.transcost = self.transcost
        new_route.attr = [a.copy() for a in self.attr]
        new_route.self = self.self.copy()
        return new_route

    def isempty(self) -> bool:
        return len(self.node_list) <= 2 or self.self.num_cus == 0

    def gat(self, i: int, j: int) -> Attr:
        nl_len = len(self.node_list)
        return self.attr[i * nl_len + j]

    def cal_attr(self, data) -> None:
        nl_len = len(self.node_list)
        if nl_len <= 2:
            self.attr = [Attr() for _ in range(nl_len * nl_len)]
            for i in range(nl_len):
                attr_for_one_node(data, self.node_list[i], self.gat(i, i))
            if nl_len == 2:
                connect_into(
                    self.gat(0, 0),
                    self.gat(1, 1),
                    self.gat(0, 1),
                    data.dist[data.DC][data.DC],
                    data.time[data.DC][data.DC],
                )
                self.self = self.gat(0, 1).copy()
            return

        end_index = nl_len - 1
        self.attr = [Attr() for _ in range(nl_len * nl_len)]

        for i in range(end_index + 1):
            attr_for_one_node(data, self.node_list[i], self.gat(i, i))

        for i in range(end_index):
            for j in range(i + 1, end_index + 1):
                connect_into(
                    self.gat(i, j - 1),
                    self.gat(j, j),
                    self.gat(i, j),
                    data.dist[self.node_list[j - 1]][self.node_list[j]],
                    data.time[self.node_list[j - 1]][self.node_list[j]],
                )
        self.self = self.gat(0, end_index).copy()

        for i in range(end_index - 1, 0, -1):
            feasible = True
            for j in range(i - 1, 0, -1):
                if (i - j + 1) > 2:
                    break
                if not feasible:
                    self.gat(i, j).num_cus = INFEASIBLE
                    continue
                if (
                    self.gat(i, j + 1).T_E
                    + self.gat(i, j + 1).T_D
                    + data.time[self.node_list[j + 1]][self.node_list[j]]
                    - self.gat(j, j).T_L
                    > 0
                ):
                    self.gat(i, j).num_cus = INFEASIBLE
                    feasible = False
                else:
                    connect_into(
                        self.gat(i, j + 1),
                        self.gat(j, j),
                        self.gat(i, j),
                        data.dist[self.node_list[j + 1]][self.node_list[j]],
                        data.time[self.node_list[j + 1]][self.node_list[j]],
                    )

    def update(self, data) -> None:
        self.cal_attr(data)
        self.dep_time = self.self.T_E
        self.ret_time = self.dep_time + self.self.T_D

    def cal_cost(self, data) -> float:
        if self.isempty():
            return 0.0
        self.transcost = self.self.dist * FITNESS_DISTANCE_WEIGHT
        return FITNESS_VEHICLE_WEIGHT + self.transcost

    def check(self, data):
        nodes = []
        nl = self.node_list
        length = len(nl)

        st_re_DC = True
        smaller_ca = True
        earlier_tw = True
        cost = 0.0

        if nl[0] != data.DC or nl[length - 1] != data.DC:
            st_re_DC = False
            return nodes, st_re_DC, smaller_ca, earlier_tw, cost

        capacity = data.vehicle.capacity
        distance = 0.0
        time_val = data.start_time
        load = 0.0

        for i in range(1, length - 1):
            nodes.append(nl[i])
            load += data.node[nl[i]].delivery

        if load > capacity:
            smaller_ca = False
            return nodes, st_re_DC, smaller_ca, earlier_tw, cost

        pre_node = nl[0]
        for i in range(1, length):
            node = nl[i]
            load = load - data.node[node].delivery + data.node[node].pickup
            if load < 0 or load > capacity:
                smaller_ca = False
                return nodes, st_re_DC, smaller_ca, earlier_tw, cost
            time_val += data.time[pre_node][node]
            if time_val > data.node[node].end:
                earlier_tw = False
                return nodes, st_re_DC, smaller_ca, earlier_tw, cost
            time_val = max(time_val, data.node[node].start) + data.node[node].s_time
            distance += data.dist[pre_node][node]
            pre_node = node

        cost = FITNESS_VEHICLE_WEIGHT + distance * FITNESS_DISTANCE_WEIGHT
        return nodes, st_re_DC, smaller_ca, earlier_tw, cost


class Solution:
    def __init__(self, data=None) -> None:
        self.route_list: List[Route] = []
        self.cost = float("inf")

    def clone(self) -> "Solution":
        new_solution = Solution()
        new_solution.route_list = [r.clone() for r in self.route_list]
        new_solution.cost = self.cost
        return new_solution

    def copy_from(self, other: "Solution") -> None:
        self.route_list = [r.clone() for r in other.route_list]
        self.cost = other.cost

    def append(self, r: Route) -> None:
        if not r.isempty():
            self.route_list.append(r.clone())

    def delete(self, index: int) -> None:
        self.route_list.pop(index)

    def get(self, index: int) -> Route:
        return self.route_list[index]

    def len(self) -> int:
        return len(self.route_list)

    def update(self, data) -> None:
        self.route_list = [r for r in self.route_list if not r.isempty()]
        for r in self.route_list:
            r.update(data)

    def local_update(self, route_indice: List[int]) -> None:
        length = self.len()
        empty_id = -1
        last_id_in = False
        for item in route_indice:
            if 0 <= item < length and self.get(item).isempty():
                empty_id = item
            if item == length - 1:
                last_id_in = True
        if empty_id != -1:
            if empty_id == length - 1:
                self.route_list.pop()
            else:
                self.route_list[empty_id] = self.route_list[-1]
                self.route_list.pop()
                if not last_id_in:
                    route_indice.append(length - 1)

    def clear(self, data=None) -> None:
        self.route_list = []
        self.cost = float("inf")

    def cal_cost(self, data) -> float:
        self.update(data)
        if len(self.route_list) == 0:
            self.cost = float("inf")
            return self.cost
        self.cost = sum(r.cal_cost(data) for r in self.route_list)
        return self.cost

    def check(self, data, verbose=True) -> bool:
        if len(self.route_list) > data.vehicle.max_num:
            if verbose:
                print(f"Vehicle limit exceeded: {len(self.route_list)} > {data.vehicle.max_num}")
            return False
        total_cost = 0.0
        record = set()
        for r in self.route_list:
            nodes, st_re_DC, smaller_ca, earlier_tw, cost = r.check(data)
            if not st_re_DC or not smaller_ca or not earlier_tw:
                if verbose:
                    print("Route check failed for route:", r.node_list)
                return False
            total_cost += cost
            for node in nodes:
                if node != data.DC and 1 <= node <= data.customer_num:
                    if node in record:
                        if verbose:
                            print(f"Customer {node} visited multiple times!")
                        return False
                    record.add(node)
        if len(record) != data.customer_num:
            if verbose:
                print(f"Visited {len(record)} customers, expected {data.customer_num}")
            return False
        return True

    def build_output_str(self) -> str:
        length = self.len()
        td = self.cost - 2000.0 * length
        output_s = "Details of the solution:\n"
        for i in range(length):
            nl = self.route_list[i].node_list
            output_s += (
                "route "
                + str(i)
                + ", node_num "
                + str(len(nl))
                + ", cost "
                + str(self.route_list[i].transcost)
                + ", nodes:"
            )
            for node in nl:
                output_s += " " + str(node)
            output_s += "\n"
        output_s += "vehicle (route) number: " + str(length) + "\n"
        output_s += "Vehicle count: " + str(length) + "\n"
        output_s += f"Total distance: {td:.4f}\n"
        output_s += "Total cost: " + str(self.cost) + "\n"
        return output_s

    def output(self, data) -> None:
        output_s = self.build_output_str()
        if not getattr(data, "if_output", False):
            print(output_s, end="", flush=True)
        else:
            with open(data.output, "w", encoding="utf-8") as out:
                out.write(output_s)
