from dataclasses import dataclass
from typing import List, Optional

from .config import (
    FITNESS_DISTANCE_WEIGHT,
    FITNESS_VEHICLE_WEIGHT,
    INFEASIBLE,
    MAX_NODE_IN_ROUTE,
)


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


def equal_attr(a: Attr, b: Attr) -> bool:
    return (
        a.num_cus == b.num_cus
        and a.dist == b.dist
        and a.s == b.s
        and a.e == b.e
        and a.T_D == b.T_D
        and a.T_E == b.T_E
        and a.T_L == b.T_L
        and a.C_E == b.C_E
        and a.C_H == b.C_H
        and a.C_L == b.C_L
    )


class Route:
    def __init__(self, data) -> None:
        self.node_list: List[int] = []
        self.dep_time = 0.0
        self.ret_time = 0.0
        self.transcost = 0.0
        self.attr: List[Attr] = []
        self.self = Attr()

        self.attr = []
        self.node_list = []
        self.node_list.append(data.DC)
        self.node_list.append(data.DC)
        self.update(data)

    def clone(self) -> "Route":
        new_route = Route.__new__(Route)
        new_route.node_list = list(self.node_list)
        new_route.dep_time = self.dep_time
        new_route.ret_time = self.ret_time
        new_route.transcost = self.transcost
        new_route.attr = [a.copy() for a in self.attr]
        new_route.self = self.self.copy()
        return new_route

    def gat(self, i: int, j: int) -> Attr:
        nl_len = len(self.node_list)
        return self.attr[i * nl_len + j]

    def cal_attr(self, data) -> None:
        nl_len = len(self.node_list)
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

    def set_node_list(self, nl: List[int]) -> None:
        self.node_list = list(nl)

    def cal_cost(self, data) -> float:
        self.transcost = self.self.dist * FITNESS_DISTANCE_WEIGHT
        dispatchcost = 0.0
        if not self.isempty():
            dispatchcost = FITNESS_VEHICLE_WEIGHT
        return self.transcost + dispatchcost

    def isempty(self) -> bool:
        return self.self.num_cus == 0

    def check(self, data):
        nodes = []
        nl = self.node_list
        length = len(nl)

        st_re_DC = True
        smaller_ca = True
        earlier_tw = True
        cost = 0.0

        if nl[0] != data.DC or nl[length - 1] != data.DC:
            print("Not starting/ending at DC")
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
            print("Whole delivery > capacity")
            return nodes, st_re_DC, smaller_ca, earlier_tw, cost

        pre_node = nl[0]
        for i in range(1, length):
            node = nl[i]
            load = load - data.node[node].delivery + data.node[node].pickup
            if load > capacity:
                smaller_ca = False
                print(
                    "Load %f > capacity %f at %d th node: %d, with delivery %f and pickup %f"
                    % (
                        load,
                        capacity,
                        i,
                        node,
                        data.node[node].delivery,
                        data.node[node].pickup,
                    )
                )
                return nodes, st_re_DC, smaller_ca, earlier_tw, cost
            time_val += data.time[pre_node][node]
            if time_val > data.node[node].end:
                earlier_tw = False
                print(
                    "Arrive at %d th node: %d at time %f > tw end %f"
                    % (i, node, time_val, data.node[node].end)
                )
                return nodes, st_re_DC, smaller_ca, earlier_tw, cost
            time_val = max(time_val, data.node[node].start) + data.node[node].s_time
            distance += data.dist[pre_node][node]
            pre_node = node

        cost = FITNESS_VEHICLE_WEIGHT + distance * FITNESS_DISTANCE_WEIGHT
        return nodes, st_re_DC, smaller_ca, earlier_tw, cost


class Solution:
    def __init__(self, data=None) -> None:
        self.route_list: List[Route] = []
        self.cost = 0.0
        if data is not None:
            self.route_list = []

    def clone(self) -> "Solution":
        new_solution = Solution()
        new_solution.route_list = [r.clone() for r in self.route_list]
        new_solution.cost = self.cost
        return new_solution

    def copy_from(self, other: "Solution") -> None:
        self.route_list = [r.clone() for r in other.route_list]
        self.cost = other.cost

    def reserve(self, data) -> None:
        return None

    def append(self, r: Route) -> None:
        self.route_list.append(r.clone())

    def delete(self, index: int) -> None:
        self.route_list.pop(index)

    def get(self, index: int) -> Route:
        return self.route_list[index]

    def get_tail(self) -> Route:
        return self.route_list[-1]

    def len(self) -> int:
        return len(self.route_list)

    def update(self, data) -> None:
        length = self.len()
        index = 0
        while index < length:
            route = self.get(index)
            route.update(data)
            if route.isempty():
                self.delete(index)
                length -= 1
            else:
                index += 1

    def local_update(self, route_indice: List[int]) -> None:
        length = self.len()
        empty_id = -1
        last_id_in = False
        for item in route_indice:
            if self.get(item).isempty():
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

    def clear(self, data) -> None:
        self.route_list = []
        self.cost = 0.0

    def cal_cost(self, data) -> float:
        self.cost = 0.0
        for route in self.route_list:
            self.cost += route.cal_cost(data)
        return self.cost

    def build_output_str(self) -> str:
        output_s = "Details of the solution:\n"
        length = self.len()
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
        output_s += "Total cost: " + str(self.cost) + "\n"
        return output_s

    def output(self, data) -> None:
        output_s = self.build_output_str()
        if not data.if_output:
            print(output_s, end="")
        else:
            with open(data.output, "w", encoding="utf-8") as out:
                out.write(output_s)

    def check(self, data) -> bool:
        total_cost = 0.0
        length = self.len()
        record = set()
        for i in range(length):
            route = self.get(i)
            nodes, st_re_DC, smaller_ca, earlier_tw, cost = route.check(data)
            if not st_re_DC or not smaller_ca or not earlier_tw:
                return False
            total_cost += cost
            for node in nodes:
                if node not in record:
                    record.add(node)
                else:
                    print("Duplicate node: %d" % node)
                    return False
        for i in range(data.customer_num + 1):
            if i == data.DC:
                continue
            if i not in record:
                print("Missing customer: %d" % i)
                return False

        print(
            "This cost %f, check total cost %f, diff %f"
            % (self.cost, total_cost, total_cost - self.cost)
        )
        return True
