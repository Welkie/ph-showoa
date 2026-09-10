from typing import List

class Attr:
    def __init__(self):
        self.s = 0
        self.e = 0
        self.load = 0.0
        self.C_H = 0.0
        self.C_L = 0.0
        self.C_E = 0.0
        self.dist = 0.0
        self.T_D = 0.0
        self.T_E = 0.0
        self.T_L = 0.0
        self.num_cus = 0

def attr_for_one_node(data, node: int) -> Attr:
    attr = Attr()
    attr.s = node
    attr.e = node
    attr.load = data.node[node].pickup - data.node[node].delivery
    attr.C_H = max(data.node[node].pickup, data.node[node].delivery)
    attr.C_L = 0.0
    attr.C_E = 0.0
    attr.dist = 0.0
    attr.T_D = data.node[node].s_time
    attr.T_E = data.node[node].start
    attr.T_L = data.node[node].end
    attr.num_cus = 1 if node != data.DC else 0
    return attr

def connect_into(attr1: Attr, attr2: Attr, target: Attr, dist_val: float, time_val: float):
    target.s = attr1.s
    target.e = attr2.e
    target.load = attr1.load + attr2.load
    target.C_H = max(attr1.C_H + attr2.C_E, attr1.C_L + attr2.C_H)
    target.dist = attr1.dist + attr2.dist + dist_val
    delta = attr1.T_E + attr1.T_D + time_val - attr2.T_L
    target.T_D = attr1.T_D + attr2.T_D + time_val + max(0.0, -delta)
    target.T_E = max(attr2.T_E - attr1.T_D - time_val, attr1.T_E)
    target.T_L = min(attr2.T_L - attr1.T_D - time_val, attr1.T_L)
    target.num_cus = attr1.num_cus + attr2.num_cus

def connect_inplace(attr1: Attr, attr2: Attr, dist_val: float, time_val: float):
    connect_into(attr1, attr2, attr1, dist_val, time_val)

class RouteSelf:
    def __init__(self):
        self.dist = 0.0
        self.load = 0.0
        self.C_H = 0.0

class Route:
    def __init__(self, data=None):
        self.node_list: List[int] = []
        self.self = RouteSelf()
        if data is not None:
            self.node_list = [data.DC, data.DC]

    def isempty(self) -> bool:
        return len(self.node_list) <= 2

    def update(self, data) -> None:
        if self.isempty():
            self.self.dist = 0.0
            self.self.load = 0.0
            self.self.C_H = 0.0
            return
        dist = 0.0
        load = 0.0
        max_load = 0.0
        for i in range(1, len(self.node_list) - 1):
            load += data.node[self.node_list[i]].delivery
        max_load = load
        for i in range(1, len(self.node_list)):
            prev = self.node_list[i - 1]
            curr = self.node_list[i]
            dist += data.dist[prev][curr]
            load = load - data.node[curr].delivery + data.node[curr].pickup
            if load > max_load:
                max_load = load
        self.self.dist = dist
        self.self.load = load
        self.self.C_H = max_load

    def cal_cost(self, data) -> float:
        if self.isempty():
            return 0.0
        return data.vehicle.d_cost + self.self.dist * data.vehicle.unit_cost

    def clone(self):
        r = Route()
        r.node_list = list(self.node_list)
        r.self.dist = self.self.dist
        r.self.load = self.self.load
        r.self.C_H = self.self.C_H
        return r


class Solution:
    def __init__(self, data=None):
        self.route_list: List[Route] = []
        self.cost = float("inf")

    def len(self) -> int:
        return len(self.route_list)

    def get(self, idx: int) -> Route:
        return self.route_list[idx]

    def append(self, route: Route) -> None:
        if not route.isempty():
            self.route_list.append(route)

    def clear(self, data=None) -> None:
        self.route_list.clear()
        self.cost = float("inf")

    def update(self, data) -> None:
        self.route_list = [r for r in self.route_list if not r.isempty()]
        for r in self.route_list:
            r.update(data)

    def cal_cost(self, data) -> float:
        self.update(data)
        if len(self.route_list) == 0:
            self.cost = float("inf")
            return self.cost
        total_cost = 0.0
        for r in self.route_list:
            total_cost += r.cal_cost(data)
        self.cost = total_cost
        return self.cost

    def clone(self):
        s = Solution()
        s.cost = self.cost
        s.route_list = [r.clone() for r in self.route_list]
        return s

    def copy_from(self, other):
        self.cost = other.cost
        self.route_list = [r.clone() for r in other.route_list]

    def check(self, data, verbose=True) -> bool:
        from .eval import _chk_route_list
        record = set()
        for r in self.route_list:
            flag, _ = _chk_route_list(r.node_list, data)
            if not flag:
                if verbose:
                    print("Route check failed for route:", r.node_list)
                return False
            for node in r.node_list:
                if node != data.DC and 0 <= node <= data.customer_num:
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

    def output(self, data) -> None:
        print("Best solution found:")
        print(f"Total cost: {self.cost:.4f}, Vehicle count: {len(self.route_list)}")
        for idx, r in enumerate(self.route_list):
            print(f"Route {idx + 1}: {r.node_list}")
