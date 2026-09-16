from .config import (
    BENCHMARKING_O_1_EVAL,
    FITNESS_DISTANCE_WEIGHT,
    FITNESS_VEHICLE_WEIGHT,
    MAX_NODE_IN_ROUTE,
)
from .solution import Attr, attr_for_one_node, connect_inplace, connect_into
from .state import call_count_move_eval, mean_duration_move_eval, mean_route_len


def check_capacity(a: Attr, b: Attr, data) -> bool:
    return max(a.C_H + b.C_E, a.C_L + b.C_H) - data.vehicle.capacity <= 0


def check_tw(a: Attr, b: Attr, data) -> bool:
    return (a.T_E + a.T_D + data.time[a.e][b.s] - b.T_L) <= 0


def _chk_route_list_cpu(nl, data):
    length = len(nl)
    if nl[0] != data.DC or nl[length - 1] != data.DC:
        return False, 0.0
    if length == 2:
        return True, 0.0

    capacity = data.vehicle.capacity
    distance = 0.0
    time_val = data.start_time
    load = 0.0
    for node in nl:
        load += data.node[node].delivery
    if load > capacity:
        return False, 0.0

    pre_node = nl[0]
    for i in range(1, length):
        node = nl[i]
        load = load - data.node[node].delivery + data.node[node].pickup
        if load > capacity:
            return False, 0.0
        time_val += data.time[pre_node][node]
        if time_val > data.node[node].end:
            return False, 0.0
        time_val = max(time_val, data.node[node].start) + data.node[node].s_time
        distance += data.dist[pre_node][node]
        pre_node = node

    cost = FITNESS_VEHICLE_WEIGHT + distance * FITNESS_DISTANCE_WEIGHT
    return True, cost


def _chk_route_list(nl, data):
    backend = getattr(data, "backend", None)
    if backend is not None:
        return backend.evaluate_route(nl)
    return _chk_route_list_cpu(nl, data)


def evaluate_route_batch(routes, data):
    backend = getattr(data, "backend", None)
    if backend is not None:
        return backend.evaluate_routes(routes)
    return [_chk_route_list_cpu(route, data) for route in routes]


def chk_nl_node_pos_O_n(nl, inserted_node: int, pos: int, data):
    tmp = list(nl)
    tmp.insert(pos, inserted_node)
    return _chk_route_list(tmp, data)


def chk_route_O_n(route, data):
    return _chk_route_list(route.node_list, data)


def eval_route(s, seq_list, seq_list_len: int, tmp_attr: Attr, data) -> bool:
    if seq_list_len < 2:
        return False

    if seq_list[0].r_index == -1:
        attr_1 = attr_for_one_node(data, seq_list[0].start_point)
    else:
        attr_1 = s.get(seq_list[0].r_index).gat(seq_list[0].start_point, seq_list[0].end_point)

    if seq_list[1].r_index == -1:
        attr_2 = attr_for_one_node(data, seq_list[1].start_point)
    else:
        attr_2 = s.get(seq_list[1].r_index).gat(seq_list[1].start_point, seq_list[1].end_point)

    if (not check_tw(attr_1, attr_2, data)) or (not check_capacity(attr_1, attr_2, data)):
        return False
    connect_into(
        attr_1,
        attr_2,
        tmp_attr,
        data.dist[attr_1.e][attr_2.s],
        data.time[attr_1.e][attr_2.s],
    )

    for i in range(2, seq_list_len):
        if seq_list[i].r_index == -1:
            attr = attr_for_one_node(data, seq_list[i].start_point)
        else:
            attr = s.get(seq_list[i].r_index).gat(seq_list[i].start_point, seq_list[i].end_point)

        if (not check_tw(tmp_attr, attr, data)) or (not check_capacity(tmp_attr, attr, data)):
            return False
        connect_inplace(tmp_attr, attr, data.dist[tmp_attr.e][attr.s], data.time[tmp_attr.e][attr.s])

    return True


def eval_move(s, m, data) -> bool:
    r_indice = [m.r_indice[0]]
    if m.r_indice[1] != -2:
        r_indice.append(m.r_indice[1])
    ori_cost = s.get(r_indice[0]).cal_cost(data)

    if not data.O_1_evl:
        target_n_l = []
        for i in range(m.len_1):
            seq = m.seqList_1[i]
            source_n_l = s.get(seq.r_index).node_list
            for index in range(seq.start_point, seq.end_point + 1):
                target_n_l.append(source_n_l[index])
        flag, new_cost = _chk_route_list(target_n_l, data)
        if not flag:
            return False

        if len(r_indice) == 2:
            target_n_l_2 = []
            for i in range(m.len_2):
                seq = m.seqList_2[i]
                if seq.r_index == -1:
                    target_n_l_2.append(data.DC)
                    continue
                source_n_l = s.get(seq.r_index).node_list
                for index in range(seq.start_point, seq.end_point + 1):
                    target_n_l_2.append(source_n_l[index])
            if r_indice[1] != -1:
                ori_cost += s.get(r_indice[1]).cal_cost(data)
            flag, cost = _chk_route_list(target_n_l_2, data)
            if not flag:
                return False
            new_cost += cost

        m.delta_cost = new_cost - ori_cost
        return True

    tmp_attr_1 = Attr()
    if not eval_route(s, m.seqList_1, m.len_1, tmp_attr_1, data):
        return False
    new_cost = 0.0
    if tmp_attr_1.num_cus != 0:
        new_cost += data.vehicle.d_cost + tmp_attr_1.dist * data.vehicle.unit_cost
    if len(r_indice) == 2:
        tmp_attr_2 = Attr()
        if not eval_route(s, m.seqList_2, m.len_2, tmp_attr_2, data):
            return False
        if r_indice[1] != -1:
            ori_cost += s.get(r_indice[1]).cal_cost(data)
        if tmp_attr_2.num_cus != 0:
            new_cost += data.vehicle.d_cost + tmp_attr_2.dist * data.vehicle.unit_cost
    m.delta_cost = new_cost - ori_cost

    return True
