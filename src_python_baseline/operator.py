import math
from typing import List

from .config import (
    INFEASIBLE,
    MAX_NODE_IN_ROUTE,
    MAX_POINT,
    PRECISION,
    RCRS,
    TD,
)
from .compute_backend import njit, _evaluate_insertions_cpu
from .eval import chk_nl_node_pos_O_n, eval_move, evaluate_route_batch
from .move import Move, Seq
from .solution import Route, make_tmp_nl
from .util import argsort, rand, randint
import numpy as np

TMP_MOVE = Move()


def _cuda_batches_enabled(data) -> bool:
    backend = getattr(data, "backend", None)
    return backend is not None and getattr(backend, "is_cuda", False)





def maintain_unrouted(i, node, index, unrouted, unrouted_d, unrouted_p, data):
    unrouted[i] = unrouted[index - 1]
    index -= 1
    unrouted_d -= data.node[node].delivery
    unrouted_p -= data.node[node].pickup
    return index, unrouted_d, unrouted_p


def cal_tc(nl, inserted_node, pos, unrouted_d, unrouted_p, data):
    tmp = list(nl)
    tmp.insert(pos, inserted_node)
    new_len = len(tmp)
    capacity = data.vehicle.capacity

    route_d = 0.0
    route_p = 0.0
    for i in range(1, new_len - 1):
        route_d += data.node[tmp[i]].delivery
        route_p += data.node[tmp[i]].pickup

    load = [0.0] * new_len
    cd = [0.0] * new_len
    cp = [0.0] * new_len

    load[0] = route_d
    for i in range(1, new_len):
        node = tmp[i]
        load[i] = load[i - 1] - data.node[node].delivery + data.node[node].pickup
        cd[i] = cd[i - 1] + data.dist[tmp[i - 1]][node]

    cp[new_len - 1] = 0.0
    for i in range(new_len - 1, 0, -1):
        cp[i - 1] = cp[i] + data.dist[tmp[i - 1]][tmp[i]]

    rd = [0.0] * new_len
    rp = [0.0] * new_len
    rd[0] = capacity - route_d
    rp[new_len - 2] = capacity - route_p

    for i in range(1, new_len - 1):
        rd[i] = min(rd[i - 1], capacity - load[i])
        rp[new_len - 2 - i] = min(rp[new_len - 1 - i], capacity - load[new_len - 2 - i])

    rdt_u = 0.0
    rdt_d = 0.0
    rpt_u = 0.0
    rpt_d = 0.0
    for i in range(new_len - 1):
        rdt_u += rd[i] * cd[i + 1]
        rdt_d += cd[i + 1]
        rpt_u += rp[i] * cp[i]
        rpt_d += cp[i]

    rdt = rdt_u / rdt_d
    rpt = rpt_u / rpt_d

    tc = (unrouted_d / data.all_delivery) * (1 - rdt / capacity) + (
        unrouted_p / data.all_pickup
    ) * (1 - rpt / capacity)
    return tc


def criterion(r, data, node, pos, unrouted_d, unrouted_p):
    nl = r.node_list
    pre = nl[pos - 1]
    suc = nl[pos]
    td = data.dist[pre][node] + data.dist[node][suc] - data.dist[pre][suc]
    if data.n_insert == TD:
        return td

    tc = cal_tc(r.node_list, node, pos, unrouted_d, unrouted_p, data)
    rs = data.dist[data.DC][node] + data.dist[node][data.DC]

    rcrs = td + data.lambda_gamma[0] * tc * (2 * data.max_dist - data.min_dist) - data.lambda_gamma[1] * rs
    return rcrs


@njit(nogil=True, fastmath=True)
def _cal_score_numba_core(
    index, r_len, feasible_arr, candidate_nodes, route_nodes,
    unrouted_d, unrouted_p, capacity, delivery, pickup, dist,
    all_delivery, all_pickup, is_td, lambda_gamma, max_dist, min_dist, DC
):
    best_scores = np.full(index, np.inf)
    best_positions = np.full(index, -1, dtype=np.int32)
    count = 0
    
    new_len = r_len + 1
    tmp = np.zeros(new_len, dtype=np.int32)
    load = np.zeros(new_len, dtype=np.float64)
    cd = np.zeros(new_len, dtype=np.float64)
    cp = np.zeros(new_len, dtype=np.float64)
    rd = np.zeros(new_len, dtype=np.float64)
    rp = np.zeros(new_len, dtype=np.float64)

    for i in range(index):
        node = candidate_nodes[i]
        for pos in range(1, r_len):
            if not feasible_arr[i * r_len + pos]:
                continue
            count += 1
            
            pre = route_nodes[pos - 1]
            suc = route_nodes[pos]
            td = dist[pre, node] + dist[node, suc] - dist[pre, suc]
            
            if is_td:
                utility = td
            else:
                for j in range(pos):
                    tmp[j] = route_nodes[j]
                tmp[pos] = node
                for j in range(pos, r_len):
                    tmp[j + 1] = route_nodes[j]
                    
                route_d = 0.0
                route_p = 0.0
                for j in range(1, new_len - 1):
                    route_d += delivery[tmp[j]]
                    route_p += pickup[tmp[j]]
                    
                load[0] = route_d
                for j in range(1, new_len):
                    curr = tmp[j]
                    load[j] = load[j - 1] - delivery[curr] + pickup[curr]
                    cd[j] = cd[j - 1] + dist[tmp[j - 1], curr]
                    
                cp[new_len - 1] = 0.0
                for j in range(new_len - 1, 0, -1):
                    cp[j - 1] = cp[j] + dist[tmp[j - 1], tmp[j]]
                    
                rd[0] = capacity - route_d
                rp[new_len - 2] = capacity - route_p
                
                for j in range(1, new_len - 1):
                    rd[j] = min(rd[j - 1], capacity - load[j])
                    rp[new_len - 2 - j] = min(rp[new_len - 1 - j], capacity - load[new_len - 2 - j])
                    
                rdt_u = 0.0
                rdt_d = 0.0
                rpt_u = 0.0
                rpt_d = 0.0
                for j in range(new_len - 1):
                    rdt_u += rd[j] * cd[j + 1]
                    rdt_d += cd[j + 1]
                    rpt_u += rp[j] * cp[j]
                    rpt_d += cp[j]
                    
                rdt = rdt_u / rdt_d if rdt_d > 0 else 0.0
                rpt = rpt_u / rpt_d if rpt_d > 0 else 0.0
                
                tc = (unrouted_d / all_delivery) * (1.0 - rdt / capacity) + \
                     (unrouted_p / all_pickup) * (1.0 - rpt / capacity)
                     
                rs = dist[DC, node] + dist[node, DC]
                utility = td + lambda_gamma[0] * tc * (2.0 * max_dist - min_dist) - lambda_gamma[1] * rs
            
            if utility - best_scores[i] < -0.001:  # PRECISION
                best_scores[i] = utility
                best_positions[i] = pos
                
    return count, best_scores, best_positions

def cal_score(feasible_pos, unrouted, score, index, r, unrouted_d, unrouted_p, data):
    if index == 0:
        return False
    r_len = len(r.node_list)

    candidate_nodes = np.array([unrouted[i][0] for i in range(index)], dtype=np.int32)
    if getattr(data, "in_initialization", False):
        feasible_arr, _cost_arr = _evaluate_insertions_cpu(r.node_list, candidate_nodes, data.backend.snapshot)
    else:
        feasible_arr, _cost_arr = data.backend.evaluate_insertions(r.node_list, candidate_nodes)

    route_nodes_arr = np.array(r.node_list, dtype=np.int32)

    count, best_scores, best_positions = _cal_score_numba_core(
        index, r_len, feasible_arr, candidate_nodes, route_nodes_arr,
        float(unrouted_d), float(unrouted_p),
        float(data.vehicle.capacity),
        data.backend.snapshot.delivery,
        data.backend.snapshot.pickup,
        data.backend.snapshot.dist,
        float(data.all_delivery),
        float(data.all_pickup),
        bool(data.n_insert == TD),
        np.array(data.lambda_gamma, dtype=np.float64),
        float(data.max_dist),
        float(data.min_dist),
        int(data.DC)
    )

    if count == 0:
        return False

    for i in range(index):
        for pos in range(1, r_len):
            feasible_pos[i * MAX_NODE_IN_ROUTE + pos] = bool(feasible_arr[i * r_len + pos])

    for i in range(index):
        score[i] = best_scores[i]
        unrouted[i][1] = best_positions[i]

    return True


def find_unrouted(s, record):
    length = s.len()
    for i in range(length):
        r = s.get(i)
        for node in r.node_list:
            record[node] = 1


def update_nodes_pm_cost(s, nodes_pm_pos, nodes_pm_cost, unrouted_nodes, data):
    if len(unrouted_nodes) == 0:
        return

    if _cuda_batches_enabled(data):
        tmp_routes = [[data.DC, node, data.DC] for node in unrouted_nodes]
        tmp_results = evaluate_route_batch(tmp_routes, data)
        for i, (flag, cost) in enumerate(tmp_results):
            if not flag:
                tmp_nl = make_tmp_nl(data)
                tmp_nl.insert(1, unrouted_nodes[i])
                print("Error: Detect not feasible 1-customer route: ", end="")
                for n in tmp_nl:
                    print(n, end="")
                print()
                raise SystemExit(-1)
            nodes_pm_pos[i][0] = 1
            nodes_pm_cost[i][0] = cost

        for r_index in range(s.len()):
            r = s.get(r_index)
            ori_cost = r.cal_cost(data)
            for i in range(len(unrouted_nodes)):
                nodes_pm_pos[i][r_index + 1] = -1
                nodes_pm_cost[i][r_index + 1] = float("inf")

            candidate_routes, candidate_meta = _build_insertion_sequences(r.node_list, unrouted_nodes)
            if len(candidate_routes) == 0:
                continue
            results = evaluate_route_batch(candidate_routes, data)
            for (node_idx, pos), (flag, cost) in zip(candidate_meta, results):
                if not flag:
                    continue
                incur_cost = cost - ori_cost
                if incur_cost - nodes_pm_cost[node_idx][r_index + 1] < -PRECISION:
                    nodes_pm_cost[node_idx][r_index + 1] = incur_cost
                    nodes_pm_pos[node_idx][r_index + 1] = pos
        return

    for i in range(len(unrouted_nodes)):
        single_node_pm_pos = nodes_pm_pos[i]
        single_node_pm_cost = nodes_pm_cost[i]
        node = unrouted_nodes[i]

        tmp_nl = make_tmp_nl(data)
        flag, cost = chk_nl_node_pos_O_n(tmp_nl, node, 1, data)
        if not flag:
            print("Error: Detect not feasible 1-customer route: ", end="")
            for n in tmp_nl:
                print(n, end="")
            print()
            raise SystemExit(-1)
        single_node_pm_pos[0] = 1
        single_node_pm_cost[0] = cost

        for r_index in range(s.len()):
            r = s.get(r_index)
            ori_cost = r.cal_cost(data)
            best_incur_cost = float("inf")
            best_pos = -1
            for pos in range(1, len(r.node_list)):
                flag, cost = chk_nl_node_pos_O_n(r.node_list, node, pos, data)
                if flag:
                    incur_cost = cost - ori_cost
                    if incur_cost - best_incur_cost < -PRECISION:
                        best_incur_cost = incur_cost
                        best_pos = pos
            single_node_pm_pos[r_index + 1] = best_pos
            single_node_pm_cost[r_index + 1] = best_incur_cost


def update_single_node_pm_cost(
    s, nodes_pm_pos, nodes_pm_cost, unrouted_nodes, changed_r_index, inserted, data
):
    r = s.get(changed_r_index)
    ori_cost = r.cal_cost(data)

    if _cuda_batches_enabled(data):
        candidate_routes = []
        route_meta = []
        for i in range(len(unrouted_nodes)):
            if inserted[i]:
                continue
            node = unrouted_nodes[i]
            for pos in range(1, len(r.node_list)):
                candidate_routes.append(r.node_list[:pos] + [node] + r.node_list[pos:])
                route_meta.append((i, pos))
        if len(candidate_routes) > 0:
            results = evaluate_route_batch(candidate_routes, data)
            for i in range(len(unrouted_nodes)):
                if inserted[i]:
                    continue
                nodes_pm_pos[i][changed_r_index + 1] = -1
                nodes_pm_cost[i][changed_r_index + 1] = float("inf")
            for (node_idx, pos), (flag, cost) in zip(route_meta, results):
                if not flag:
                    continue
                incur_cost = cost - ori_cost
                if incur_cost - nodes_pm_cost[node_idx][changed_r_index + 1] < -PRECISION:
                    nodes_pm_cost[node_idx][changed_r_index + 1] = incur_cost
                    nodes_pm_pos[node_idx][changed_r_index + 1] = pos
        return

    for i in range(len(unrouted_nodes)):
        if inserted[i]:
            continue
        single_node_pm_pos = nodes_pm_pos[i]
        single_node_pm_cost = nodes_pm_cost[i]
        node = unrouted_nodes[i]
        best_incur_cost = float("inf")
        best_pos = -1

        for pos in range(1, len(r.node_list)):
            flag, cost = chk_nl_node_pos_O_n(r.node_list, node, pos, data)
            if flag:
                incur_cost = cost - ori_cost
                if incur_cost - best_incur_cost < -PRECISION:
                    best_incur_cost = incur_cost
                    best_pos = pos
        single_node_pm_pos[changed_r_index + 1] = best_pos
        single_node_pm_cost[changed_r_index + 1] = best_incur_cost


def greedy_insertion(s, data):
    num_cus = data.customer_num
    record = [0 for _ in range(num_cus + 1)]
    find_unrouted(s, record)
    unrouted_nodes = []
    for i in range(num_cus + 1):
        if i == data.DC:
            continue
        if record[i] == 0:
            unrouted_nodes.append(i)
    unroute_len = len(unrouted_nodes)
    inserted = [False for _ in range(unroute_len)]

    single_node_pm_pos = [0 for _ in range(data.vehicle.max_num + 1)]
    single_node_pm_cost = [0.0 for _ in range(data.vehicle.max_num + 1)]
    nodes_pm_pos = []
    nodes_pm_cost = []
    for _ in range(unroute_len):
        nodes_pm_pos.append(list(single_node_pm_pos))
        nodes_pm_cost.append(list(single_node_pm_cost))
    update_nodes_pm_cost(s, nodes_pm_pos, nodes_pm_cost, unrouted_nodes, data)

    while unroute_len > 0:
        min_cost = float("inf")
        best_node_index = -1
        best_route_index = -2
        for i in range(len(unrouted_nodes)):
            if inserted[i]:
                continue
            single_node_pm_pos = nodes_pm_pos[i]
            single_node_pm_cost = nodes_pm_cost[i]
            best_incur_cost = single_node_pm_cost[0]
            best_r = -1
            for r_index in range(s.len()):
                if single_node_pm_cost[r_index + 1] - best_incur_cost < -PRECISION:
                    best_incur_cost = single_node_pm_cost[r_index + 1]
                    best_r = r_index
            if best_incur_cost - min_cost < -PRECISION:
                min_cost = best_incur_cost
                best_node_index = i
                best_route_index = best_r
        if min_cost == float("inf"):
            print("Error: Min insertion cost -INFINITY, this should not happen")
            raise SystemExit(-1)

        single_node_pm_pos = nodes_pm_pos[best_node_index]
        node = unrouted_nodes[best_node_index]

        if best_route_index == -1:
            r = Route(data)
            r.node_list.insert(1, node)
            r.update(data)
            s.append(r)
        else:
            r = s.get(best_route_index)
            r.node_list.insert(single_node_pm_pos[best_route_index + 1], node)
            r.update(data)

        inserted[best_node_index] = True
        unroute_len -= 1

        changed_r_index = best_route_index
        if changed_r_index == -1:
            changed_r_index = s.len() - 1
        update_single_node_pm_cost(
            s, nodes_pm_pos, nodes_pm_cost, unrouted_nodes, changed_r_index, inserted, data
        )
    s.cal_cost(data)


def regret_insertion(s, data):
    num_cus = data.customer_num
    record = [0 for _ in range(num_cus + 1)]
    find_unrouted(s, record)
    unrouted_nodes = []
    for i in range(num_cus + 1):
        if i == data.DC:
            continue
        if record[i] == 0:
            unrouted_nodes.append(i)
    unroute_len = len(unrouted_nodes)
    inserted = [False for _ in range(unroute_len)]

    single_node_pm_pos = [0 for _ in range(data.vehicle.max_num + 1)]
    single_node_pm_cost = [0.0 for _ in range(data.vehicle.max_num + 1)]

    nodes_pm_pos = []
    nodes_pm_cost = []
    for _ in range(unroute_len):
        nodes_pm_pos.append(list(single_node_pm_pos))
        nodes_pm_cost.append(list(single_node_pm_cost))
    update_nodes_pm_cost(s, nodes_pm_pos, nodes_pm_cost, unrouted_nodes, data)

    while unroute_len > 0:
        max_regret = -float("inf")
        best_node_index = -1
        best_route_index = -2
        for i in range(len(unrouted_nodes)):
            if inserted[i]:
                continue
            single_node_pm_pos = nodes_pm_pos[i]
            single_node_pm_cost = nodes_pm_cost[i]
            best_incur_cost = single_node_pm_cost[0]
            best_r = -1
            second_incur_cost = float("inf")
            for r_index in range(s.len()):
                if single_node_pm_cost[r_index + 1] - best_incur_cost < -PRECISION:
                    second_incur_cost = best_incur_cost
                    best_incur_cost = single_node_pm_cost[r_index + 1]
                    best_r = r_index
                elif single_node_pm_cost[r_index + 1] - second_incur_cost < -PRECISION:
                    second_incur_cost = single_node_pm_cost[r_index + 1]
            regret = second_incur_cost - best_incur_cost
            if second_incur_cost == float("inf"):
                regret = 0.0
            if regret - max_regret > PRECISION:
                max_regret = regret
                best_node_index = i
                best_route_index = best_r
        if max_regret == -float("inf"):
            print("Error: Max regret -INFINITY, this should not happen")
            raise SystemExit(-1)

        if max_regret == 0.0:
            max_regret = float("inf")
            for i in range(len(unrouted_nodes)):
                if inserted[i]:
                    continue
                single_node_pm_cost = nodes_pm_cost[i]
                if single_node_pm_cost[0] - max_regret < -PRECISION:
                    max_regret = single_node_pm_cost[0]
                    best_node_index = i
                    best_route_index = -1

        single_node_pm_pos = nodes_pm_pos[best_node_index]
        node = unrouted_nodes[best_node_index]

        if best_route_index == -1:
            r = Route(data)
            r.node_list.insert(1, node)
            r.update(data)
            s.append(r)
        else:
            r = s.get(best_route_index)
            r.node_list.insert(single_node_pm_pos[best_route_index + 1], node)
            r.update(data)

        inserted[best_node_index] = True
        unroute_len -= 1

        changed_r_index = best_route_index
        if changed_r_index == -1:
            changed_r_index = s.len() - 1
        update_single_node_pm_cost(
            s, nodes_pm_pos, nodes_pm_cost, unrouted_nodes, changed_r_index, inserted, data
        )
    s.cal_cost(data)


def new_route_insertion(s, data, initial_node=None):
    if initial_node is None:
        num_cus = data.customer_num
        record = [0 for _ in range(num_cus + 1)]
        find_unrouted(s, record)

        unrouted = [[0, 0] for _ in range(data.customer_num)]
        index = 0
        for i in range(num_cus + 1):
            if i != data.DC and record[i] == 0:
                unrouted[index][0] = i
                index += 1
        if index == 0:
            return
        if data.ksize == 1:
            selected = randint(0, index - 1, data.rng)
            node = unrouted[selected][0]
            new_route_insertion(s, data, node)
        else:
            best_cost = float("inf")
            best_s = s.clone()
            for i in range(min(data.ksize, index)):
                selected = randint(0, index - 1 - i, data.rng)
                node = unrouted[selected][0]
                tmp_s = s.clone()
                new_route_insertion(tmp_s, data, node)
                if tmp_s.cost - best_cost < -PRECISION:
                    best_s = tmp_s
                    best_cost = tmp_s.cost
                unrouted[selected] = unrouted[index - 1 - i]
            s.copy_from(best_s)
        return

    score = [0.0 for _ in range(MAX_POINT)]
    score_argrank = [0 for _ in range(MAX_POINT)]
    ties = [0 for _ in range(MAX_POINT)]
    feasible_pos = [False for _ in range(MAX_NODE_IN_ROUTE * MAX_POINT)]

    unrouted_d = data.all_delivery
    unrouted_p = data.all_pickup

    num_cus = data.customer_num
    record = [0 for _ in range(num_cus + 1)]
    find_unrouted(s, record)

    unrouted = [[0, 0] for _ in range(data.customer_num)]
    index = 0
    for i in range(num_cus + 1):
        if i != data.DC and record[i] == 0:
            unrouted[index][0] = i
            index += 1
        elif i != data.DC and record[i] == 1:
            unrouted_d -= data.node[i].delivery
            unrouted_p -= data.node[i].pickup

    while index > 0:
        r = Route(data)
        selected = -1
        if index == data.customer_num:
            first_node = initial_node
            for i in range(index):
                if unrouted[i][0] == first_node:
                    selected = i
                    break
        else:
            selected = randint(0, index - 1, data.rng)
            first_node = unrouted[selected][0]
        index, unrouted_d, unrouted_p = maintain_unrouted(
            selected, first_node, index, unrouted, unrouted_d, unrouted_p, data
        )
        r.node_list.insert(1, first_node)

        while cal_score(feasible_pos, unrouted, score, index, r, unrouted_d, unrouted_p, data):
            argsort(score, score_argrank, index)
            best_score = score[score_argrank[0]]
            ties[0] = score_argrank[0]
            tie_count = 1
            for j in range(1, index):
                if abs(best_score - score[score_argrank[j]]) < -PRECISION:
                    ties[tie_count] = score_argrank[j]
                    tie_count += 1
                else:
                    break
            if tie_count > 1:
                selected = ties[randint(0, tie_count - 1, data.rng)]
            else:
                selected = ties[0]

            node = unrouted[selected][0]
            pos = unrouted[selected][1]
            r.node_list.insert(pos, node)
            index, unrouted_d, unrouted_p = maintain_unrouted(
                selected, node, index, unrouted, unrouted_d, unrouted_p, data
            )
        r.update(data)
        s.append(r)
    s.cal_cost(data)


def two_opt(r1, r2, s, data, m):
    m.delta_cost = float("inf")

    r = s.get(r1)
    n_l = r.node_list
    length = len(n_l)
    if length < 4:
        return
    for start in range(1, length - 2):
        if data.pruning and (
            not data.pm[n_l[start - 1]][n_l[start + 1]]
            or not data.pm[n_l[start + 1]][n_l[start]]
            or not data.pm[n_l[start]][n_l[start + 2]]
        ):
            continue
        if r.gat(start + 1, start).num_cus == INFEASIBLE:
            continue
        TMP_MOVE.r_indice[0] = r1
        TMP_MOVE.r_indice[1] = -2
        TMP_MOVE.len_1 = 3
        TMP_MOVE.seqList_1[0] = Seq(r1, 0, start - 1)
        TMP_MOVE.seqList_1[1] = Seq(r1, start + 1, start)
        TMP_MOVE.seqList_1[2] = Seq(r1, start + 2, length - 1)
        TMP_MOVE.len_2 = 0
        if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
            m.copy_from(TMP_MOVE)


def two_opt_star(r1, r2, s, data, m):
    m.delta_cost = float("inf")

    r_1 = s.get(r1)
    n_l_1 = r_1.node_list
    len_1 = len(n_l_1)

    r_2 = s.get(r2)
    n_l_2 = r_2.node_list
    len_2 = len(n_l_2)
    for pos_1 in range(1, len_1):
        for pos_2 in range(1, len_2):
            if (pos_1 == 1 and pos_2 == 1) or (pos_1 == len_1 - 1 and pos_2 == len_2 - 1):
                continue
            if data.pruning and (
                not data.pm[n_l_1[pos_1 - 1]][n_l_2[pos_2]]
                or not data.pm[n_l_2[pos_2 - 1]][n_l_1[pos_1]]
            ):
                continue
            TMP_MOVE.r_indice[0] = r1
            TMP_MOVE.r_indice[1] = r2
            TMP_MOVE.len_1 = 2
            TMP_MOVE.seqList_1[0] = Seq(r1, 0, pos_1 - 1)
            TMP_MOVE.seqList_1[1] = Seq(r2, pos_2, len_2 - 1)
            TMP_MOVE.len_2 = 2
            TMP_MOVE.seqList_2[0] = Seq(r2, 0, pos_2 - 1)
            TMP_MOVE.seqList_2[1] = Seq(r1, pos_1, len_1 - 1)
            if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
                m.copy_from(TMP_MOVE)


def or_opt_single(r1, r2, s, data, m):
    m.delta_cost = float("inf")
    r = s.get(r1)
    n_l = r.node_list
    length = len(n_l)
    for start in range(1, length - 1):
        for seq_len in range(1, data.or_opt_len + 1):
            end = start + seq_len - 1
            if end >= length - 1:
                continue
            if data.pruning and (not data.pm[n_l[start - 1]][n_l[end + 1]]):
                continue
            for pos in range(1, start):
                if data.pruning and (
                    not data.pm[n_l[pos - 1]][n_l[start]]
                    or not data.pm[n_l[end]][n_l[pos]]
                ):
                    continue
                TMP_MOVE.r_indice[0] = r1
                TMP_MOVE.r_indice[1] = -2
                TMP_MOVE.len_1 = 4
                TMP_MOVE.seqList_1[0] = Seq(r1, 0, pos - 1)
                TMP_MOVE.seqList_1[1] = Seq(r1, start, end)
                TMP_MOVE.seqList_1[2] = Seq(r1, pos, start - 1)
                TMP_MOVE.seqList_1[3] = Seq(r1, end + 1, length - 1)
                TMP_MOVE.len_2 = 0
                if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
                    m.copy_from(TMP_MOVE)
            for pos in range(end + 2, length):
                if data.pruning and (
                    not data.pm[n_l[pos - 1]][n_l[start]] or not data.pm[n_l[end]][n_l[pos]]
                ):
                    continue
                TMP_MOVE.r_indice[0] = r1
                TMP_MOVE.r_indice[1] = -2
                TMP_MOVE.len_1 = 4
                TMP_MOVE.seqList_1[0] = Seq(r1, 0, start - 1)
                TMP_MOVE.seqList_1[1] = Seq(r1, end + 1, pos - 1)
                TMP_MOVE.seqList_1[2] = Seq(r1, start, end)
                TMP_MOVE.seqList_1[3] = Seq(r1, pos, length - 1)
                TMP_MOVE.len_2 = 0
                if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
                    m.copy_from(TMP_MOVE)

            TMP_MOVE.r_indice[0] = r1
            TMP_MOVE.r_indice[1] = -1
            TMP_MOVE.len_1 = 2
            TMP_MOVE.seqList_1[0] = Seq(r1, 0, start - 1)
            TMP_MOVE.seqList_1[1] = Seq(r1, end + 1, length - 1)
            TMP_MOVE.len_2 = 3
            TMP_MOVE.seqList_2[0] = Seq(-1, data.DC, data.DC)
            TMP_MOVE.seqList_2[1] = Seq(r1, start, end)
            TMP_MOVE.seqList_2[2] = Seq(-1, data.DC, data.DC)
            if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
                m.copy_from(TMP_MOVE)


def or_opt_double(r_index_1, r_index_2, s, data, m):
    m.delta_cost = float("inf")
    for i in range(2):
        if i == 0:
            r1 = r_index_1
            r2 = r_index_2
        else:
            r1 = r_index_2
            r2 = r_index_1
        r = s.get(r1)
        n_l = r.node_list
        length = len(n_l)
        for start in range(1, length - 1):
            for seq_len in range(1, data.or_opt_len + 1):
                end = start + seq_len - 1
                if end >= length - 1:
                    continue
                if data.pruning and (not data.pm[n_l[start - 1]][n_l[end + 1]]):
                    continue

                if r1 == r2:
                    continue
                r_2 = s.get(r2)
                n_l_2 = r_2.node_list
                len_2 = len(n_l_2)
                for pos in range(1, len_2):
                    if data.pruning and (
                        not data.pm[n_l_2[pos - 1]][n_l[start]]
                        or not data.pm[n_l[end]][n_l_2[pos]]
                    ):
                        continue
                    TMP_MOVE.r_indice[0] = r1
                    TMP_MOVE.r_indice[1] = r2
                    TMP_MOVE.len_1 = 2
                    TMP_MOVE.seqList_1[0] = Seq(r1, 0, start - 1)
                    TMP_MOVE.seqList_1[1] = Seq(r1, end + 1, length - 1)
                    TMP_MOVE.len_2 = 3
                    TMP_MOVE.seqList_2[0] = Seq(r2, 0, pos - 1)
                    TMP_MOVE.seqList_2[1] = Seq(r1, start, end)
                    TMP_MOVE.seqList_2[2] = Seq(r2, pos, len_2 - 1)
                    if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
                        m.copy_from(TMP_MOVE)


def two_exchange(r1, r2, s, data, m):
    m.delta_cost = float("inf")
    r_1 = s.get(r1)
    n_l_1 = r_1.node_list
    len_1 = len(n_l_1)

    r_2 = s.get(r2)
    n_l_2 = r_2.node_list
    len_2 = len(n_l_2)
    for start_1 in range(1, len_1 - 1):
        for seq_len_1 in range(1, data.exchange_len + 1):
            end_1 = start_1 + seq_len_1 - 1
            if end_1 >= len_1 - 1:
                continue
            for start_2 in range(1, len_2 - 1):
                for seq_len_2 in range(1, data.exchange_len + 1):
                    end_2 = start_2 + seq_len_2 - 1
                    if end_2 >= len_2 - 1:
                        continue
                    if data.pruning and (
                        not data.pm[n_l_1[start_1 - 1]][n_l_2[start_2]]
                        or not data.pm[n_l_2[end_2]][n_l_1[end_1 + 1]]
                        or not data.pm[n_l_2[start_2 - 1]][n_l_1[start_1]]
                        or not data.pm[n_l_1[end_1]][n_l_2[end_2 + 1]]
                    ):
                        continue
                    TMP_MOVE.r_indice[0] = r1
                    TMP_MOVE.r_indice[1] = r2
                    TMP_MOVE.len_1 = 3
                    TMP_MOVE.seqList_1[0] = Seq(r1, 0, start_1 - 1)
                    TMP_MOVE.seqList_1[1] = Seq(r2, start_2, end_2)
                    TMP_MOVE.seqList_1[2] = Seq(r1, end_1 + 1, len_1 - 1)
                    TMP_MOVE.len_2 = 3
                    TMP_MOVE.seqList_2[0] = Seq(r2, 0, start_2 - 1)
                    TMP_MOVE.seqList_2[1] = Seq(r1, start_1, end_1)
                    TMP_MOVE.seqList_2[2] = Seq(r2, end_2 + 1, len_2 - 1)
                    if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
                        m.copy_from(TMP_MOVE)


def removal_from_s(s, flag):
    s_len = s.len()
    for r_index in range(s_len):
        r = s.get(r_index)
        n_l = r.node_list
        i = 0
        while i < len(n_l):
            if flag[n_l[i]] == 1:
                n_l.pop(i)
            else:
                i += 1


def related_removal(s, data):
    selected = randint(0, data.customer_num, data.rng)
    while selected == data.DC:
        selected = randint(0, data.customer_num, data.rng)
    flag = [0 for _ in range(data.customer_num + 1)]
    selected_cus = []
    flag[selected] = 1
    selected_cus.append(selected)
    total_remove = round(data.customer_num * rand(data.destroy_ratio_l, data.destroy_ratio_u, data.rng))
    already_remove = 1
    while already_remove < total_remove:
        ref_cus = selected_cus[randint(0, len(selected_cus) - 1, data.rng)]
        argrank = data.rm_argrank[ref_cus]
        best_two = []
        for i in range(data.customer_num - 1):
            if flag[argrank[i]] == 1:
                continue
            best_two.append(argrank[i])
            if len(best_two) == 2:
                break
        if len(best_two) != 2:
            print("Cound not find not 2 inserted customers in related removal")
            raise SystemExit(-1)
        prob = data.rm[ref_cus][best_two[1]] / (
            data.rm[ref_cus][best_two[0]] + data.rm[ref_cus][best_two[1]]
        )
        if rand(0, 1, data.rng) < prob:
            selected = best_two[0]
        else:
            selected = best_two[1]
        flag[selected] = 1
        selected_cus.append(selected)
        already_remove += 1
    removal_from_s(s, flag)
    s.update(data)


def random_removal(s, data):
    customers = [0 for _ in range(data.customer_num)]
    indice = [0 for _ in range(data.customer_num + 1)]
    count = 0
    for i in range(data.customer_num + 1):
        if i == data.DC:
            continue
        customers[count] = i
        count += 1
    data.rng.shuffle(customers)
    boundray = int(
        round(float(data.customer_num) * rand(data.destroy_ratio_l, data.destroy_ratio_u, data.rng))
    )
    for i in range(boundray + 1):
        indice[customers[i]] = 1
    removal_from_s(s, indice)
    s.update(data)


def apply_move(s, m, data):
    r_indice = [m.r_indice[0]]
    if m.r_indice[1] != -2:
        r_indice.append(m.r_indice[1])

    if r_indice[0] in (-2, -1):
        print("Error: detect -1 or -2 in r_indice[0] in move")
        raise SystemExit(-1)

    r = s.get(r_indice[0])
    target_n_l = []

    for i in range(m.len_1):
        seq = m.seqList_1[i]
        source_n_l = s.get(seq.r_index).node_list
        if seq.start_point <= seq.end_point:
            for index in range(seq.start_point, seq.end_point + 1):
                target_n_l.append(source_n_l[index])
        else:
            for index in range(seq.start_point, seq.end_point - 1, -1):
                target_n_l.append(source_n_l[index])

    if len(r_indice) == 2:
        target_n_l_2 = []
        for i in range(m.len_2):
            seq = m.seqList_2[i]
            if seq.r_index == -1:
                target_n_l_2.append(data.DC)
                continue
            source_n_l = s.get(seq.r_index).node_list
            if seq.start_point <= seq.end_point:
                for index in range(seq.start_point, seq.end_point + 1):
                    target_n_l_2.append(source_n_l[index])
            else:
                for index in range(seq.start_point, seq.end_point - 1, -1):
                    target_n_l_2.append(source_n_l[index])
        if r_indice[1] == -1:
            r_new = Route(data)
            r_new.node_list = target_n_l_2
            r_new.update(data)
            s.append(r_new)
            r_indice[1] = s.len() - 1
        else:
            r_2 = s.get(r_indice[1])
            r_2.node_list = target_n_l_2
            r_2.update(data)

    r.node_list = target_n_l
    r.update(data)

    s.local_update(r_indice)
    return r_indice


def output_move(m):
    print("r_indice: %d%d" % (m.r_indice[0], m.r_indice[1]))
    print("len_1: %d" % m.len_1)
    for i in range(m.len_1):
        print("%d%d%d" % (m.seqList_1[i].r_index, m.seqList_1[i].start_point, m.seqList_1[i].end_point))
    print("len_2: %d" % m.len_2)
    for i in range(m.len_2):
        print("%d%d%d" % (m.seqList_2[i].r_index, m.seqList_2[i].start_point, m.seqList_2[i].end_point))


def snippet(r1, r2, opt, s, data, target):
    m = data.get_mem(opt, r1, r2)
    small_opt_map[opt](r1, r2, s, data, m)
    if m.delta_cost - target.delta_cost < -PRECISION:
        target.copy_from(m)


def find_local_optima(s, data):
    if data.skip_finding_lo:
        return
    move_list = [Move() for _ in range(len(data.small_opts))]

    length = s.len()
    for i in range(len(move_list)):
        move_list[i].delta_cost = float("inf")
        opt = data.small_opts[i]
        if opt in ("2opt", "oropt_single"):
            for r in range(length):
                snippet(r, -1, opt, s, data, move_list[i])
        elif opt in ("2opt*", "2exchange", "oropt_double"):
            for r1 in range(length):
                for r2 in range(r1 + 1, length):
                    snippet(r1, r2, opt, s, data, move_list[i])
        else:
            print("Unknown opt: %s" % opt)
            raise SystemExit(-1)

    tour_id_array = []
    while True:
        best_index = -1
        min_delta_cost = float("inf")
        for i in range(len(move_list)):
            if move_list[i].delta_cost - min_delta_cost < -PRECISION:
                best_index = i
                min_delta_cost = move_list[i].delta_cost
        if min_delta_cost < -PRECISION:
            tour_id_array = apply_move(s, move_list[best_index], data)

            length = s.len()
            for i in range(len(move_list)):
                move_list[i].delta_cost = float("inf")
                opt = data.small_opts[i]
                if opt in ("2opt", "oropt_single"):
                    for r in tour_id_array:
                        if r >= length:
                            continue
                        snippet(r, -1, opt, s, data, move_list[i])
                    for r in range(length):
                        if data.get_mem(opt, r, -1).delta_cost - move_list[i].delta_cost < -PRECISION:
                            move_list[i].copy_from(data.get_mem(opt, r, -1))
                elif opt in ("2opt*", "2exchange", "oropt_double"):
                    for r in tour_id_array:
                        if r >= length:
                            continue
                        for r1 in range(r):
                            snippet(r1, r, opt, s, data, move_list[i])
                        for r1 in range(r + 1, length):
                            snippet(r, r1, opt, s, data, move_list[i])
                    for r1 in range(length):
                        for r2 in range(r1 + 1, length):
                            if data.get_mem(opt, r1, r2).delta_cost - move_list[i].delta_cost < -PRECISION:
                                move_list[i].copy_from(data.get_mem(opt, r1, r2))
                else:
                    print("Unknown opt: %s" % opt)
                    raise SystemExit(-1)
        else:
            break


def _local_search_worker(task):
    index, s, data = task
    find_local_optima(s, data)
    s.cal_cost(data)
    return index, s


def do_local_search(s, data, executor=None):
    if len(data.small_opts) == 0:
        print("No small stepsize operator used, directly return.")
        return
    if data.escape_local_optima == -1:
        return

    find_local_optima(s, data)
    s.cal_cost(data)
    if data.escape_local_optima == 0:
        return

    tmp_solution_num = len(data.destroy_opts) * len(data.repair_opts)
    if data.rd_removal_insertion:
        tmp_solution_num = 1
    s_vector = [s.clone() for _ in range(tmp_solution_num)]

    no_improve = 0
    while no_improve < data.escape_local_optima:
        for i in range(tmp_solution_num):
            s_vector[i] = s.clone()
        perturb(s_vector, data)
        best_index = -1
        best_cost = float("inf")
        if executor is not None and tmp_solution_num > 1:
            tasks = [(i, s_vector[i], data) for i in range(tmp_solution_num)]
            results = list(executor.map(_local_search_worker, tasks))
            for i, new_s in results:
                s_vector[i].copy_from(new_s)
                if new_s.cost - best_cost < -PRECISION:
                    best_index = i
                    best_cost = new_s.cost
        else:
            for i in range(tmp_solution_num):
                find_local_optima(s_vector[i], data)
                s_vector[i].cal_cost(data)
                if s_vector[i].cost - best_cost < -PRECISION:
                    best_index = i
                    best_cost = s_vector[i].cost
        if s_vector[best_index].cost - s.cost < -PRECISION:
            s.copy_from(s_vector[best_index])
            no_improve = 0
        else:
            no_improve += 1


def perturb(s_vector, data):
    if data.rd_removal_insertion:
        i = randint(0, len(data.destroy_opts) - 1, data.rng)
        j = randint(0, len(data.repair_opts) - 1, data.rng)
        destroy_opt_map[data.destroy_opts[i]](s_vector[0], data)
        repair_opt_map[data.repair_opts[j]](s_vector[0], data)
        return
    count = 0
    for i in range(len(data.destroy_opts)):
        for j in range(len(data.repair_opts)):
            destroy_opt_map[data.destroy_opts[i]](s_vector[count], data)
            repair_opt_map[data.repair_opts[j]](s_vector[count], data)
            count += 1


small_opt_map = {
    "2opt": two_opt,
    "2opt*": two_opt_star,
    "oropt_single": or_opt_single,
    "oropt_double": or_opt_double,
    "2exchange": two_exchange,
}

destroy_opt_map = {
    "random_removal": random_removal,
    "related_removal": related_removal,
}

repair_opt_map = {
    "regret_insertion": regret_insertion,
    "greedy_insertion": greedy_insertion,
}
