import random
import math
import time
import torch
from src_python_gpu_SA_RCRS_GRASP.data import Data
from src_python_gpu_SA_RCRS_GRASP.argparse_util import ArgumentParser
from src_python_gpu_SA_RCRS_GRASP.compute_backend import create_backend
from src_python_gpu_SA_RCRS_GRASP.eval import _chk_route_list

def rcrs_score(nl, data, c, pos, w_td=1.0, w_rc=0.5, w_rs=0.3, rc_thr=0.70):
    prev, nxt = nl[pos - 1], nl[pos]
    d_prev_c = data.dist[prev][c]
    d_c_nxt = data.dist[c][nxt]
    d_prev_nxt = data.dist[prev][nxt]
    delta_td = max(0.0, d_prev_c + d_c_nxt - d_prev_nxt)
    
    # RC penalty approximation
    cap = data.vehicle.capacity
    tot_load = sum(data.node[n].delivery for n in nl if n != data.DC) + max(data.node[c].delivery, data.node[c].pickup)
    rc_penalty = max(0.0, tot_load - cap * rc_thr)
    
    # RS penalty
    dx_prev = data.dist[data.DC][prev]
    dx_c = data.dist[data.DC][c]
    rs_penalty = abs(dx_prev + d_prev_c - dx_c)
    
    return w_td * delta_td + w_rc * rc_penalty + w_rs * rs_penalty

def rcrs_grasp_init_py(data, alpha):
    depot = data.DC
    unrouted = [i for i in range(1, data.customer_num + 1) if i != depot]
    random.shuffle(unrouted)
    routes = []
    
    while unrouted:
        best_per_customer = []
        global_best_score = float('inf')
        max_score = float('-inf')
        
        # To make it fast, evaluate candidates
        cand_unrouted = unrouted if len(unrouted) <= 40 else random.sample(unrouted, 40)
        
        for c in cand_unrouted:
            best_c = {'customer': c, 'r_idx': -1, 'pos': -1, 'score': float('inf')}
            for r_idx, r in enumerate(routes):
                for pos in range(1, len(r)):
                    cand_nl = r[:pos] + [c] + r[pos:]
                    flag, _ = _chk_route_list(cand_nl, data)
                    if not flag:
                        continue
                    score = rcrs_score(r, data, c, pos)
                    if score < best_c['score']:
                        best_c['score'] = score
                        best_c['r_idx'] = r_idx
                        best_c['pos'] = pos
                        
            if best_c['r_idx'] != -1:
                if best_c['score'] < global_best_score:
                    global_best_score = best_c['score']
                if best_c['score'] > max_score:
                    max_score = best_c['score']
            best_per_customer.append(best_c)
            
        rcl = []
        forced = []
        threshold = float('inf') if (global_best_score == float('inf') or max_score == float('-inf')) else global_best_score + alpha * (max_score - global_best_score)
        
        for item in best_per_customer:
            if item['r_idx'] == -1:
                forced.append(item)
            elif item['score'] <= threshold + 1e-6:
                rcl.append(item)
                
        if not rcl:
            # Pick a customer from forced (or random unrouted) and open new route
            c = random.choice(forced)['customer'] if forced else random.choice(unrouted)
            routes.append([depot, c, depot])
            unrouted.remove(c)
        else:
            chosen = random.choice(rcl)
            routes[chosen['r_idx']].insert(chosen['pos'], chosen['customer'])
            unrouted.remove(chosen['customer'])
            
    return routes

def route_cost(nl, data):
    if len(nl) <= 2:
        return 0.0
    dist = sum(data.dist[nl[i]][nl[i+1]] for i in range(len(nl) - 1))
    return data.vehicle.d_cost + dist * data.vehicle.unit_cost

def total_sol_cost(routes, data):
    return sum(route_cost(r, data) for r in routes if len(r) > 2)

def sa_post_refinement(routes, data, itermax=50):
    t0 = 100.0
    alpha = 0.95
    tmin = 0.1
    t = t0
    
    current_routes = [list(r) for r in routes if len(r) > 2]
    best_routes = [list(r) for r in current_routes]
    best_cost = total_sol_cost(best_routes, data)
    curr_cost = best_cost
    
    while t > tmin:
        for _ in range(itermax):
            move_type = random.randint(1, 5)
            if move_type <= 3 and current_routes:
                r_idx = random.randint(0, len(current_routes) - 1)
                nl = list(current_routes[r_idx])
                if len(nl) < 4:
                    continue
                if move_type == 1: # swap
                    i1, i2 = random.randint(1, len(nl) - 2), random.randint(1, len(nl) - 2)
                    nl[i1], nl[i2] = nl[i2], nl[i1]
                elif move_type == 2: # insert
                    i1 = random.randint(1, len(nl) - 2)
                    val = nl.pop(i1)
                    i2 = random.randint(1, len(nl) - 1)
                    nl.insert(i2, val)
                elif move_type == 3: # reverse
                    i1, i2 = random.randint(1, len(nl) - 2), random.randint(1, len(nl) - 2)
                    if i1 > i2: i1, i2 = i2, i1
                    nl[i1:i2+1] = reversed(nl[i1:i2+1])
                    
                flag, _ = _chk_route_list(nl, data)
                if flag:
                    old_rc = route_cost(current_routes[r_idx], data)
                    new_rc = route_cost(nl, data)
                    delta = new_rc - old_rc
                    if delta < 0 or random.random() < math.exp(-delta / (1e-6 + t * max(1.0, abs(curr_cost)))):
                        current_routes[r_idx] = nl
                        curr_cost += delta
                        if curr_cost < best_cost:
                            best_cost = curr_cost
                            best_routes = [list(r) for r in current_routes if len(r) > 2]
            elif move_type in (4, 5) and len(current_routes) >= 2:
                r1 = random.randint(0, len(current_routes) - 1)
                r2 = random.randint(0, len(current_routes) - 1)
                while r1 == r2:
                    r2 = random.randint(0, len(current_routes) - 1)
                nl1 = list(current_routes[r1])
                nl2 = list(current_routes[r2])
                
                if move_type == 4: # Pd-Shift: shift customer from r1 to r2
                    if len(nl1) < 3:
                        continue
                    i1 = random.randint(1, len(nl1) - 2)
                    val = nl1.pop(i1)
                    # Try best insertion position in nl2!
                    best_pos2 = -1
                    best_c2 = float('inf')
                    for p2 in range(1, len(nl2)):
                        cand2 = nl2[:p2] + [val] + nl2[p2:]
                        flag2, _ = _chk_route_list(cand2, data)
                        if flag2:
                            c2 = route_cost(cand2, data)
                            if c2 < best_c2:
                                best_c2 = c2
                                best_pos2 = p2
                    if best_pos2 != -1:
                        nl2.insert(best_pos2, val)
                    else:
                        continue
                else: # Pd-Exchange
                    if len(nl1) < 3 or len(nl2) < 3:
                        continue
                    i1 = random.randint(1, len(nl1) - 2)
                    i2 = random.randint(1, len(nl2) - 2)
                    nl1[i1], nl2[i2] = nl2[i2], nl1[i1]
                    
                flag1, _ = _chk_route_list(nl1, data)
                flag2, _ = _chk_route_list(nl2, data)
                if flag1 and flag2:
                    old_c = route_cost(current_routes[r1], data) + route_cost(current_routes[r2], data)
                    new_c = route_cost(nl1, data) + route_cost(nl2, data)
                    delta = new_c - old_c
                    if delta < 0 or random.random() < math.exp(-delta / (1e-6 + t * max(1.0, abs(curr_cost)))):
                        current_routes[r1] = nl1
                        current_routes[r2] = nl2
                        # Delete empty routes if any
                        current_routes = [r for r in current_routes if len(r) > 2]
                        curr_cost = total_sol_cost(current_routes, data)
                        if curr_cost < best_cost:
                            best_cost = curr_cost
                            best_routes = [list(r) for r in current_routes]
        t *= alpha
        
    return best_routes

def test(prob_path):
    parser = ArgumentParser()
    parser.add_argument("--problem", 1, False)
    parser.parse(["test", prob_path])
    data = Data(parser)
    print(f"\nTesting {prob_path} ({data.customer_num} customers)")
    
    best_nv = 999
    best_td = 999999.0
    t0 = time.time()
    
    for i in range(5):
        alpha = random.uniform(0.1, 0.4)
        routes = rcrs_grasp_init_py(data, alpha)
        routes = sa_post_refinement(routes, data, itermax=25)
        nv = len(routes)
        td = sum(sum(data.dist[r[j]][r[j+1]] for j in range(len(r)-1)) for r in routes)
        if nv < best_nv or (nv == best_nv and td < best_td):
            best_nv = nv
            best_td = td
        print(f"  Sample {i}: NV = {nv}, TD = {td:.2f}")
        
    print(f"--> Best after 5 init samples in {time.time()-t0:.2f}s: NV = {best_nv}, TD = {best_td:.2f}")

if __name__ == "__main__":
    test("dataset/explicit_rcdp5004.vrpsdptw")
    test("dataset/explicit_rcdp205.vrpsdptw")
    test("dataset/explicit_rdp210.vrpsdptw")
