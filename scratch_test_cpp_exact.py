import sys
sys.path.insert(0, ".")
import random
import math
from src_python_gpu_SA_RCRS_GRASP.data import Data
from src_python_gpu_SA_RCRS_GRASP.argparse_util import ArgumentParser
from src_python_gpu_SA_RCRS_GRASP.eval import _chk_route_list

def rcrs_score_cpp(r_nl, data, c, pos, w_td=1.0, w_rc=0.5, w_rs=0.3, rc_thr=0.70):
    prev = r_nl[pos - 1]
    next_node = r_nl[pos]
    delta_td = max(0.0, data.dist[prev][c] + data.dist[c][next_node] - data.dist[prev][next_node])
    
    # C_H_new approx
    c_h_new = sum(data.node[n].delivery for n in r_nl if n != data.DC) + max(data.node[c].delivery, data.node[c].pickup)
    cap = data.vehicle.capacity
    rc_penalty = max(0.0, c_h_new - cap * rc_thr)
    
    dx_prev = data.dist[data.DC][prev]
    dx_c = data.dist[data.DC][c]
    rs_penalty = abs(dx_prev + data.dist[prev][c] - dx_c)
    
    return w_td * delta_td + w_rc * rc_penalty + w_rs * rs_penalty

def rcrs_grasp_init_cpp_exact(data, alpha):
    depot = data.DC
    unrouted = [i for i in range(1, data.customer_num + 1) if i != depot]
    random.shuffle(unrouted)
    
    routes = []
    
    while unrouted:
        best_per_customer = []
        global_best_score = float('inf')
        max_score = float('-inf')
        
        for c in unrouted:
            best_c = {'customer': c, 'r_idx': -1, 'pos': -1, 'score': float('inf')}
            for r_idx, r in enumerate(routes):
                for pos in range(1, len(r)):
                    cand = r[:pos] + [c] + r[pos:]
                    flag, _ = _chk_route_list(cand, data)
                    if not flag:
                        continue
                    score = rcrs_score_cpp(r, data, c, pos)
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
            
        threshold = float('inf') if (global_best_score == float('inf') or max_score == float('-inf')) else global_best_score + alpha * (max_score - global_best_score)
        
        rcl = []
        forced = []
        for item in best_per_customer:
            if item['r_idx'] == -1:
                forced.append(item)
            elif item['score'] <= threshold + 1e-9:
                rcl.append(item)
                
        if not rcl:
            if not forced:
                break
            pick_forced = random.choice(forced)
            c = pick_forced['customer']
            routes.append([depot, c, depot])
            unrouted.remove(c)
        else:
            chosen = random.choice(rcl)
            routes[chosen['r_idx']].insert(chosen['pos'], chosen['customer'])
            unrouted.remove(chosen['customer'])
            
    return routes

def test_rdp210():
    parser = ArgumentParser()
    parser.add_argument("--problem", 1, False)
    parser.parse(["test", "dataset/explicit_rdp210.vrpsdptw"])
    data = Data(parser)
    
    print("Running exact C++ RCRS-GRASP initialization on rdp210...")
    nv_counts = []
    for i in range(10):
        alpha = random.uniform(0.1, 0.4)
        routes = rcrs_grasp_init_cpp_exact(data, alpha)
        nv = len(routes)
        td = sum(sum(data.dist[r[j]][r[j+1]] for j in range(len(r)-1)) for r in routes)
        print(f"Sample {i}: NV = {nv}, TD = {td:.2f}")
        nv_counts.append(nv)
        
    print(f"Min NV: {min(nv_counts)}, Counts: {nv_counts}")

if __name__ == "__main__":
    test_rdp210()
