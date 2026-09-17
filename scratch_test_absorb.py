import sys
sys.path.insert(0, ".")
import torch
import random
from src_python_gpu_SA_RCRS_GRASP.data import Data
from src_python_gpu_SA_RCRS_GRASP.argparse_util import ArgumentParser
from src_python_gpu_SA_RCRS_GRASP.compute_backend import create_backend
from src_python_gpu_SA_RCRS_GRASP.eval import _chk_route_list
from src_python_gpu_SA_RCRS_GRASP.operator import optimize_route_nodes_2opt

def test_rcdp205_4vehicles():
    parser = ArgumentParser()
    parser.add_argument("--problem", 1, False)
    parser.parse(["test", "dataset/explicit_rcdp205.vrpsdptw"])
    data = Data(parser)
    depot = data.DC
    
    # We saw in C++ that rcdp205 has NV=4.
    # Total customers = 100.
    # Let's see if 4 routes can be formed.
    from scratch_test_cpp_init import rcrs_grasp_init_py, sa_post_refinement
    
    print("Testing absorption of 5 routes into 4 routes on rcdp205...")
    for trial in range(20):
        alpha = random.uniform(0.1, 0.4)
        routes = rcrs_grasp_init_py(data, alpha)
        routes = sa_post_refinement(routes, data, itermax=50)
        
        if len(routes) == 5:
            # Try to absorb the shortest route
            routes.sort(key=lambda r: len(r))
            victim = routes[0]
            victim_custs = victim[1:-1]
            others = [list(r) for r in routes[1:]]
            
            print(f"Trial {trial}: NV=5, victim route has {len(victim_custs)} customers: lengths={[len(r)-2 for r in routes]}")
            
            # Sort victim customers by tightest window first
            victim_custs.sort(key=lambda c: (data.node[c].end - data.node[c].start, -(data.node[c].delivery + data.node[c].pickup)))
            
            temp_others = [list(r) for r in others]
            success = True
            for c in victim_custs:
                best_r = -1
                best_pos = -1
                best_d = float('inf')
                for r_i, r in enumerate(temp_others):
                    for pos in range(1, len(r)):
                        cand = r[:pos] + [c] + r[pos:]
                        flag, _ = _chk_route_list(cand, data)
                        if not flag:
                            # Try 2-opt untangling
                            cand_opt = optimize_route_nodes_2opt(cand, data)
                            flag, _ = _chk_route_list(cand_opt, data)
                            if flag:
                                cand = cand_opt
                        if flag:
                            dist = sum(data.dist[cand[j]][cand[j+1]] for j in range(len(cand)-1))
                            if dist < best_d:
                                best_d = dist
                                best_r = r_i
                                best_cand = cand
                if best_r != -1:
                    temp_others[best_r] = best_cand
                else:
                    success = False
                    break
            if success:
                td = sum(sum(data.dist[r[j]][r[j+1]] for j in range(len(r)-1)) for r in temp_others)
                print(f"--> SUCCESS! Found NV = 4, TD = {td:.2f} on trial {trial}!")
                return
    print("Absorption did not complete in 20 trials without search.")

if __name__ == "__main__":
    test_rcdp205_4vehicles()
