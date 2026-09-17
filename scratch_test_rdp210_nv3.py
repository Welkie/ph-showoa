import sys
sys.path.insert(0, ".")
import random
from src_python_gpu_SA_RCRS_GRASP.data import Data
from src_python_gpu_SA_RCRS_GRASP.argparse_util import ArgumentParser
from scratch_test_cpp_exact import rcrs_grasp_init_cpp_exact
from scratch_test_cpp_init import sa_post_refinement

def test_rdp210_sa_3vehicles():
    parser = ArgumentParser()
    parser.add_argument("--problem", 1, False)
    parser.parse(["test", "dataset/explicit_rdp210.vrpsdptw"])
    data = Data(parser)
    
    print("Testing rdp210 initialization + SA post-refinement over 30 individuals (P=30)...")
    found_nv3 = False
    for i in range(30):
        alpha = random.uniform(0.1, 0.4)
        routes = rcrs_grasp_init_cpp_exact(data, alpha)
        nv_before = len(routes)
        routes = sa_post_refinement(routes, data, itermax=50)
        nv_after = len(routes)
        td = sum(sum(data.dist[r[j]][r[j+1]] for j in range(len(r)-1)) for r in routes)
        cost = nv_after * 2000.0 + td
        print(f"Ind {i}: NV before={nv_before}, NV after={nv_after}, TD={td:.2f}, Cost={cost:.2f}")
        if nv_after == 3:
            found_nv3 = True
            print(f"--> FOUND NV=3 on individual {i}!")
            break
            
    print("Result:", "SUCCESS NV=3 FOUND!" if found_nv3 else "No NV=3 yet.")

if __name__ == "__main__":
    test_rdp210_sa_3vehicles()
