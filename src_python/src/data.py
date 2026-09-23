import math
import random
from dataclasses import dataclass
from typing import List

from .argparse_util import ArgumentParser
from .config import (
    BENCHMARKING_O_1_EVAL,
    CIRCLE,
    DEFAULT_2_EX,
    DEFAULT_2_OPT,
    DEFAULT_2_OPT_STAR,
    DEFAULT_ALPHA,
    DEFAULT_CROSSOVER,
    DEFAULT_DESTROY_RATIO_L,
    DEFAULT_DESTROY_RATIO_U,
    DEFAULT_ELO,
    DEFAULT_GD_INSERTION,
    DEFAULT_IF_OUTPUT,
    DEFAULT_INIT,
    DEFAULT_LS_PROB,
    DEFAULT_MAX_ITER,
    DEFAULT_NO_CROSSOVER,
    DEFAULT_O_1_EVAL,
    DEFAULT_OR_OPT,
    DEFAULT_OR_OPT_LEN,
    DEFAULT_PARALLEL_WORKERS,
    DEFAULT_PRUNING,
    DEFAULT_RD_REMOVAL,
    DEFAULT_RD_R_I,
    DEFAULT_RG_INSERTION,
    DEFAULT_RT_REMOVAL,
    DEFAULT_SEED,
    DEFAULT_DIVERSIFY_RATIO,
    DEFAULT_COMPUTE_BACKEND,
    DEFAULT_HYBRID_MODE,
    DEFAULT_LOCAL_SEARCH_INTERVAL,
    DEFAULT_SHO_MUTATION_PROB,
    DEFAULT_STAGNATION_INTERVAL,
    DEFAULT_SKIP_FINDING_LO,
    DEFAUTL_EX_LEN,
    G_1,
    INF,
    K,
    HYBRID_MODE_PH_SHOWOA,
    HYBRID_MODE_SHO,
    HYBRID_MODE_WOA,
    NO_LIMIT,
    OUTPUT_PER_GENS,
    P_SIZE,
    PENALTY_FACTOR,
    PRECISION,
    RCRS,
    RCRS_RANDOM,
    RDSELECTION,
    REGRET,
    RUNS,
    TD,
    TOURNAMENT,
    V_NUM_RELAX,
)
from .move import Move
from .util import argsort, chk_p_square, rand, randint, split, trim
from .compute_backend import create_backend


@dataclass
class Point:
    id: int
    pickup: float
    delivery: float
    s_time: float
    start: float
    end: float


@dataclass
class Vehicle:
    type: int = 0
    capacity: float = 0.0
    max_num: int = 0
    unit_cost: float = 0.0
    d_cost: float = 0.0


class Data:
    def __init__(self, parser: ArgumentParser) -> None:
        self.problem_name = ""
        self.node: List[Point] = []
        self.customer_num = 0
        self.dist: List[List[float]] = []
        self.time: List[List[float]] = []
        self.rm: List[List[float]] = []
        self.rm_argrank: List[List[int]] = []
        self.pm: List[List[bool]] = []

        self.vehicle = Vehicle()
        self.max_dist = 0.0
        self.min_dist = float("inf")
        self.all_pickup = 0.0
        self.all_delivery = 0.0
        self.start_time = 0.0
        self.end_time = 0.0

        self.DC = 0

        self.pruning = DEFAULT_PRUNING
        self.O_1_evl = DEFAULT_O_1_EVAL
        self.no_crossover = DEFAULT_NO_CROSSOVER
        self.if_output = DEFAULT_IF_OUTPUT
        self.output = " "
        self.tmax = NO_LIMIT
        self.g_1 = G_1
        self.max_iter = DEFAULT_MAX_ITER
        self.runs = RUNS
        self.p_size = P_SIZE
        self.parallel_workers = DEFAULT_PARALLEL_WORKERS
        self.local_search_interval = DEFAULT_LOCAL_SEARCH_INTERVAL
        self.stagnation_interval = DEFAULT_STAGNATION_INTERVAL
        self.diversify_ratio = DEFAULT_DIVERSIFY_RATIO
        self.sho_mutation_prob = DEFAULT_SHO_MUTATION_PROB
        self.hybrid_mode = DEFAULT_HYBRID_MODE
        self.compute_backend = DEFAULT_COMPUTE_BACKEND
        self.k_init = K
        self.k_crossover = K
        self.ksize = 0
        self.seed = DEFAULT_SEED
        self.bks = -1.0
        self.rng = random.Random()
        self.init = DEFAULT_INIT
        self.cross_repair = DEFAULT_CROSSOVER
        self.lambda_gamma = (0.0, 0.0)
        self.latin = []
        self.n_insert = ""

        self.selection = CIRCLE
        self.replacement = "one_on_one"

        self.ls_prob = DEFAULT_LS_PROB
        self.skip_finding_lo = DEFAULT_SKIP_FINDING_LO
        self.mem = {}
        self.two_opt = DEFAULT_2_OPT
        self.mem_2opt = []
        self.two_opt_star = DEFAULT_2_OPT_STAR
        self.mem_2optstar = []
        self.or_opt = DEFAULT_OR_OPT
        self.mem_oropt_single = []
        self.mem_oropt_double = []
        self.two_exchange = DEFAULT_2_EX
        self.mem_2ex = []

        self.backend = None

        self.or_opt_len = DEFAULT_OR_OPT_LEN
        self.exchange_len = DEFAUTL_EX_LEN

        self.escape_local_optima = DEFAULT_ELO
        self.destroy_ratio_l = DEFAULT_DESTROY_RATIO_L
        self.destroy_ratio_u = DEFAULT_DESTROY_RATIO_U
        self.random_removal = DEFAULT_RD_REMOVAL
        self.related_removal = DEFAULT_RT_REMOVAL
        self.greedy_insertion = DEFAULT_GD_INSERTION
        self.regret_insertion = DEFAULT_RG_INSERTION
        self.alpha = DEFAULT_ALPHA
        self.r = 0.0
        self.rd_removal_insertion = DEFAULT_RD_R_I

        self.small_opts = []
        self.destroy_opts = []
        self.repair_opts = []

        pro_file = parser.retrieve("problem")
        with open(pro_file, "r", encoding="utf-8") as fp:
            lines = fp.readlines()

        all_pickup = 0.0
        all_delivery = 0.0
        all_dist = 0.0
        all_time = 0.0

        i = 0
        while i < len(lines):
            line = trim(lines[i])
            if len(line) == 0:
                i += 1
                continue
            results = split(line, ":")
            key = trim(results[0])
            value = trim(results[1]) if len(results) > 1 else ""

            if key == "NAME":
                print(line)
                self.problem_name = value
            elif key == "TYPE":
                print(line)
            elif key == "DIMENSION":
                print(line)
                self.customer_num = int(value) - 1
                tmp_v_1 = [0.0 for _ in range(self.customer_num + 1)]
                tmp_v_2 = [False for _ in range(self.customer_num + 1)]
                tmp_v_3 = [0 for _ in range(self.customer_num + 1)]
                for _ in range(self.customer_num + 1):
                    self.node.append(Point(0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    self.dist.append(list(tmp_v_1))
                    self.time.append(list(tmp_v_1))
                    self.rm.append(list(tmp_v_1))
                    self.rm_argrank.append(list(tmp_v_3))
                    self.pm.append(list(tmp_v_2))
            elif key == "VEHICLES":
                print(line)
                self.vehicle.max_num = int(value) + V_NUM_RELAX
            elif key == "DISPATCHINGCOST":
                print(line)
                self.vehicle.d_cost = float(value)
            elif key == "UNITCOST":
                print(line)
                self.vehicle.unit_cost = float(value)
            elif key == "CAPACITY":
                print(line)
                self.vehicle.capacity = float(value)
            elif key == "EDGE_WEIGHT_TYPE":
                print(line)
                if value != "EXPLICIT":
                    print(
                        "Expect edge weight type: EXPLICIT, while accept type: %s"
                        % value
                    )
                    raise SystemExit(-1)
            elif key == "NODE_SECTION":
                i += 1
                while i < len(lines):
                    line = trim(lines[i])
                    if len(line) == 0:
                        i += 1
                        continue
                    r = split(line, ",")
                    if len(r) > 1:
                        idx = int(trim(r[0]))
                        self.node[idx].delivery = float(trim(r[1]))
                        all_delivery += self.node[idx].delivery
                        self.node[idx].pickup = float(trim(r[2]))
                        all_pickup += self.node[idx].pickup
                        self.node[idx].start = float(trim(r[3]))
                        self.node[idx].end = float(trim(r[4]))
                        self.node[idx].s_time = float(trim(r[5]))
                        i += 1
                    else:
                        break
                continue
            elif key == "DISTANCETIME_SECTION":
                i += 1
                while i < len(lines):
                    line = trim(lines[i])
                    if len(line) == 0:
                        i += 1
                        continue
                    r = split(line, ",")
                    if len(r) > 1:
                        idx_i = int(trim(r[0]))
                        idx_j = int(trim(r[1]))
                        d_val = float(trim(r[2]))
                        t_val = float(trim(r[3]))
                        self.dist[idx_i][idx_j] = d_val
                        all_dist += d_val
                        self.time[idx_i][idx_j] = t_val
                        all_time += t_val
                        if d_val < self.min_dist:
                            self.min_dist = d_val
                        if d_val > self.max_dist:
                            self.max_dist = d_val
                        i += 1
                    else:
                        break
                continue
            elif key == "DEPOT_SECTION":
                i += 1
                if i < len(lines):
                    line = trim(lines[i])
                    self.DC = int(line)
            i += 1

        self.start_time = self.node[self.DC].start
        self.end_time = self.node[self.DC].end
        self.all_delivery = all_delivery
        self.all_pickup = all_pickup

        print(
            "Avg pick-up/dilvery demand: %.4f,%.4f"
            % (self.all_pickup / self.customer_num, self.all_delivery / self.customer_num)
        )
        print(
            "Starting/end time of DC: %.4f,%.4f"
            % (self.start_time, self.end_time)
        )
        print()

        if parser.exists("random_seed"):
            self.seed = int(parser.retrieve("random_seed"))
        self.rng.seed(self.seed)
        print("Initial random seed: %d" % self.seed)

        if parser.exists("pruning"):
            print("Pruning: on")
            self.pruning = True
        else:
            print("Pruning: off")

        if parser.exists("output"):
            self.if_output = True
            self.output = parser.retrieve("output")
            print("Write best solution to %s" % self.output)

        if parser.exists("time"):
            self.tmax = int(parser.retrieve("time"))
        print("Time limit: %d seconds" % self.tmax)

        if parser.exists("runs"):
            self.runs = int(parser.retrieve("runs"))
        print("Runs: %d" % self.runs)

        if parser.exists("g_1"):
            self.g_1 = int(parser.retrieve("g_1"))
        print("g_1: %d" % self.g_1)

        self.max_iter = self.g_1
        if parser.exists("max_iter"):
            self.max_iter = int(parser.retrieve("max_iter"))
        print("Max PH-SHOWOA iterations: %d" % self.max_iter)

        if parser.exists("pop_size"):
            self.p_size = int(parser.retrieve("pop_size"))
        print("Population size: %d" % self.p_size)

        if parser.exists("workers"):
            self.parallel_workers = int(parser.retrieve("workers"))
        if self.parallel_workers <= 0:
            print("Parallel workers: all available CPU cores")
        else:
            print("Parallel workers: %d" % self.parallel_workers)
        if parser.exists("local_search_interval"):
            self.local_search_interval = int(parser.retrieve("local_search_interval"))
        if self.local_search_interval <= 0:
            print("Expect local_search_interval to be positive")
            raise SystemExit(-1)
        print("Periodic local-search interval: %d" % self.local_search_interval)

        if parser.exists("stagnation_interval"):
            self.stagnation_interval = int(parser.retrieve("stagnation_interval"))
        if self.stagnation_interval <= 0:
            print("Expect stagnation_interval to be positive")
            raise SystemExit(-1)
        print("Diversification stagnation interval: %d" % self.stagnation_interval)

        if parser.exists("diversify_ratio"):
            self.diversify_ratio = float(parser.retrieve("diversify_ratio"))
        if self.diversify_ratio < 0.0 or self.diversify_ratio > 1.0:
            print("Expect diversify_ratio in [0, 1]")
            raise SystemExit(-1)
        print("Diversification ratio: %.2f" % self.diversify_ratio)

        if parser.exists("sho_mutation_prob"):
            self.sho_mutation_prob = float(parser.retrieve("sho_mutation_prob"))
        if self.sho_mutation_prob < 0.0 or self.sho_mutation_prob > 1.0:
            print("Expect sho_mutation_prob in [0, 1]")
            raise SystemExit(-1)
        print("SHO mutation probability: %.2f" % self.sho_mutation_prob)

        if parser.exists("hybrid_mode"):
            self.hybrid_mode = parser.retrieve("hybrid_mode")
        if self.hybrid_mode not in (HYBRID_MODE_PH_SHOWOA, HYBRID_MODE_SHO, HYBRID_MODE_WOA):
            print(
                "Expect hybrid_mode to be one of: %s, %s, %s"
                % (HYBRID_MODE_PH_SHOWOA, HYBRID_MODE_SHO, HYBRID_MODE_WOA)
            )
            raise SystemExit(-1)
        print("Hybrid mode: %s" % self.hybrid_mode)

        if parser.exists("compute_backend"):
            self.compute_backend = parser.retrieve("compute_backend")
        self.compute_backend = trim(self.compute_backend).lower()
        if self.compute_backend not in ("auto", "cpu", "cuda"):
            print("Expect compute_backend to be one of: auto, cpu, cuda")
            raise SystemExit(-1)
        print("Compute backend: %s" % self.compute_backend)

        if not chk_p_square(self.p_size):
            print("Expect popsize to be perfect squrare number")
            raise SystemExit(-1)
        sr = int(math.sqrt(float(self.p_size)))
        if sr == 1:
            self.latin.append((0.5, 0.5))
        else:
            step = 1.0 / (sr - 1)
            for i in range(sr):
                for j in range(sr):
                    lambda_val = min(1.0, step * i)
                    gamma_val = min(1.0, step * j)
                    self.latin.append((lambda_val, gamma_val))
            self.rng.shuffle(self.latin)

        if parser.exists("init"):
            self.init = parser.retrieve("init")
        print("Insertion for initialization: %s" % self.init)
        if parser.exists("k_init"):
            self.k_init = int(parser.retrieve("k_init"))
        if self.k_init == K:
            self.k_init = self.customer_num
        print("k_init: %d" % self.k_init)

        if parser.exists("cross_repair"):
            self.cross_repair = parser.retrieve("cross_repair")
        print("Insertion for crossover: %s" % self.cross_repair)

        if parser.exists("k_crossover"):
            self.k_crossover = int(parser.retrieve("k_crossover"))
        if self.k_crossover == K:
            self.k_crossover = self.customer_num
        print("k_crossover: %d" % self.k_crossover)

        if parser.exists("parent_selection"):
            self.selection = parser.retrieve("parent_selection")
        print("Parent selection: %s" % self.selection)

        if parser.exists("replacement"):
            self.replacement = parser.retrieve("replacement")
        print("Replacement strategy: %s" % self.replacement)

        if parser.exists("ls_prob"):
            self.ls_prob = float(parser.retrieve("ls_prob"))
        print("Local search probability: %.2f" % self.ls_prob)

        if parser.exists("skip_finding_lo"):
            print("Skip finding_local_optima")
            self.skip_finding_lo = True

        if parser.exists("O_1_eval"):
            print("O(1) evaluation: on")
            self.O_1_evl = True
        else:
            print("O(1) evaluation: off")

        if parser.exists("no_crossover"):
            print("No crossover used")
            self.no_crossover = True

        if parser.exists("two_opt"):
            print("2-opt: on")
            self.two_opt = True
            self.small_opts.append("2opt")
            self.mem["2opt"] = [Move() for _ in range(self.vehicle.max_num)]
        else:
            print("2-opt: off")

        if parser.exists("two_opt_star"):
            print("2-opt*: on")
            self.two_opt_star = True
            self.small_opts.append("2opt*")
            self.mem["2opt*"] = [Move() for _ in range(self.vehicle.max_num * self.vehicle.max_num)]
        else:
            print("2-opt*: off")

        if parser.exists("or_opt"):
            print("or-opt: on")
            self.or_opt = True
            self.or_opt_len = int(parser.retrieve("or_opt"))
            self.small_opts.append("oropt_single")
            self.small_opts.append("oropt_double")
            self.mem["oropt_single"] = [Move() for _ in range(self.vehicle.max_num)]
            self.mem["oropt_double"] = [Move() for _ in range(self.vehicle.max_num * self.vehicle.max_num)]
        else:
            print("or-opt: off")

        if parser.exists("two_exchange"):
            print("2-exchange: on")
            self.two_exchange = True
            self.exchange_len = int(parser.retrieve("two_exchange"))
            self.small_opts.append("2exchange")
            self.mem["2exchange"] = [Move() for _ in range(self.vehicle.max_num * self.vehicle.max_num)]
        else:
            print("2-exchange: off")

        if parser.exists("elo"):
            self.escape_local_optima = int(parser.retrieve("elo"))
        print("escape local optima number: %d" % self.escape_local_optima)

        if parser.exists("random_removal"):
            print("random_removal: on")
            self.random_removal = True
            self.destroy_opts.append("random_removal")
        else:
            print("random_removal: off")

        if parser.exists("related_removal"):
            print("related_removal: on")
            self.related_removal = True
            if parser.exists("alpha"):
                self.alpha = float(parser.retrieve("alpha"))
            self.r = self.alpha * (all_dist / all_time)
            self.destroy_opts.append("related_removal")
            print("alpha: %f, relateness norm factor: %f" % (self.alpha, self.r))
        else:
            print("related_removal: off")

        if parser.exists("removal_lower"):
            self.destroy_ratio_l = float(parser.retrieve("removal_lower"))
        print("Destroy lower ration: %f" % self.destroy_ratio_l)
        if parser.exists("removal_upper"):
            self.destroy_ratio_u = float(parser.retrieve("removal_upper"))
        print("Destroy upper ration: %f" % self.destroy_ratio_u)

        if parser.exists("regret_insertion"):
            print("regret_insertion: on")
            self.regret_insertion = True
            self.repair_opts.append("regret_insertion")
        else:
            print("regret_insertion: off")

        if parser.exists("greedy_insertion"):
            print("greedy_insertion: on")
            self.greedy_insertion = True
            self.repair_opts.append("greedy_insertion")
        else:
            print("greedy_insertion: off")

        if parser.exists("rd_removal_insertion"):
            print("Random removal and insertion: on")
            self.rd_removal_insertion = True
        else:
            print("Random removal and insertion: off")

        if parser.exists("bks"):
            self.bks = float(parser.retrieve("bks"))

        c_num = self.customer_num
        for i in range(c_num + 1):
            for j in range(c_num + 1):
                self.pm[i][j] = True
        self.pre_processing()

        self.backend = create_backend(self, self.compute_backend)
        print("Compute backend requested: %s" % self.compute_backend)
        print("Compute backend selected: %s" % self.backend.name)
        if self.backend.is_cuda and self.parallel_workers != 1:
            print(
                "CUDA backend uses a single process. Forcing workers from %d to 1"
                % self.parallel_workers
            )
            self.parallel_workers = 1

    def pre_processing(self) -> None:
        print("--------------------------------------------")
        if self.related_removal:
            c_num = self.customer_num
            DC = self.DC
            for i in range(c_num + 1):
                if i == DC:
                    continue
                for j in range(c_num + 1):
                    if j == DC or j == i:
                        self.rm[i][j] = float("inf")
                    else:
                        node_i = self.node[i]
                        node_j = self.node[j]
                        tmp_1 = self.r * max(
                            node_j.start - node_i.s_time - self.time[i][j] - node_i.end,
                            0.0,
                        )
                        tmp_2 = self.r * PENALTY_FACTOR * max(
                            node_i.start + node_i.s_time + self.time[i][j] - node_j.end,
                            0.0,
                        )
                        tmp_3 = self.dist[i][j]
                        self.rm[i][j] = tmp_3 + tmp_1 + tmp_2
                argsort(self.rm[i], self.rm_argrank[i], c_num + 1)
        if self.pruning:
            print("Do Pruning")
            c_num = self.customer_num
            DC = self.DC
            count_tw = 0
            count_c = 0
            for i in range(c_num + 1):
                if i == DC:
                    continue
                for j in range(c_num + 1):
                    if j == DC or j == i:
                        continue

                    a_i = self.node[i].start
                    s_i = self.node[i].s_time
                    d_i = self.node[i].delivery
                    p_i = self.node[i].pickup

                    b_j = self.node[j].end
                    d_j = self.node[j].delivery
                    p_j = self.node[j].pickup
                    time_ij = self.time[i][j]

                    if a_i + s_i + time_ij > b_j:
                        self.pm[i][j] = False
                        count_tw += 1
                    if d_i + d_j > self.vehicle.capacity or p_i + p_j > self.vehicle.capacity:
                        self.pm[i][j] = False
                        count_c += 1
            total_edges = c_num * (c_num - 1)
            print(
                "Total edges %d, prune by time window %d(%.4f%%),prune by capacity %d(%.4f%%)"
                % (
                    total_edges,
                    count_tw,
                    100.0 * float(count_tw) / total_edges,
                    count_c,
                    100.0 * float(count_c) / total_edges,
                )
            )

    def clear_mem(self) -> None:
        for opt in self.mem.values():
            for move in opt:
                move.len_1 = 0

    def get_mem(self, opt: str, r1: int, r2: int) -> Move:
        if opt == "2opt":
            return self.mem[opt][r1]
        if opt == "2opt*":
            return self.mem[opt][r1 * self.vehicle.max_num + r2]
        if opt == "oropt_single":
            return self.mem[opt][r1]
        if opt == "oropt_double":
            return self.mem[opt][r1 * self.vehicle.max_num + r2]
        if opt == "2exchange":
            return self.mem[opt][r1 * self.vehicle.max_num + r2]
        print("Unknown opt name: %s" % opt)
        raise SystemExit(-1)
