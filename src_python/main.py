import signal
import sys

from .argparse_util import ArgumentParser
from .data import Data
from .search_framework import search_framework
from . import state


def _signal_handler(signum, _frame):
    print("Interrupt signal (%d) received." % signum)
    print("Best cost: %.4f." % state.best_s_cost)
    print("Time to find this solution: %d." % int(state.find_best_time))
    print("Time to surpass BKS: %d." % int(state.find_bks_time))
    raise SystemExit(signum)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    state.best_s.cost = float("inf")
    state.best_s_cost = -1.0
    state.find_best_time = 0
    state.find_bks_time = 0
    state.find_best_run = 0
    state.find_bks_run = 0
    state.find_best_gen = 0
    state.find_bks_gen = 0
    state.find_better = False
    state.call_count_move_eval = 0
    state.mean_duration_move_eval = 0.0
    state.mean_route_len = 0.0

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = ArgumentParser()
    parser.add_argument("--problem", 1, False)

    parser.add_argument("--pruning")
    parser.add_argument("--output", 1)
    parser.add_argument("--time", 1)
    parser.add_argument("--runs", 1)
    parser.add_argument("--g_1", 1)
    parser.add_argument("--max_iter", 1)
    parser.add_argument("--pop_size", 1)
    parser.add_argument("--workers", 1)
    parser.add_argument("--local_search_interval", 1)
    parser.add_argument("--stagnation_interval", 1)
    parser.add_argument("--diversify_ratio", 1)
    parser.add_argument("--sho_mutation_prob", 1)
    parser.add_argument("--hybrid_mode", 1)
    parser.add_argument("--compute_backend", 1)
    parser.add_argument("--init", 1)
    parser.add_argument("--k_init", 1)
    parser.add_argument("--no_crossover")
    parser.add_argument("--cross_repair", 1)
    parser.add_argument("--k_crossover", 1)
    parser.add_argument("--parent_selection", 1)
    parser.add_argument("--replacement", 1)
    parser.add_argument("--ls_prob", 1)
    parser.add_argument("--skip_finding_lo")
    parser.add_argument("--O_1_eval")
    parser.add_argument("--two_opt")
    parser.add_argument("--two_opt_star")
    parser.add_argument("--or_opt", 1)
    parser.add_argument("--two_exchange", 1)
    parser.add_argument("--elo", 1)
    parser.add_argument("--random_removal")
    parser.add_argument("--related_removal")
    parser.add_argument("--alpha", 1)
    parser.add_argument("--removal_lower", 1)
    parser.add_argument("--removal_upper", 1)
    parser.add_argument("--regret_insertion")
    parser.add_argument("--greedy_insertion")
    parser.add_argument("--rd_removal_insertion")
    parser.add_argument("--bks", 1)
    parser.add_argument("--random_seed", 1)
    parser.add_argument("--paper_flags")


    parser.parse(argv)
    data = Data(parser)
    search_framework(data, state.best_s)

    state.best_s_cost = state.best_s.cost


if __name__ == "__main__":
    main()
