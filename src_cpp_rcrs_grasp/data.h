#pragma once

#include <string>
#include <vector>
#include <cmath>

struct Point {
    int id;
    double pickup;
    double delivery;
    double s_time;
    double start;
    double end;
};

struct Vehicle {
    int type = 0;
    double capacity = 0.0;
    int max_num = 0;
    double unit_cost = 0.0;
    double d_cost = 0.0;
};

class Data {
public:
    std::string problem_name;
    int customer_num = 0;
    int DC = 0;
    std::vector<Point> node;
    std::vector<std::vector<double>> dist;
    std::vector<std::vector<double>> time;
    
    Vehicle vehicle;
    double max_dist = 0.0;
    double min_dist = 1e9;
    double all_pickup = 0.0;
    double all_delivery = 0.0;
    double start_time = 0.0;
    double end_time = 0.0;

    // Hyperparameters & CLI args
    int seed = 42;
    bool pruning = false;
    std::string output_path = "";
    int tmax = -1;
    int runs = 10;
    int max_iter = 500;
    int p_size = 64;
    int parallel_workers = 0;
    int local_search_interval = 25;
    int stagnation_interval = 50;
    double diversify_ratio = 0.40;
    double sho_mutation_prob = 0.35;
    std::string hybrid_mode = "ph_showoa";
    std::string compute_backend = "auto";
    std::string init = "rcrs_grasp";   // Options: "rcrs_grasp"/"rcg" (RCG, default), "sa", "rcrs"
    int k_init = -1;
    std::string cross_repair = "rcrs";
    int k_crossover = -1;
    std::string selection = "circle";
    std::string replacement = "one_on_one";
    double ls_prob = 1.0;
    bool skip_finding_lo = false;
    bool O_1_evl = false;
    bool no_crossover = false;

    // Advanced search configuration
    bool two_opt = false;
    bool two_opt_star = true;
    bool or_opt = false;
    int or_opt_len = 3;
    bool two_exchange = false;
    int ex_len = 2;
    int elo = 1;
    bool random_removal = false;
    bool related_removal = false;
    double alpha = 1.0;
    double removal_lower = 0.25;
    double removal_upper = 0.40;
    bool regret_insertion = false;
    bool greedy_insertion = false;
    bool rd_removal_insertion = false;
    double bks = 0.0;

    std::vector<std::pair<double, double>> latin;
    std::vector<std::string> small_opts;

    // Parser functions
    bool load_problem(const std::string& filepath);
    void parse_args(int argc, char* argv[]);
};
