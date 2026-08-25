#include "data.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace {
constexpr double PENALTY_FACTOR = 10.0;
}

// Trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// Trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// Trim from both ends (in place)
static inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
}

// Split string by a delimiter
static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

bool Data::load_problem(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open problem file: " << filepath << std::endl;
        return false;
    }

    std::vector<std::string> lines;
    std::string raw_line;
    while (std::getline(file, raw_line)) {
        lines.push_back(raw_line);
    }

    double all_delivery_val = 0.0;
    double all_pickup_val = 0.0;
    double all_dist_val = 0.0;
    double all_time_val = 0.0;

    int i = 0;
    while (i < lines.size()) {
        std::string line = lines[i];
        trim(line);
        if (line.empty()) {
            i++;
            continue;
        }

        std::vector<std::string> parts = split(line, ':');
        std::string key = parts[0];
        trim(key);
        std::string value = parts.size() > 1 ? parts[1] : "";
        trim(value);

        if (key == "NAME") {
            problem_name = value;
            std::cout << line << std::endl;
        } else if (key == "TYPE") {
            std::cout << line << std::endl;
        } else if (key == "DIMENSION") {
            std::cout << line << std::endl;
            customer_num = std::stoi(value) - 1;
            node.assign(customer_num + 1, Point{0, 0.0, 0.0, 0.0, 0.0, 0.0});
            dist.assign(customer_num + 1, std::vector<double>(customer_num + 1, 0.0));
            time.assign(customer_num + 1, std::vector<double>(customer_num + 1, 0.0));
        } else if (key == "VEHICLES") {
            std::cout << line << std::endl;
            // V_NUM_RELAX = 3
            vehicle.max_num = std::stoi(value) + 3;
        } else if (key == "DISPATCHINGCOST") {
            std::cout << line << std::endl;
            vehicle.d_cost = std::stod(value);
        } else if (key == "UNITCOST") {
            std::cout << line << std::endl;
            vehicle.unit_cost = std::stod(value);
        } else if (key == "CAPACITY") {
            std::cout << line << std::endl;
            vehicle.capacity = std::stod(value);
        } else if (key == "EDGE_WEIGHT_TYPE") {
            std::cout << line << std::endl;
            if (value != "EXPLICIT") {
                std::cerr << "Expect edge weight type: EXPLICIT, while accept type: " << value << std::endl;
                return false;
            }
        } else if (key == "NODE_SECTION") {
            i++;
            while (i < lines.size()) {
                std::string sub_line = lines[i];
                trim(sub_line);
                if (sub_line.empty()) {
                    i++;
                    continue;
                }
                std::vector<std::string> r = split(sub_line, ',');
                if (r.size() > 1) {
                    std::string idx_str = r[0]; trim(idx_str);
                    int idx = std::stoi(idx_str);
                    node[idx].id = idx;
                    std::string del_str = r[1]; trim(del_str);
                    node[idx].delivery = std::stod(del_str);
                    all_delivery_val += node[idx].delivery;
                    std::string pick_str = r[2]; trim(pick_str);
                    node[idx].pickup = std::stod(pick_str);
                    all_pickup_val += node[idx].pickup;
                    std::string start_str = r[3]; trim(start_str);
                    node[idx].start = std::stod(start_str);
                    std::string end_str = r[4]; trim(end_str);
                    node[idx].end = std::stod(end_str);
                    std::string s_str = r[5]; trim(s_str);
                    node[idx].s_time = std::stod(s_str);
                    i++;
                } else {
                    break;
                }
            }
            continue;
        } else if (key == "DISTANCETIME_SECTION") {
            i++;
            while (i < lines.size()) {
                std::string sub_line = lines[i];
                trim(sub_line);
                if (sub_line.empty()) {
                    i++;
                    continue;
                }
                std::vector<std::string> r = split(sub_line, ',');
                if (r.size() > 1) {
                    std::string idx_i_str = r[0]; trim(idx_i_str);
                    std::string idx_j_str = r[1]; trim(idx_j_str);
                    std::string d_str = r[2]; trim(d_str);
                    std::string t_str = r[3]; trim(t_str);

                    int idx_i = std::stoi(idx_i_str);
                    int idx_j = std::stoi(idx_j_str);
                    double d_val = std::stod(d_str);
                    double t_val = std::stod(t_str);

                    dist[idx_i][idx_j] = d_val;
                    all_dist_val += d_val;
                    time[idx_i][idx_j] = t_val;
                    all_time_val += t_val;

                    if (d_val < min_dist) min_dist = d_val;
                    if (d_val > max_dist) max_dist = d_val;
                    i++;
                } else {
                    break;
                }
            }
            continue;
        } else if (key == "DEPOT_SECTION") {
            i++;
            if (i < lines.size()) {
                std::string sub_line = lines[i];
                trim(sub_line);
                DC = std::stoi(sub_line);
            }
        }
        i++;
    }

    start_time = node[DC].start;
    end_time = node[DC].end;
    all_delivery = all_delivery_val;
    all_pickup = all_pickup_val;

    std::printf("Avg pick-up/dilvery demand: %.4f,%.4f\n", all_pickup / customer_num, all_delivery / customer_num);
    std::printf("Starting/end time of DC: %.4f,%.4f\n", start_time, end_time);
    std::cout << std::endl;

    return true;
}

void Data::parse_args(int argc, char* argv[]) {
    // Simple basic CLI arguments parsing for standalone test
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--random_seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::stoi(argv[++i]);
        } else if (arg == "--pop_size" && i + 1 < argc) {
            p_size = std::stoi(argv[++i]);
        } else if (arg == "--max_iter" && i + 1 < argc) {
            max_iter = std::stoi(argv[++i]);
        } else if (arg == "--workers" && i + 1 < argc) {
            parallel_workers = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--compute_backend" && i + 1 < argc) {
            compute_backend = argv[++i];
        } else if (arg == "--no_gpu_solution_eval") {
            gpu_solution_eval = false;
        } else if (arg == "--gpu_batched_insertion") {
            gpu_batched_insertion = true;
        } else if (arg == "--gpu_route_min_batch" && i + 1 < argc) {
            gpu_route_min_batch = std::stoi(argv[++i]);
        } else if (arg == "--gpu_route_min_work" && i + 1 < argc) {
            gpu_route_min_work = std::stoi(argv[++i]);
        } else if (arg == "--gpu_insertion_min_batch" && i + 1 < argc) {
            gpu_insertion_min_batch = std::stoi(argv[++i]);
        } else if (arg == "--gpu_serialize_workers") {
            gpu_serialize_workers = true;
        } else if (arg == "--gpu_request_broker") {
            gpu_request_broker = true;
        } else if (arg == "--execution_policy" && i + 1 < argc) {
            execution_policy = argv[++i];
            if (execution_policy != "legacy" && execution_policy != "adaptive" &&
                execution_policy != "cuda_force") {
                std::cerr << "Unknown execution policy '" << execution_policy
                          << "'. Use 'legacy', 'adaptive', or 'cuda_force'." << std::endl;
                execution_policy = "legacy";
            }
        } else if (arg == "--gpu_solution_verify_interval" && i + 1 < argc) {
            gpu_solution_verify_interval = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--hybrid_mode" && i + 1 < argc) {
            hybrid_mode = argv[++i];
            if (hybrid_mode != "ph_showoa" && hybrid_mode != "sho" && hybrid_mode != "woa") {
                std::cerr << "Unknown hybrid mode '" << hybrid_mode
                          << "'. Use 'ph_showoa', 'sho', or 'woa'." << std::endl;
                hybrid_mode = "ph_showoa";
            }
        } else if (arg == "--sho_mutation_prob" && i + 1 < argc) {
            sho_mutation_prob = std::clamp(std::stod(argv[++i]), 0.0, 1.0);
        } else if (arg == "--local_search_interval" && i + 1 < argc) {
            local_search_interval = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--stagnation_interval" && i + 1 < argc) {
            stagnation_interval = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--diversify_ratio" && i + 1 < argc) {
            diversify_ratio = std::clamp(std::stod(argv[++i]), 0.0, 1.0);
        } else if (arg == "--pruning") {
            pruning = true;
        } else if (arg == "--O_1_eval") {
            O_1_evl = true;
        } else if (arg == "--two_opt") {
            two_opt = true;
            small_opts.push_back("2opt");
        } else if (arg == "--two_opt_star") {
            two_opt_star = true;
            small_opts.push_back("2opt*");
        } else if (arg == "--or_opt" && i + 1 < argc) {
            or_opt = true;
            or_opt_len = std::stoi(argv[++i]);
            small_opts.push_back("oropt_single");
        } else if (arg == "--two_exchange" && i + 1 < argc) {
            two_exchange = true;
            ex_len = std::stoi(argv[++i]);
            small_opts.push_back("2exchange");
        } else if (arg == "--related_removal") {
            related_removal = true;
        } else if (arg == "--regret_insertion") {
            regret_insertion = true;
        } else if (arg == "--init" && i + 1 < argc) {
            init = argv[++i];
        } else if (arg == "--k_init" && i + 1 < argc) {
            k_init = std::stoi(argv[++i]);
        } else if (arg == "--grasp_alpha_lo" && i + 1 < argc) {
            grasp_alpha_lo = std::clamp(std::stod(argv[++i]), 0.0, 1.0);
        } else if (arg == "--grasp_alpha_hi" && i + 1 < argc) {
            grasp_alpha_hi = std::clamp(std::stod(argv[++i]), 0.0, 1.0);
        } else if (arg == "--paper_flags") {
            paper_flags = true;
            pruning = true;
            O_1_evl = true;
            two_opt = true;
            two_opt_star = true;
            or_opt = true;
            or_opt_len = 2;
            small_opts.push_back("2opt");
            small_opts.push_back("2opt*");
            small_opts.push_back("oropt_single");
            two_exchange = true;
            ex_len = 2;
            small_opts.push_back("2exchange");
            related_removal = true;
            regret_insertion = true;
            init = "sa";
        } else if (arg == "--profile") {
            profile = true;
        } else if (arg == "--architecture" && i + 1 < argc) {
            architecture = argv[++i];
            if (architecture != "legacy" && architecture != "hybrid_v2" &&
                architecture != "full_gpu") {
                std::cerr << "Unknown architecture '" << architecture
                          << "'. Use 'legacy', 'hybrid_v2', or 'full_gpu'." << std::endl;
                architecture = "legacy";
            }
        } else if (arg == "--objective" && i + 1 < argc) {
            objective = argv[++i];
            if (objective != "weighted" && objective != "lexicographic") {
                std::cerr << "Unknown objective '" << objective << "'. Use 'weighted' or 'lexicographic'." << std::endl;
                objective = "weighted";
            }
        } else if (arg == "--num_islands" && i + 1 < argc) {
            num_islands = std::stoi(argv[++i]);
        } else if (arg == "--migration_interval" && i + 1 < argc) {
            migration_interval = std::stoi(argv[++i]);
        } else if (arg == "--migration_size" && i + 1 < argc) {
            migration_size = std::stoi(argv[++i]);
        } else if (arg == "--migration_mode" && i + 1 < argc) {
            migration_mode = argv[++i];
        }
    }

    num_islands = std::max(1, num_islands);
    migration_interval = std::max(1, migration_interval);
    migration_size = std::max(1, migration_size);
    if (architecture == "hybrid_v2" || architecture == "full_gpu") {
        objective = "lexicographic";
        gpu_batched_insertion = true;
    }
    if (architecture == "full_gpu" && compute_backend == "auto") {
        compute_backend = "cuda";
    }

    std::printf("Initial random seed: %d\n", seed);
    std::printf("Pruning: %s\n", pruning ? "on" : "off");
    if (!output_path.empty()) {
        std::printf("Write best solution to %s\n", output_path.c_str());
    }
    std::printf("Max PH-SHOWOA iterations: %d\n", max_iter);
    std::printf("Population size: %d\n", p_size);
    if (k_init == -1) {
        k_init = customer_num;
    }
    std::printf("k_init: %d\n", k_init);
    std::printf("Compute backend: %s\n", compute_backend.c_str());
    std::printf("GPU solution evaluation: %s\n", gpu_solution_eval ? "on" : "off");
    std::printf("GPU batched insertion: %s\n", gpu_batched_insertion ? "on" : "off");
    std::printf("GPU route min batch: %d\n", gpu_route_min_batch);
    std::printf("GPU route min work: %d\n", gpu_route_min_work);
    std::printf("GPU insertion min batch: %d\n", gpu_insertion_min_batch);
    std::printf("GPU serialize workers: %s\n", gpu_serialize_workers ? "on" : "off");
    std::printf("GPU request broker: %s\n", gpu_request_broker ? "on" : "off");
    std::printf("Execution policy: %s\n", execution_policy.c_str());
    std::printf("GPU solution verify interval: %d\n", gpu_solution_verify_interval);
    std::printf("Profiling: %s\n", profile ? "on" : "off");
    std::printf("Architecture: %s\n", architecture.c_str());
    std::printf("Objective mode: %s\n", objective.c_str());
    std::printf("Island Model - Num islands: %d\n", num_islands);
    std::printf("Island Model - Migration interval: %d\n", migration_interval);
    std::printf("Island Model - Migration size: %d\n", migration_size);
    std::printf("Island Model - Migration mode: %s\n", migration_mode.c_str());

    pre_process();

    // Generate Latin square slots
    int sr = static_cast<int>(std::sqrt(static_cast<double>(p_size)));
    if (sr == 1) {
        latin.push_back({0.5, 0.5});
    } else {
        double step = 1.0 / (sr - 1);
        for (int row = 0; row < sr; ++row) {
            for (int col = 0; col < sr; ++col) {
                double lambda_val = std::min(1.0, step * row);
                double gamma_val = std::min(1.0, step * col);
                latin.push_back({lambda_val, gamma_val});
            }
        }
    }
}

void Data::pre_process() {
    pm.assign(customer_num + 1, std::vector<bool>(customer_num + 1, true));
    if (pruning) {
        for (int i = 0; i <= customer_num; ++i) {
            for (int j = 0; j <= customer_num; ++j) {
                if (i == j) {
                    pm[i][j] = false;
                    continue;
                }
                const double delivery_sum = node[i].delivery + node[j].delivery;
                const double pickup_sum = node[i].pickup + node[j].pickup;
                if (delivery_sum > vehicle.capacity || pickup_sum > vehicle.capacity) {
                    pm[i][j] = false;
                    continue;
                }
                const double earliest_depart = node[i].start + node[i].s_time + time[i][j];
                if (earliest_depart > node[j].end) {
                    pm[i][j] = false;
                }
            }
        }
    }

    rm.assign(customer_num + 1, std::vector<double>(customer_num + 1, std::numeric_limits<double>::infinity()));
    rm_argrank.assign(customer_num + 1, std::vector<int>());
    if (related_removal) {
        for (int i = 0; i <= customer_num; ++i) {
            rm_argrank[i].resize(customer_num + 1);
            std::iota(rm_argrank[i].begin(), rm_argrank[i].end(), 0);
            for (int j = 0; j <= customer_num; ++j) {
                if (i == j || i == DC || j == DC) {
                    rm[i][j] = std::numeric_limits<double>::infinity();
                    continue;
                }
                const double late_penalty = alpha * std::max(node[j].start - node[i].s_time - time[i][j] - node[i].end, 0.0);
                const double early_penalty = alpha * PENALTY_FACTOR * std::max(node[i].start + node[i].s_time + time[i][j] - node[j].end, 0.0);
                rm[i][j] = dist[i][j] + late_penalty + early_penalty;
            }
            std::sort(rm_argrank[i].begin(), rm_argrank[i].end(), [&](int a, int b) {
                return rm[i][a] < rm[i][b];
            });
        }
    }

    destroy_opts.clear();
    repair_opts.clear();
    if (related_removal) {
        destroy_opts.push_back("related_removal");
    }
    if (random_removal || destroy_opts.empty()) {
        destroy_opts.push_back("random_removal");
    }
    if (regret_insertion) {
        repair_opts.push_back("regret_insertion");
    }
    if (greedy_insertion) {
        repair_opts.push_back("greedy_insertion");
    }
    if (repair_opts.empty()) {
        repair_opts.push_back("regret_insertion");
    }
}
