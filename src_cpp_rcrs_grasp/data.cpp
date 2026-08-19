#include "data.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

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
        } else if (arg == "--paper_flags") {
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
        }
    }

    std::printf("Initial random seed: %d\n", seed);
    std::printf("Pruning: %s\n", pruning ? "on" : "off");
    if (!output_path.empty()) {
        std::printf("Write best solution to %s\n", output_path.c_str());
    }
    std::printf("Max PH-SHOWOA iterations: %d\n", max_iter);
    std::printf("Population size: %d\n", p_size);
    std::printf("Compute backend: %s\n", compute_backend.c_str());

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
