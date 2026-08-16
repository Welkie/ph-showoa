#pragma once

#include <mutex>
#include <string>
#include <vector>
#include "data.h"
#include "solution_encoding.h"

struct RouteEval {
    bool feasible;
    double cost;
};

enum EvaluationContext {
    EVALUATION_CONTEXT_UNKNOWN = 0,
    EVALUATION_CONTEXT_INITIALIZATION = 1,
    EVALUATION_CONTEXT_REPAIR = 2,
    EVALUATION_CONTEXT_LOCAL_SEARCH = 3,
    EVALUATION_CONTEXT_POPULATION = 4,
    EVALUATION_CONTEXT_ROUTE_BATCH = 5
};

enum class EvalTarget {
    CpuIncremental,
    CpuBatch,
    CudaBatch
};

struct WorkShape {
    int context = EVALUATION_CONTEXT_UNKNOWN;
    size_t item_count = 0;
    size_t total_nodes = 0;
    size_t max_length = 0;
    size_t transfer_bytes = 0;
};

enum InsertionSourceContext {
    INSERTION_CONTEXT_UNKNOWN = 0,
    INSERTION_CONTEXT_INITIALIZATION = 1,
    INSERTION_CONTEXT_REPAIR = 2,
    INSERTION_CONTEXT_LOCAL_SEARCH = 3
};

enum InsertionViolationFlags {
    INSERTION_OK = 0,
    INSERTION_BAD_ROUTE = 1 << 0,
    INSERTION_BAD_POSITION = 1 << 1,
    INSERTION_BAD_CUSTOMER = 1 << 2,
    INSERTION_CAPACITY = 1 << 3,
    INSERTION_TIME_WINDOW = 1 << 4
};

enum CandidateOperatorType {
    CANDIDATE_OPERATOR_UNKNOWN = 0,
    CANDIDATE_OPERATOR_INITIALIZATION = 1,
    CANDIDATE_OPERATOR_REPAIR = 2,
    CANDIDATE_OPERATOR_OR_OPT = 3,
    CANDIDATE_OPERATOR_RELOCATE = 4,
    CANDIDATE_OPERATOR_EXCHANGE = 5,
    CANDIDATE_OPERATOR_TWO_OPT = 6,
    CANDIDATE_OPERATOR_TWO_OPT_STAR = 7,
    CANDIDATE_OPERATOR_ROUTE_ELIMINATION = 8
};

struct CandidateContext {
    int solution_id = -1;
    int island_id = -1;
    int agent_id = -1;
    int operator_type = CANDIDATE_OPERATOR_UNKNOWN;
    int stage_id = 0;
    int local_order = 0;
    int route_version = 0;
};

struct InsertionRequest {
    int request_id = 0;
    int route_id = 0;
    int customer_id = 0;
    int insert_position = 0;
    int source_context = INSERTION_CONTEXT_UNKNOWN;
    CandidateContext context;
};

struct InsertionScore {
    int request_id = 0;
    bool feasible = false;
    double delta_distance = 0.0;
    double total_distance_after = 0.0;
    double fitness_after = 0.0;
    int violation_flags = INSERTION_OK;
};

struct BackendSnapshot {
    int depot = 0;
    int customer_num = 0;
    double capacity = 0.0;
    double start_time = 0.0;
    double end_time = 0.0;
    double dispatch_cost = 0.0;
    double unit_cost = 0.0;
    std::vector<double> delivery;
    std::vector<double> pickup;
    std::vector<double> start;
    std::vector<double> end;
    std::vector<double> service;
    std::vector<double> dist;
    std::vector<double> time;

    BackendSnapshot() = default;
    explicit BackendSnapshot(const Data& data);
};

class BaseComputeBackend {
public:
    virtual ~BaseComputeBackend() = default;
    virtual RouteEval evaluate_route(const std::vector<int>& route) = 0;
    virtual std::vector<RouteEval> evaluate_routes(const std::vector<std::vector<int>>& routes) = 0;
    virtual void evaluate_insertions(
        const std::vector<int>& route,
        const std::vector<int>& candidates,
        std::vector<int>& out_feasible,
        std::vector<double>& out_costs
    ) = 0;
    virtual std::vector<InsertionScore> evaluate_insertion_batch(
        const std::vector<std::vector<int>>& routes,
        const std::vector<InsertionRequest>& requests
    ) = 0;
    virtual std::vector<SolutionEval> evaluate_solutions(const EncodedPopulation& encoded) = 0;
    virtual EvalTarget choose_target(const WorkShape&) const { return EvalTarget::CpuBatch; }
    virtual bool is_gpu_backend() const { return false; }
};

class CpuComputeBackend : public BaseComputeBackend {
private:
    const Data& data;
public:
    CpuComputeBackend(const Data& d) : data(d) {}
    RouteEval evaluate_route(const std::vector<int>& route) override;
    std::vector<RouteEval> evaluate_routes(const std::vector<std::vector<int>>& routes) override;
    void evaluate_insertions(
        const std::vector<int>& route,
        const std::vector<int>& candidates,
        std::vector<int>& out_feasible,
        std::vector<double>& out_costs
    ) override;
    std::vector<InsertionScore> evaluate_insertion_batch(
        const std::vector<std::vector<int>>& routes,
        const std::vector<InsertionRequest>& requests
    ) override;
    std::vector<SolutionEval> evaluate_solutions(const EncodedPopulation& encoded) override;
    EvalTarget choose_target(const WorkShape& shape) const override;
};

#ifdef USE_CUDA
BaseComputeBackend* create_cuda_backend(const Data& data);
#endif

BaseComputeBackend* create_backend(const Data& data, const std::string& mode);
