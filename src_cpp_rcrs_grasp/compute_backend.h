#pragma once

#include <vector>
#include "data.h"

struct RouteEval {
    bool feasible;
    double cost;
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
};

#ifdef USE_CUDA
class CudaComputeBackend : public BaseComputeBackend {
private:
    const Data& data;
    // GPU pointers for ProblemData representation
    double* d_delivery = nullptr;
    double* d_pickup = nullptr;
    double* d_start = nullptr;
    double* d_end = nullptr;
    double* d_service = nullptr;
    double* d_dist = nullptr;
    double* d_time = nullptr;

    void ensure_device_cache();
    void free_device_cache();

public:
    CudaComputeBackend(const Data& d);
    ~CudaComputeBackend() override;
    RouteEval evaluate_route(const std::vector<int>& route) override;
    std::vector<RouteEval> evaluate_routes(const std::vector<std::vector<int>>& routes) override;
    void evaluate_insertions(
        const std::vector<int>& route,
        const std::vector<int>& candidates,
        std::vector<int>& out_feasible,
        std::vector<double>& out_costs
    ) override;
};
#endif

BaseComputeBackend* create_backend(const Data& data, const std::string& mode);
