#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>

struct ProfileCounter {
    const char* name;
    std::atomic<long long> calls;
    std::atomic<long long> nanos;
    std::atomic<long long> units;
    std::atomic<long long> max_units;

    explicit ProfileCounter(const char* counter_name)
        : name(counter_name), calls(0), nanos(0), units(0), max_units(0) {}

    void reset() {
        calls.store(0, std::memory_order_relaxed);
        nanos.store(0, std::memory_order_relaxed);
        units.store(0, std::memory_order_relaxed);
        max_units.store(0, std::memory_order_relaxed);
    }

    void add(long long elapsed_nanos, long long unit_count = 0) {
        calls.fetch_add(1, std::memory_order_relaxed);
        nanos.fetch_add(elapsed_nanos, std::memory_order_relaxed);
        units.fetch_add(unit_count, std::memory_order_relaxed);
        long long observed = max_units.load(std::memory_order_relaxed);
        while (unit_count > observed && !max_units.compare_exchange_weak(
            observed,
            unit_count,
            std::memory_order_relaxed,
            std::memory_order_relaxed
        )) {}
    }
};

struct ProfileRegistry {
    ProfileCounter initialization{"initialization"};
    ProfileCounter fitness{"fitness_calculation"};
    ProfileCounter feasibility_single{"feasibility_single_route"};
    ProfileCounter feasibility_batch{"feasibility_route_batch"};
    ProfileCounter insertion{"insertion_scoring"};
    ProfileCounter insertion_batch{"insertion_batch_scoring"};
    ProfileCounter insertion_broker_submission{"insertion_broker_submission"};
    ProfileCounter insertion_selection{"insertion_cpu_selection"};
    ProfileCounter insertion_selection_initialization{"insertion_selection_initialization"};
    ProfileCounter insertion_selection_repair{"insertion_selection_repair"};
    ProfileCounter insertion_selection_local_search{"insertion_selection_local_search"};
    ProfileCounter dispatch_cpu_incremental{"dispatch_cpu_incremental"};
    ProfileCounter dispatch_cpu_batch{"dispatch_cpu_batch"};
    ProfileCounter dispatch_cuda_batch{"dispatch_cuda_batch"};
    ProfileCounter population_update{"population_update"};
    ProfileCounter sho_update{"sho_guided_crossover"};
    ProfileCounter woa_update{"woa_intensification"};
    ProfileCounter repair{"repair"};
    ProfileCounter local_search{"local_search"};
    ProfileCounter diversification{"diversification"};
    ProfileCounter cuda_alloc{"cuda_allocation"};
    ProfileCounter cuda_h2d{"cuda_copy_h2d"};
    ProfileCounter cuda_kernel{"cuda_kernel"};
    ProfileCounter cuda_d2h{"cuda_copy_d2h"};
    ProfileCounter cuda_route_cpu_fallback{"cuda_route_cpu_fallback"};
    ProfileCounter cuda_insertion_alloc{"cuda_insertion_allocation"};
    ProfileCounter cuda_insertion_h2d{"cuda_insertion_copy_h2d"};
    ProfileCounter cuda_insertion_kernel{"cuda_insertion_kernel"};
    ProfileCounter cuda_insertion_d2h{"cuda_insertion_copy_d2h"};
    ProfileCounter cuda_insertion_cpu_fallback{"cuda_insertion_cpu_fallback"};
    ProfileCounter insertion_scheduler_cpu{"insertion_scheduler_cpu"};
    ProfileCounter route_scheduler_cpu{"route_scheduler_cpu"};

    void reset() {
        initialization.reset();
        fitness.reset();
        feasibility_single.reset();
        feasibility_batch.reset();
        insertion.reset();
        insertion_batch.reset();
        insertion_broker_submission.reset();
        insertion_selection.reset();
        insertion_selection_initialization.reset();
        insertion_selection_repair.reset();
        insertion_selection_local_search.reset();
        dispatch_cpu_incremental.reset();
        dispatch_cpu_batch.reset();
        dispatch_cuda_batch.reset();
        population_update.reset();
        sho_update.reset();
        woa_update.reset();
        repair.reset();
        local_search.reset();
        diversification.reset();
        cuda_alloc.reset();
        cuda_h2d.reset();
        cuda_kernel.reset();
        cuda_d2h.reset();
        cuda_route_cpu_fallback.reset();
        cuda_insertion_alloc.reset();
        cuda_insertion_h2d.reset();
        cuda_insertion_kernel.reset();
        cuda_insertion_d2h.reset();
        cuda_insertion_cpu_fallback.reset();
        insertion_scheduler_cpu.reset();
        route_scheduler_cpu.reset();
    }
};

inline ProfileRegistry& profile_registry() {
    static ProfileRegistry registry;
    return registry;
}

inline void profile_reset() {
    profile_registry().reset();
}

inline void profile_print_counter(const ProfileCounter& counter) {
    long long calls = counter.calls.load(std::memory_order_relaxed);
    long long nanos = counter.nanos.load(std::memory_order_relaxed);
    long long units = counter.units.load(std::memory_order_relaxed);
    long long max_units = counter.max_units.load(std::memory_order_relaxed);
    if (calls == 0) {
        return;
    }

    double total_ms = static_cast<double>(nanos) / 1e6;
    double avg_ms = total_ms / static_cast<double>(calls);
    if (units > 0) {
        double avg_units = static_cast<double>(units) / static_cast<double>(calls);
        std::printf("%-28s calls=%8lld units=%10lld avg_units=%10.2f max_units=%10lld total_ms=%12.3f avg_ms=%10.6f\n",
                    counter.name, calls, units, avg_units, max_units, total_ms, avg_ms);
    } else {
        std::printf("%-28s calls=%8lld total_ms=%12.3f avg_ms=%10.6f\n",
                    counter.name, calls, total_ms, avg_ms);
    }
}

inline void profile_print() {
    ProfileRegistry& p = profile_registry();
    std::printf("------------Profile-----------\n");
    profile_print_counter(p.initialization);
    profile_print_counter(p.fitness);
    profile_print_counter(p.feasibility_single);
    profile_print_counter(p.feasibility_batch);
    profile_print_counter(p.insertion);
    profile_print_counter(p.insertion_batch);
    profile_print_counter(p.insertion_broker_submission);
    profile_print_counter(p.insertion_selection);
    profile_print_counter(p.insertion_selection_initialization);
    profile_print_counter(p.insertion_selection_repair);
    profile_print_counter(p.insertion_selection_local_search);
    profile_print_counter(p.dispatch_cpu_incremental);
    profile_print_counter(p.dispatch_cpu_batch);
    profile_print_counter(p.dispatch_cuda_batch);
    profile_print_counter(p.population_update);
    profile_print_counter(p.sho_update);
    profile_print_counter(p.woa_update);
    profile_print_counter(p.repair);
    profile_print_counter(p.local_search);
    profile_print_counter(p.diversification);
    profile_print_counter(p.cuda_alloc);
    profile_print_counter(p.cuda_h2d);
    profile_print_counter(p.cuda_kernel);
    profile_print_counter(p.cuda_d2h);
    profile_print_counter(p.cuda_route_cpu_fallback);
    profile_print_counter(p.cuda_insertion_alloc);
    profile_print_counter(p.cuda_insertion_h2d);
    profile_print_counter(p.cuda_insertion_kernel);
    profile_print_counter(p.cuda_insertion_d2h);
    profile_print_counter(p.cuda_insertion_cpu_fallback);
    profile_print_counter(p.insertion_scheduler_cpu);
    profile_print_counter(p.route_scheduler_cpu);
}

class ScopedProfileTimer {
public:
    ScopedProfileTimer(bool enabled, ProfileCounter& counter, long long unit_count = 0)
        : enabled_(enabled),
          counter_(enabled ? &counter : nullptr),
          unit_count_(unit_count),
          start_(enabled ? std::chrono::high_resolution_clock::now() : ClockTime{}) {}

    ~ScopedProfileTimer() {
        if (!enabled_ || counter_ == nullptr) {
            return;
        }
        auto elapsed = std::chrono::high_resolution_clock::now() - start_;
        long long nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        counter_->add(nanos, unit_count_);
    }

    ScopedProfileTimer(const ScopedProfileTimer&) = delete;
    ScopedProfileTimer& operator=(const ScopedProfileTimer&) = delete;

private:
    using ClockTime = std::chrono::high_resolution_clock::time_point;

    bool enabled_;
    ProfileCounter* counter_;
    long long unit_count_;
    ClockTime start_;
};
