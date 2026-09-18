#pragma once

#include <limits>
#include <string>

struct ObjectiveValue {
    bool feasible = false;
    int nv = std::numeric_limits<int>::max();
    double dist = std::numeric_limits<double>::infinity();
    double cost = std::numeric_limits<double>::infinity();
    int violation_flags = 0;
};

inline bool objective_better(
    const ObjectiveValue& candidate,
    const ObjectiveValue& incumbent,
    const std::string& objective
) {
    if (candidate.feasible != incumbent.feasible) {
        return candidate.feasible;
    }
    if (!candidate.feasible) {
        return candidate.violation_flags < incumbent.violation_flags;
    }
    if (objective == "lexicographic") {
        if (candidate.nv != incumbent.nv) {
            return candidate.nv < incumbent.nv;
        }
        if (candidate.dist != incumbent.dist) {
            return candidate.dist < incumbent.dist - 1e-9;
        }
    }
    return candidate.cost < incumbent.cost - 1e-9;
}

inline ObjectiveValue worst_objective_value() {
    return ObjectiveValue{};
}
