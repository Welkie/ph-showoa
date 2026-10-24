from __future__ import annotations

from typing import List, Optional
from .problem import Problem


class Solution:
    def __init__(self, routes: List[List[int]], problem: Optional[Problem] = None) -> None:
        # Filter empty routes and store customer sequences
        self.routes: List[List[int]] = [list(r) for r in routes if len(r) > 0]
        self.nv: int = len(self.routes)
        self.td: float = 0.0
        self.cost: float = 0.0
        self.fitness: float = 0.0

        if problem is not None:
            self.evaluate(problem)

    def evaluate(self, problem: Problem) -> None:
        self.nv = len(self.routes)
        total_dist = 0.0
        for r in self.routes:
            if not r:
                continue
            prev = 0
            for curr in r:
                total_dist += problem.distance[prev][curr]
                prev = curr
            total_dist += problem.distance[prev][0]

        self.td = total_dist
        # Hierarchical cost: Eq. (1) in Section 3 with u1 = 2000, u2 = 1.0
        self.cost = problem.dispatch_cost * self.nv + problem.unit_cost * self.td
        self.fitness = self.cost

    def copy(self) -> Solution:
        new_sol = Solution([list(r) for r in self.routes])
        new_sol.nv = self.nv
        new_sol.td = self.td
        new_sol.cost = self.cost
        new_sol.fitness = self.fitness
        return new_sol

    def __lt__(self, other: Solution) -> bool:
        # Hierarchical lexicographic: NV first, then TD
        if self.nv != other.nv:
            return self.nv < other.nv
        return self.td < other.td

    def __le__(self, other: Solution) -> bool:
        if self.nv != other.nv:
            return self.nv < other.nv
        return self.td <= other.td

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Solution):
            return False
        return self.routes == other.routes

    def __repr__(self) -> str:
        return f"Solution(NV={self.nv}, TD={self.td:.2f}, Cost={self.cost:.2f}, routes={len(self.routes)})"
