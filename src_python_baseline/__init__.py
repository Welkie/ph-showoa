"""
PH-SHOWOA Canonical Paper-Aligned Python Solver
-----------------------------------------------
Strict implementation of Nguyen et al. (PLOS ONE 2026) under the 4-level Priority Hierarchy of Truth:
1. Section 3 (Mathematical formulation Eq. 1-11) has top priority.
2. Pseudocode (Algorithms 1-10) has second priority.
3. Textual descriptions resolve ambiguities in pseudocode.
4. Cited standard literature (Li & Lim [49], Liu et al. [11]) provide canonical choices.

Strictly contains only algorithms and operators explicitly specified in the paper.
"""

from .problem import Problem, Node
from .solution import Solution
from .algorithm1_solver import PHSHOWOASolver, SolverConfig, RunResult

__all__ = ["Problem", "Node", "Solution", "PHSHOWOASolver", "SolverConfig", "RunResult"]
