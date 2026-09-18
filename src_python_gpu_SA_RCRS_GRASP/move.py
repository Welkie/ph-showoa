from dataclasses import dataclass, field
from typing import List


@dataclass
class Seq:
    r_index: int
    start_point: int
    end_point: int


@dataclass
class Move:
    r_indice: List[int] = field(default_factory=lambda: [-2, -2])
    seqList_1: List[Seq] = field(default_factory=lambda: [Seq(0, 0, 0) for _ in range(4)])
    len_1: int = 0
    seqList_2: List[Seq] = field(default_factory=lambda: [Seq(0, 0, 0) for _ in range(4)])
    len_2: int = 0
    delta_cost: float = float("inf")

    def copy_from(self, other: "Move") -> None:
        self.r_indice = list(other.r_indice)
        self.seqList_1 = [Seq(s.r_index, s.start_point, s.end_point) for s in other.seqList_1]
        self.len_1 = other.len_1
        self.seqList_2 = [Seq(s.r_index, s.start_point, s.end_point) for s in other.seqList_2]
        self.len_2 = other.len_2
        self.delta_cost = other.delta_cost

    def clone(self) -> "Move":
        new_move = Move()
        new_move.copy_from(self)
        return new_move
