from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union


@dataclass
class Node:
    id: int
    x: float
    y: float
    delivery: float
    pickup: float
    ready: float
    due: float
    service: float


CANONICAL_INSTANCES = {
    # 9 small
    "rcdp1001": "RCdp1001", "rcdp1004": "RCdp1004", "rcdp1007": "RCdp1007",
    "rcdp2501": "RCdp2501", "rcdp2504": "RCdp2504", "rcdp2507": "RCdp2507",
    "rcdp5001": "RCdp5001", "rcdp5004": "RCdp5004", "rcdp5007": "RCdp5007",
    "rcdp0501": "RCdp5001", "rcdp0504": "RCdp5004", "rcdp0507": "RCdp5007",
    # 56 medium
    "rcdp101": "RCdp101", "rcdp102": "RCdp102", "rcdp103": "RCdp103", "rcdp104": "RCdp104",
    "rcdp105": "RCdp105", "rcdp106": "RCdp106", "rcdp107": "RCdp107", "rcdp108": "RCdp108",
    "rcdp201": "RCdp201", "rcdp202": "RCdp202", "rcdp203": "RCdp203", "rcdp204": "RCdp204",
    "rcdp205": "RCdp205", "rcdp206": "RCdp206", "rcdp207": "RCdp207", "rcdp208": "RCdp208",
    "cdp101": "Cdp101", "cdp102": "Cdp102", "cdp103": "Cdp103", "cdp104": "Cdp104",
    "cdp105": "Cdp105", "cdp106": "Cdp106", "cdp107": "Cdp107", "cdp108": "Cdp108", "cdp109": "Cdp109",
    "cdp201": "Cdp201", "cdp202": "Cdp202", "cdp203": "Cdp203", "cdp204": "Cdp204",
    "cdp205": "Cdp205", "cdp206": "Cdp206", "cdp207": "Cdp207", "cdp208": "Cdp208",
    "rdp101": "Rdp101", "rdp102": "Rdp102", "rdp103": "Rdp103", "rdp104": "Rdp104",
    "rdp105": "Rdp105", "rdp106": "Rdp106", "rdp107": "Rdp107", "rdp108": "Rdp108",
    "rdp109": "Rdp109", "rdp110": "Rdp110", "rdp111": "Rdp111", "rdp112": "Rdp112",
    "rdp201": "Rdp201", "rdp202": "Rdp202", "rdp203": "Rdp203", "rdp204": "Rdp204",
    "rdp205": "Rdp205", "rdp206": "Rdp206", "rdp207": "Rdp207", "rdp208": "Rdp208",
    "rdp209": "Rdp209", "rdp210": "Rdp210", "rdp211": "Rdp211",
}


def normalize_instance_name(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.lower().startswith("explicit_"):
        cleaned = cleaned[9:]
    for ext in [".txt", ".vrpsdptw"]:
        if cleaned.lower().endswith(ext):
            cleaned = cleaned[:-len(ext)]
    return CANONICAL_INSTANCES.get(cleaned.lower(), raw)


class Problem:
    def __init__(
        self,
        name: str,
        nodes: List[Node],
        distance: List[List[float]],
        travel_time: List[List[float]],
        max_vehicles: int,
        capacity: float,
        dispatch_cost: float = 2000.0,
        unit_cost: float = 1.0,
    ) -> None:
        self.name = normalize_instance_name(name)
        self.nodes = nodes
        self.num_customers = len(nodes) - 1
        self.distance = distance
        self.travel_time = travel_time
        self.max_vehicles = max_vehicles
        self.capacity = capacity
        self.dispatch_cost = dispatch_cost
        self.unit_cost = unit_cost

        # Convenient lookup arrays
        self.delivery = [n.delivery for n in nodes]
        self.pickup = [n.pickup for n in nodes]
        self.ready_time = [n.ready for n in nodes]
        self.due_time = [n.due for n in nodes]
        self.service_time = [n.service for n in nodes]

    @classmethod
    def load(cls, path_or_str: Union[str, Path]) -> Problem:
        path = Path(path_or_str)
        if not path.is_file():
            raise FileNotFoundError(f"Instance file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Detect format: explicit VRPSDPTW or Wang-Chen / Solomon .txt
        is_explicit = any(
            line.upper().startswith("DIMENSION")
            or line.upper().startswith("NODE_SECTION")
            or "DISTANCETIME_SECTION" in line.upper()
            for line in lines[:20]
        )

        if is_explicit:
            return cls._load_explicit(path.stem, lines)
        else:
            return cls._load_wang_chen(path.stem, lines)

    @classmethod
    def _load_wang_chen(cls, name: str, lines: List[str]) -> Problem:
        # Wang & Chen format
        max_vehicles = 100
        capacity = 200.0

        for line in lines:
            tokens = line.replace(",", " ").split()
            if len(tokens) == 3 and tokens[0].isdigit() and tokens[1].isdigit() and tokens[2].isdigit():
                max_vehicles = int(tokens[1])
                capacity = float(tokens[2])
                break
            elif len(tokens) == 2 and tokens[0].isdigit() and tokens[1].isdigit():
                max_vehicles = int(tokens[0])
                capacity = float(tokens[1])
                break

        nodes: List[Node] = []
        for line in lines:
            tokens = line.replace(",", " ").split()
            if len(tokens) >= 7 and tokens[0].isdigit():
                cid = int(tokens[0])
                x = float(tokens[1])
                y = float(tokens[2])
                if len(tokens) == 7:
                    # Solomon with delivery = demand, pickup = 0
                    delivery = float(tokens[3])
                    pickup = 0.0
                    ready = float(tokens[4])
                    due = float(tokens[5])
                    service = float(tokens[6])
                else:
                    # Wang & Chen format: CustNo, X, Y, DDEMAND, PDEMAND, READY, DUE, SERVICE
                    delivery = float(tokens[3])
                    pickup = float(tokens[4])
                    ready = float(tokens[5])
                    due = float(tokens[6])
                    service = float(tokens[7])

                nodes.append(Node(cid, x, y, delivery, pickup, ready, due, service))

        # Re-index nodes to contiguous 0..N if needed
        nodes.sort(key=lambda n: n.id)
        for idx, n in enumerate(nodes):
            n.id = idx

        n_count = len(nodes)
        distance = [[0.0] * n_count for _ in range(n_count)]
        travel_time = [[0.0] * n_count for _ in range(n_count)]

        for i in range(n_count):
            for j in range(n_count):
                if i != j:
                    dx = nodes[i].x - nodes[j].x
                    dy = nodes[i].y - nodes[j].y
                    d = math.sqrt(dx * dx + dy * dy)
                    distance[i][j] = d
                    travel_time[i][j] = d

        return cls(name, nodes, distance, travel_time, max_vehicles, capacity)

    @classmethod
    def _load_explicit(cls, name: str, lines: List[str]) -> Problem:
        dimension = 0
        max_vehicles = 100
        capacity = 200.0
        dispatch_cost = 2000.0
        unit_cost = 1.0

        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().upper()
                val = val.strip()
                if key == "DIMENSION":
                    dimension = int(val)
                elif key == "VEHICLES":
                    max_vehicles = int(val)
                elif key == "CAPACITY":
                    capacity = float(val)
                elif key == "DISPATCHINGCOST":
                    dispatch_cost = float(val)
                elif key == "UNITCOST":
                    unit_cost = float(val)

        nodes: List[Node] = []
        in_nodes = False
        in_matrix = False
        matrix_lines: List[List[str]] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            up = line_str.upper()
            if up.startswith("NODE_SECTION"):
                in_nodes = True
                in_matrix = False
                continue
            elif up.startswith("DISTANCETIME_SECTION") or up.startswith("EDGE_WEIGHT_SECTION"):
                in_nodes = False
                in_matrix = True
                continue
            elif up.startswith("EOF") or up.startswith("DEPOT_SECTION"):
                in_nodes = False
                in_matrix = False
                if up.startswith("EOF"):
                    break
                continue

            if in_nodes:
                tokens = line_str.replace(",", " ").split()
                if len(tokens) >= 6:
                    cid = int(tokens[0])
                    if len(tokens) == 6:
                        x, y = 0.0, 0.0
                        delivery = float(tokens[1])
                        pickup = float(tokens[2])
                        ready = float(tokens[3])
                        due = float(tokens[4])
                        service = float(tokens[5])
                    else:
                        x = float(tokens[1])
                        y = float(tokens[2])
                        delivery = float(tokens[3])
                        pickup = float(tokens[4])
                        ready = float(tokens[5])
                        due = float(tokens[6])
                        service = float(tokens[7])
                    nodes.append(Node(cid, x, y, delivery, pickup, ready, due, service))
            elif in_matrix:
                tokens = line_str.replace(",", " ").split()
                if len(tokens) >= 4:
                    matrix_lines.append(tokens)

        nodes.sort(key=lambda n: n.id)
        for idx, n in enumerate(nodes):
            n.id = idx

        n_count = dimension if dimension > 0 else len(nodes)
        distance = [[0.0] * n_count for _ in range(n_count)]
        travel_time = [[0.0] * n_count for _ in range(n_count)]

        if matrix_lines:
            for tokens in matrix_lines:
                i, j = int(tokens[0]), int(tokens[1])
                d, t = float(tokens[2]), float(tokens[3])
                if i < n_count and j < n_count:
                    distance[i][j] = d
                    travel_time[i][j] = t
        else:
            for i in range(n_count):
                for j in range(n_count):
                    if i != j and i < len(nodes) and j < len(nodes):
                        dx = nodes[i].x - nodes[j].x
                        dy = nodes[i].y - nodes[j].y
                        d = math.sqrt(dx * dx + dy * dy)
                        distance[i][j] = d
                        travel_time[i][j] = d

        return cls(name, nodes, distance, travel_time, max_vehicles, capacity, dispatch_cost, unit_cost)
