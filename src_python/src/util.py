import math


def split(s: str, delimiter: str):
    return s.split(delimiter)


def trim(s: str) -> str:
    return s.strip()


def mean(values, start_index: int, length: int) -> float:
    total = 0.0
    for i in range(start_index, start_index + length):
        total += values[i]
    return total / float(length)


def chk_p_square(x: float) -> bool:
    sr = math.sqrt(x)
    return (sr - math.floor(sr)) == 0


def argsort(values, indices, length: int) -> None:
    order = sorted(range(length), key=lambda i: values[i])
    if len(indices) < length:
        indices.extend([0] * (length - len(indices)))
    for i in range(length):
        indices[i] = order[i]


def randint(from_value: int, end_value: int, rng) -> int:
    return rng.randint(from_value, end_value)


def rand(from_value: float, end_value: float, rng) -> float:
    return rng.uniform(from_value, end_value)
