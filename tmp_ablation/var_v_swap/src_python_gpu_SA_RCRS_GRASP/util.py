import math
import random

def trim(s: str) -> str:
    return s.strip()

def split(s: str, delim: str):
    return [trim(p) for p in s.split(delim)]

def argsort(fit_list, argrank, length=None):
    if length is None:
        length = len(fit_list)
    indices = list(range(length))
    indices.sort(key=lambda i: fit_list[i])
    for i, idx in enumerate(indices):
        argrank[i] = idx

def mean(fit_list, start=0, end=None):
    if end is None:
        end = len(fit_list)
    if start >= end:
        return 0.0
    sub = fit_list[start:end]
    return sum(sub) / len(sub)

def rand(low: float, high: float, rng: random.Random) -> float:
    return rng.uniform(low, high)

def randint(low: int, high: int, rng: random.Random) -> int:
    return rng.randint(low, high)

def chk_p_square(p_size: int) -> bool:
    sr = math.isqrt(p_size)
    return sr * sr == p_size
