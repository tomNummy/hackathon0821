import numpy as np
import pandas as pd
from typing import Any, Dict, List, DefaultDict, Optional
from collections import defaultdict

# TODO:
#


def norm_vectors(vs: np.array) -> np.array:
    """Standardize each component of a set of vectors

    Each vector is along the 0 dim
    """
    means = np.mean(vs, axis=0)
    stds = np.std(vs, axis=0)
    return (vs - means) / stds


def array_to_int(v: np.array) -> int:
    r = 0
    for idx, x in enumerate(v):
        if x:
            r |= 1 << idx
    return r


def hamming_dist(x: int, y: int) -> int:
    return bin(x ^ y).count("1")


class RSP:
    _plane: np.array

    def __init__(self, n, seed: int = None) -> None:
        prng = np.random.default_rng(seed)
        self._plane = prng.choice([-1, 1], size=n)

    def hash(self, v: np.array) -> int:
        return np.sign(np.dot(v, self._plane))


class RSPHash:
    _hashers: List[RSP]

    def __init__(
        self, dim: int = None, width: int = None, rsps: List[RSP] = None
    ) -> None:
        if rsps:
            self._hashers = np.array(rsps)
        elif dim and width:
            self._hashers = np.array([RSP(dim) for _ in range(width)])
        else:
            raise ValueError()

    def hash(self, v: np.array) -> int:
        res = np.vectorize(lambda x: x.hash(v))(self._hashers)
        return array_to_int(res)
