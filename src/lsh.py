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
        res = np.vectorize(lambda x: x.hash(v))(self._hashers) > 0
        return array_to_int(res)

    def batch_hash(self, vs: np.array) -> np.array:
        """Hash a set of vectors and return an array of uint hashes"""
        return np.vectorize(lambda x: self.hash(x), signature="(n)->()")(vs)


class LSHashMap:
    _rsp: RSPHash
    _bins: Dict[int, List[int]]

    def __init__(self, dim: int, width: int):
        self._rsp = RSPHash(dim=dim, width=width)
        self._bins = {}

    def build(self, vs: np.array) -> None:
        hashes = self._rsp.batch_hash(vs)
        df = pd.DataFrame({"idx": np.arange(len(hashes)), "hashes": hashes})
        self._bins = df.groupby("hashes").apply(lambda x: list(x["idx"])).to_dict()

    def __getitem__(self, key: int) -> List[int]:
        return self._bins[key]

    def items(self):
        return self._bins.items()

    def keys(self):
        return self._bins.keys()

    def values(self):
        return self._bins.values()
