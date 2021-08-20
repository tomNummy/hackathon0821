import numpy as np
import pandas as pd
import itertools
from typing import Any, Dict, List, DefaultDict, Optional


def norm_vectors(vs: np.array) -> np.array:
    """Standardize each component of a set of vectors

    Each vector is along the 0 dim
    """

    means = np.mean(vs, axis=0)
    stds = np.std(vs, axis=0)
    res = (vs - means) / stds
    return np.nan_to_num(res)


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
        self,
        dim: int = None,
        width: int = None,
        rsps: List[RSP] = None,
        seed: int = None,
    ) -> None:
        if rsps:
            self._hashers = np.array(rsps)
        elif dim and width:
            self._hashers = np.array([RSP(dim, seed=seed + i) for i in range(width)])
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
    bins: Dict[int, List[int]]

    def __init__(self, vects: np.array, width: int = 16, seed: int = None):
        dim = vects.shape[1]
        self._rsp = RSPHash(dim=dim, width=width, seed=seed)
        self.vects = vects

        hashes = self._rsp.batch_hash(self.vects)
        df = pd.DataFrame({"idx": np.arange(len(hashes)), "hashes": hashes})
        self.bins = df.groupby("hashes").apply(lambda x: list(x["idx"])).to_dict()

    def __getitem__(self, key: int) -> List[int]:
        # maybe should swap this with `get_bucket`?
        return self.bins[key]

    def items(self):
        return self.bins.items()

    def keys(self):
        return self.bins.keys()

    def values(self):
        return self.bins.values()

    def get_bucket_id(self, v: np.array) -> int:
        """Get the bucket id for a given vector"""
        return self._rsp.hash(v)

    def get_neighbors(self, v: np.array, d: int = 0) -> List[int]:
        """Get the neighbors in the buckets a hamming distance `d` away"""
        b_id = self.get_bucket_id(v)
        neighbors = []
        for k in self.keys():
            if hamming_dist(b_id, k) == d:
                neighbors.extend(self[k])
        return neighbors

    def key_distances(self):
        h_dists = np.asarray(
            [hamming_dist(x, y) for x, y in itertools.product(self.keys(), repeat=2)]
        )
        return h_dists.reshape(len(self.bins), len(self.bins))

    def histogram(self, bins: str = "auto"):
        return np.histogram(np.array([len(x) for x in self.values()]), bins=bins)
