import numpy as np
import pandas as pd
import math


def bloom(s: pd.Series, m: int, k: int = 0) -> np.array:

    n = s.shape[0]

    if k == 0:

        k = int(math.ceil(m / n * math.log(2)))
        print(f"No k received, optimal k set to {k}")

    bc = np.zeros((k, m), dtype=int)

    """
    hash function that changes with h
    """

    def hashing(s: pd.Series, h: int):
        hstring = str(h)
        hashkey = "0" * (16 - len(hstring)) + hstring
        hash = (
            pd.util.hash_pandas_object(
                s.astype(str), index=False, hash_key=hashkey
            ).to_numpy()
            % m
        )

        return hash

    for j in range(k):  # iterate through k different hash functions

        hashidx = hashing(s, j)

        for i in hashidx:
            bc[j, i] += 1

    return bc.flatten()
