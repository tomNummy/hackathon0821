from src.lsh import LSHashMap, hamming_dist, norm_vectors
from src.bloom_count import bloom

import pandas as pd
import numpy as np
from dask.dataframe.hyperloglog import compute_hll_array, reduce_state, estimate_count


class PatternOverlap:
    def __init__(
        self,
        patterns,
        BITS: int = 8,
        LSHwidth: int = 16,
        LSHseed: int = 123456,
        mBloom: int = 0,
        kBloom: int = 0,
        normalize_embeds: bool = True,
    ):

        self.BITS = BITS
        self.patterns = patterns
        self.NPats = len(patterns)

        if mBloom == 0:
            mBloom = 2 ** (self.BITS - 3)

        if kBloom == 0:
            kBloom = 2 ** 3

        self.mBloom = mBloom
        self.kBloom = kBloom

        # compute and store embeddings (also store hll embeddings for later)
        self.embs, self.hlls = self.embeds(normalize_embeds)

        # compute distance between embeddings using lsh
        self.lsh = LSHashMap(self.embs, width=LSHwidth, seed=LSHseed)

        return

    def embeds(self, normalize_embeds: bool = True) -> np.array:

        hll_embeds = [compute_hll_array(s, self.BITS) for s in self.patterns]
        cms_embeds = [bloom(s, self.mBloom, self.kBloom) for s in self.patterns]

        concat_embeds = norm_vectors(
            np.asarray([np.concatenate([h, c]) for h, c in zip(hll_embeds, cms_embeds)])
        )

        if normalize_embeds:
            embed_norm = np.sqrt(np.sum(concat_embeds ** 2, axis=1))
            concat_embeds = concat_embeds / embed_norm[:, None]

        return concat_embeds, np.asarray(hll_embeds)

    def get_overlaps(self, max_ham_distance: int = 0):

        overlaps = np.zeros((self.NPats, self.NPats), dtype=int)
        neighbor_sets = set()

        for i in range(self.NPats):

            neighbors = self.lsh.get_neighbors(self.embs[i, :])
            neighbors2 = []
            for d in range(1, max_ham_distance + 1):

                neighbor_d = self.lsh.get_neighbors(self.embs[i, :], d)
                neighbors.extend(neighbor_d)
                if neighbor_d:
                    neighbors2.append(neighbor_d)

            neighbors.sort()
            neighbor_sets.add(tuple(neighbors))

            for j in neighbors:
                if j > i:

                    hll_union = reduce_state(
                        np.concatenate(((self.hlls[i, :], self.hlls[j, :]))),
                        self.BITS,
                    )
                    union_size = estimate_count(hll_union, self.BITS)
                    overlaps[i, j] = (
                        len(self.patterns[i]) + len(self.patterns[j]) - union_size
                    )
                elif i > j:
                    overlaps[i, j] = overlaps[j, i]

            overlaps[i, i] = len(self.patterns[i])

        return overlaps, neighbor_sets
