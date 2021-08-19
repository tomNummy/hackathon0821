import pandas as pd
import numpy as np


class PandasPatternGenerator:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.pattern_index = range(1, self.k + 1)
        self.patterns = {i: self.generate_pattern(i) for i in self.pattern_index}
        self.overlaps = pd.DataFrame(
            index=self.pattern_index,
            columns=self.pattern_index,
            data=[
                [
                    self.get_overlap(i, j) if i != j else n // i
                    for i in self.pattern_index
                ]
                for j in self.pattern_index
            ],
        )

    def generate_pattern(self, i: int) -> pd.Series:
        return pd.Series(range(0, self.n, i))

    def get_overlap(self, i, j):
        pat_1 = self.patterns[i]
        pat_2 = self.patterns[j]
        return len(pat_1.to_frame().merge(pat_2.to_frame(), on=0))
