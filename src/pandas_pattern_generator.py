import pandas as pd
import numpy as np

class PandasPatternGenerator():
    def __init__(self,n,k,hll_bits):
        self.n = n
        self.k = k
        self.hll_bits = hll_bits
        self.pattern_index = range(1,self.k+1)
        self.patterns = {i: self.generate_pattern(n,i) for i in self.pattern_index}
        self.overlaps = pd.DataFrame(
            index=self.pattern_index,
            columns=self.pattern_index,
            data=[
                [self.get_overlap(i,j) if i!=j else n//i for i in self.pattern_index] for j in self.pattern_index
            ]
        )
        self.hll_patterns = {k: self.compute_hll_array(v,self.hll_bits) for k,v in self.patterns.items()}

    def generate_pattern(self,n,i):
        return pd.Series(range(0,n,i))

    def get_overlap(self,i,j):
        pat_1 = self.patterns[i]
        pat_2 = self.patterns[j]
        return len(pat_1.to_frame().merge(pat_2.to_frame(),on=0))

    def compute_first_bit(self,a):
        "Compute the position of the first nonzero bit for each int in an array."
        # TODO: consider making this less memory-hungry
        bits = np.bitwise_and.outer(a, 1 << np.arange(32))
        bits = bits.cumsum(axis=1).astype(bool)
        return 33 - bits.sum(axis=1)

    def compute_hll_array(self, obj, b):
        # b is the number of bits

        if not 8 <= b <= 16:
            raise ValueError("b should be between 8 and 16")
        num_bits_discarded = 32 - b
        m = 1 << b

        # Get an array of the hashes
        hashes = hash_pandas_object(obj, index=False)
        if isinstance(hashes, pd.Series):
            hashes = hashes._values
        hashes = hashes.astype(np.uint32)

        # Of the first b bits, which is the first nonzero?
        j = hashes >> num_bits_discarded
        first_bit = self.compute_first_bit(hashes)

        # Pandas can do the max aggregation
        df = pd.DataFrame({"j": j, "first_bit": first_bit})
        series = df.groupby("j").max()["first_bit"]

        # Return a dense array so we can concat them and get a result
        # that is easy to deal with
        return series.reindex(np.arange(m), fill_value=0).values.astype(np.uint8)
