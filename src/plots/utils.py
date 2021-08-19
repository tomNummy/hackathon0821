import numpy as np
import umap
import pandas as pd
from typing import Any


def umap_embed(
    embeds: np.array,
    n_neighbors: int = 5,
    umap_kwargs: Any = None,
) -> pd.DataFrame:

    umap_kwargs = umap_kwargs if umap_kwargs else {}
    reducer = umap.UMAP(n_neighbors=n_neighbors, **umap_kwargs)
    embedding = reducer.fit_transform(np.nan_to_num(embeds))

    epts = pd.DataFrame(embedding, columns=["u_x", "u_y"])
    return epts
