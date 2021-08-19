from ..lsh import norm_vectors
from ..bloom_count import bloom
from .utils import umap_embed

import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, Any
from dask.dataframe.hyperloglog import compute_hll_array
from sklearn.cluster import DBSCAN

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import Figure, figure
from bokeh.palettes import Category20
from bokeh.transform import factor_cmap


def load_patterns(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def embed(s: pd.Series, bits: int = 8) -> np.array:
    hll_embeds = compute_hll_array(s, bits)
    cms_embeds = bloom(s, 2 ** (bits - 3), 2 ** 3)
    return np.concatenate([hll_embeds, cms_embeds])


def build_embeds(patterns, bits: int = 8) -> pd.DataFrame:
    uids = []
    embeds = []
    hows = []
    for idx, (uid, pdict) in enumerate(patterns.items()):
        uids.append(uid)
        hows.append(json.dumps(pdict["hows"], indent=2))
        embeds.append(embed(pdict["row_set"], bits))
    nembeds = np.asarray(embeds)
    nembeds = norm_vectors(embeds)
    return pd.DataFrame({"uid": uids, "embeds": list(nembeds), "how": hows})


def load(path: str) -> pd.DataFrame:
    pats = load_patterns(path)
    return build_embeds(pats)


def make_plot(
    patterns: pd.DataFrame, bits: int = 3, dbscan_kwargs: Any = None
) -> pd.DataFrame:
    umbeds = umap_embed(np.vstack(patterns["embeds"].values), n_neighbors=20)
    umbeds["how"] = patterns["how"]

    if not dbscan_kwargs:
        dbscan_kwargs = {}

    X = umbeds[["u_x", "u_y"]].values
    clustering = DBSCAN(**dbscan_kwargs).fit(X)
    umbeds["cluster"] = clustering.labels_

    return make_bokeh_figure(umbeds)


def make_bokeh_figure(df: pd.DataFrame) -> Figure:
    umbeds = df.copy()
    umbeds["cluster"] = umbeds["cluster"].astype(str)

    cmap = factor_cmap(
        "cluster",
        palette=Category20[len(umbeds["cluster"].unique())],
        factors=umbeds["cluster"].unique(),
    )

    datasource = ColumnDataSource(umbeds)
    plot_figure = figure(
        title=f"UMAP projection of embedded patterns",
        plot_width=600,
        plot_height=600,
        tools=("pan, wheel_zoom, reset"),
    )

    plot_figure.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div>
            <span style='font-size: 12px; color: #224499'>Filters:</span>
            <span style='font-size: 14px'>@how</span>
        </div>
    </div>
    """
        )
    )

    plot_figure.circle(
        "u_x",
        "u_y",
        source=datasource,
        line_alpha=0.6,
        fill_alpha=0.6,
        size=12,
        color=cmap,
    )

    return plot_figure
