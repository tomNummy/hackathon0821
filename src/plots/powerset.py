from ..lsh import norm_vectors
from ..bloom_count import bloom
from .utils import umap_embed

import numpy as np
import pandas as pd
import itertools
from typing import List

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import Figure, figure


def embed(s: pd.Series, bits: int = 8) -> np.array:
    # drop dask hll so we can go down to low bits
    # for the powerset example
    #     hll_embeds = compute_hll_array(s, bits)
    cms_embeds = bloom(s, 2 ** (bits - 3), 2 ** 3)
    return cms_embeds


def make_powerset(n: int) -> List[List[int]]:
    powerset = []
    for size in range(n + 1):
        for set_ in itertools.combinations(range(n), size):
            powerset.append(pd.Series(list(set_)))
    return powerset


def make_powerset_patterns(n: int, bits: int = 3) -> pd.DataFrame:
    sets = make_powerset(n)
    embeds = np.asarray([embed(x, bits) for x in sets])
    embeds = norm_vectors(embeds)

    umbeds = umap_embed(embeds, n_neighbors=20)
    umbeds["set"] = [list(x) for x in sets]
    return umbeds


def make_bokeh_figure(umbeds: pd.DataFrame) -> Figure:
    datasource = ColumnDataSource(umbeds)

    plot_figure = figure(
        title=f"UMAP projection of embedded patterns",
        plot_width=600,
        plot_height=600,
        tools=("pan, wheel_zoom, reset"),
    )

    #     plot_figure.add_layout(Legend(title="Perspective"))

    plot_figure.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div>
            <span style='font-size: 12px; color: #224499'>Rows:</span>
            <span style='font-size: 14px'>@set</span>
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
    )

    return plot_figure
