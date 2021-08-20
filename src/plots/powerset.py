from ..lsh import norm_vectors
from ..bloom_count import bloom
from .utils import umap_embed
from ..pattern_overlap import PatternOverlap

import numpy as np
import pandas as pd
import itertools
from typing import List
import networkx as nx


from bokeh.models import ColumnDataSource, HoverTool, StaticLayoutProvider
from bokeh.plotting import Figure, figure, from_networkx


# def embed(s: pd.Series, bits: int = 8) -> np.array:
#     # drop dask hll so we can go down to low bits
#     # for the powerset example
#     #     hll_embeds = compute_hll_array(s, bits)
#     cms_embeds = bloom(s, 2 ** (bits - 3), 2 ** 3)
#     return cms_embeds


def make_powerset(n: int) -> List[List[int]]:
    powerset = []
    for size in range(n + 1):
        for set_ in itertools.combinations(range(n), size):
            powerset.append(pd.Series(list(set_)))
    return powerset


def make_powerset_patterns(n: int, bits: int = 3) -> pd.DataFrame:
    sets = make_powerset(n)

    po = PatternOverlap(np.asarray(sets))

    embeds = po.embs
    overlaps, neighbor_sets = po.get_overlaps(max_ham_distance=10)

    G = nx.convert_matrix.from_numpy_matrix(overlaps)

    umbeds = umap_embed(embeds, n_neighbors=20)
    umbeds["set"] = [list(x) for x in sets]
    return umbeds, G


def make_bokeh_figure(umbeds: pd.DataFrame, G=None) -> Figure:
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
        "u_x", "u_y", source=datasource, line_alpha=0.6, fill_alpha=0.6, size=12,
    )

    if G:

        graph_renderer = from_networkx(G, nx.spring_layout)

        # fixed_layout = {i: [u.ux, u.uy] for
        fixed_layout = {
            index: [row["u_x"], row["u_y"]] for index, row in umbeds.iterrows()
        }
        fixed_layout_provider = StaticLayoutProvider(graph_layout=fixed_layout)
        graph_renderer.layout_provider = fixed_layout_provider

        plot_figure.renderers.append(graph_renderer)

    return plot_figure
