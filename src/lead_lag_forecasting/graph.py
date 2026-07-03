"""Lead-lag 페어를 그래프로 보고 community 기반 보정 점수를 계산한다.

페어 그래프에서 동일 community 안에서 잘 연결된 leader 일수록 follower 의 미래
값을 잘 설명할 가능성이 높다는 가정으로, 단순한 graph score 를 후보 정렬에 더한다.
"""

from __future__ import annotations

import community.community_louvain as community_louvain
import networkx as nx
import numpy as np
import pandas as pd

from .config import GraphConfig


def build_pair_graph(pairs: pd.DataFrame, *, score_column: str = "composite_score") -> nx.DiGraph:
    """가중 directed 그래프 (leader → follower)."""

    graph = nx.DiGraph()
    for row in pairs.itertuples(index=False):
        graph.add_edge(
            getattr(row, "leading_item_id"),
            getattr(row, "following_item_id"),
            weight=float(getattr(row, score_column)),
            best_lag=int(getattr(row, "best_lag")),
        )
    return graph


def annotate_with_graph_score(pairs: pd.DataFrame, config: GraphConfig) -> pd.DataFrame:
    """후보 데이터프레임에 community / pagerank 기반 graph score 를 추가한다."""

    if pairs.empty:
        return pairs.assign(graph_score=[], community_id=[])

    graph = build_pair_graph(pairs)
    undirected = graph.to_undirected()
    partition = community_louvain.best_partition(undirected, resolution=config.community_resolution, random_state=42)
    page_rank = nx.pagerank(graph, weight="weight")

    same_community = pairs.apply(
        lambda row: int(partition.get(row["leading_item_id"], -1) == partition.get(row["following_item_id"], -2)),
        axis=1,
    )
    leader_authority = pairs["leading_item_id"].map(page_rank).fillna(0.0)
    follower_authority = pairs["following_item_id"].map(page_rank).fillna(0.0)

    annotated = pairs.copy()
    annotated["leader_pagerank"] = leader_authority.astype(np.float32)
    annotated["follower_pagerank"] = follower_authority.astype(np.float32)
    annotated["same_community"] = same_community.astype(np.int8)
    annotated["community_id"] = pairs["leading_item_id"].map(partition).astype("Int32")
    annotated["graph_score"] = (
        annotated["composite_score"]
        + 5.0 * annotated["same_community"].astype(float)
        + 50.0 * annotated["leader_pagerank"]
    )
    annotated = annotated.sort_values("graph_score", ascending=False).reset_index(drop=True)
    annotated = annotated[annotated["graph_score"] >= config.score_threshold].reset_index(drop=True)
    return annotated
