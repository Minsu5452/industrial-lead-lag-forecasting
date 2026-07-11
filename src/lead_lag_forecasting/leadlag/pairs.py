"""후보 lead-lag 페어 발굴.

모든 (i, j) 페어에 대해 best lag 의 상관을 계산하기 때문에 item 수가 크면 연산 비용이 크다.
실전에서는 후보를 줄이는 사전 필터(예: top item 만 사용) 로 부담을 낮춘다.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics import (
    best_lag_correlation,
    normalized_dtw,
    persistence_score,
    trend_alignment,
)
from .quality import quality_score


def build_candidate_pairs(
    monthly_pivot: pd.DataFrame,
    *,
    min_overlap_months: int = 18,
    max_lag: int = 9,
    min_abs_corr: float = 0.30,
    max_pvalue: float = 0.10,
    progress: bool = True,
) -> pd.DataFrame:
    """`(date × item)` pivot 으로부터 점수 매겨진 후보 페어 데이터프레임을 만든다."""

    items: list = list(monthly_pivot.columns)
    if len(items) < 2:
        return pd.DataFrame()

    series_array = monthly_pivot.to_numpy(dtype=np.float64).T  # shape (n_items, n_months)
    item_to_index = {item: idx for idx, item in enumerate(items)}

    rows: list[dict] = []
    pair_iter = combinations(items, 2)
    if progress:
        total = len(items) * (len(items) - 1) // 2
        pair_iter = tqdm(pair_iter, total=total, desc="lead-lag scan")

    for left_item, right_item in pair_iter:
        left = series_array[item_to_index[left_item]]
        right = series_array[item_to_index[right_item]]

        valid_mask = (left + right) != 0
        if valid_mask.sum() < min_overlap_months:
            continue

        for leader_item, leader, follower_item, follower in (
            (left_item, left, right_item, right),
            (right_item, right, left_item, left),
        ):
            best = best_lag_correlation(leader, follower, max_lag=max_lag)
            if best.best_lag == 0:
                continue
            if abs(best.correlation) < min_abs_corr or best.p_value > max_pvalue:
                continue

            quality = quality_score(leader, follower, best.best_lag)
            persistence = persistence_score(leader, follower, best.best_lag)
            direction = trend_alignment(leader, follower, best.best_lag)
            dtw = normalized_dtw(leader, follower, best.best_lag)

            rows.append(
                {
                    "leading_item_id": leader_item,
                    "following_item_id": follower_item,
                    "best_lag": int(best.best_lag),
                    "max_corr": float(best.correlation),
                    "p_value": float(best.p_value),
                    "quality_score": float(quality["total"]),
                    "persistence_score": float(persistence),
                    "direction_score": float(direction),
                    "dtw_distance": float(dtw),
                }
            )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates

    candidates["composite_score"] = (
        70.0 * candidates["max_corr"].abs()
        + 20.0 * (candidates["p_value"] < 0.01).astype(float)
        + 10.0 * (candidates["best_lag"] <= 3).astype(float)
        + 0.5 * candidates["quality_score"]
        + 5.0 * candidates["direction_score"]
        + 5.0 * candidates["persistence_score"].clip(lower=0)
    )
    candidates = candidates.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return candidates
