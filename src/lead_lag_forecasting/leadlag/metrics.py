"""두 시계열의 lead-lag 관계를 평가하는 지표 모음.

모든 함수는 1D numpy 배열을 받고, 가능한 lag 범위에서 안전한 fallback (NaN/-inf)
을 반환하도록 통일했다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler


def pearson_with_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """0 분산 입력에 대해 안전한 pearson 상관/유의확률."""

    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    corr, p = pearsonr(x, y)
    if np.isnan(corr):
        return 0.0, 1.0
    return float(corr), float(p)


def _align_pair(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray] | None:
    if lag <= 0 or len(x) <= lag or len(y) <= lag:
        return None
    return x[:-lag], y[lag:]


@dataclass(frozen=True)
class LagSearchResult:
    best_lag: int
    correlation: float
    p_value: float


def best_lag_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> LagSearchResult:
    """lag ∈ [1, max_lag] 에서 절댓값 상관이 최대인 지점을 찾는다."""

    best = LagSearchResult(best_lag=0, correlation=0.0, p_value=1.0)
    for lag in range(1, max_lag + 1):
        aligned = _align_pair(x, y, lag)
        if aligned is None:
            continue
        corr, p = pearson_with_pvalue(*aligned)
        if abs(corr) > abs(best.correlation):
            best = LagSearchResult(best_lag=lag, correlation=corr, p_value=p)
    return best


def normalized_dtw(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """O(n*m) 동적 시간 와핑 거리. 정렬된 좌우 윈도우를 표준화한 뒤 길이로 나눈다."""

    aligned = _align_pair(x, y, lag)
    if aligned is None:
        return float("inf")
    x_aligned, y_aligned = aligned
    if len(x_aligned) < 4:
        return float("inf")

    scaler = StandardScaler()
    nx = scaler.fit_transform(x_aligned.reshape(-1, 1)).flatten()
    ny = scaler.fit_transform(y_aligned.reshape(-1, 1)).flatten()

    n, m = len(nx), len(ny)
    matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(nx[i - 1] - ny[j - 1])
            matrix[i, j] = cost + min(matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1])
    return float(matrix[n, m] / max(n, 1))


def trend_alignment(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """1차 차분 부호 일치 비율 — 두 시계열의 방향성이 얼마나 같이 움직이는지."""

    aligned = _align_pair(x, y, lag)
    if aligned is None:
        return 0.0
    x_aligned, y_aligned = aligned
    if len(x_aligned) < 3:
        return 0.0
    x_diff = np.diff(x_aligned)
    y_diff = np.diff(y_aligned)
    if len(x_diff) == 0:
        return 0.0
    return float(np.mean(np.sign(x_diff) == np.sign(y_diff)))


def direction_match(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """변화량 부호 일치율의 0~1 정규화 버전."""

    return trend_alignment(x, y, lag)


def persistence_score(x: np.ndarray, y: np.ndarray, lag: int, *, recent_window: int = 6) -> float:
    """최근 윈도우와 전체 구간의 상관이 얼마나 일관되게 유지되는지."""

    aligned = _align_pair(x, y, lag)
    if aligned is None:
        return 0.0
    x_aligned, y_aligned = aligned
    if len(x_aligned) <= recent_window:
        return 0.0
    full_corr, _ = pearson_with_pvalue(x_aligned, y_aligned)
    recent_corr, _ = pearson_with_pvalue(x_aligned[-recent_window:], y_aligned[-recent_window:])
    if full_corr == 0:
        return 0.0
    return float(min(recent_corr / full_corr, 2.0))
