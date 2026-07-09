"""Lead-lag 페어의 품질 점수.

상관·DTW 만으로는 "우연히 관계가 있어 보이는 시계열" 을 거르지 못한다. 본 모듈은
다음 8개 항목으로 구성된 휴리스틱 품질 점수(0~85점) 를 계산해 페어 후보 정렬에
사용한다.

| 점수 항목 | 의미 |
| --- | --- |
| zero_ratio | 0 비율이 너무 높은 시계열 페널티 |
| coefficient_of_variation | 변동계수 — 너무 적은 분산도 페널티 |
| spike | 큰 outlier(spike) 페널티 |
| stability | 전반/후반부 상관 일관성 |
| trend | 두 시계열 추세 부호 일치 |
| direction | 변화량 부호 일치 |
| significance | p-value 기반 유의성 |
| autocorr | follower 의 자기상관이 적정 범위인지 |
"""

from __future__ import annotations

import numpy as np

from .metrics import pearson_with_pvalue


def quality_score(leader: np.ndarray, follower: np.ndarray, lag: int) -> dict[str, float]:
    scores: dict[str, float] = {}

    # 1) Zero ratio
    zero_ratio_a = float((leader == 0).mean())
    zero_ratio_b = float((follower == 0).mean())
    max_zero = max(zero_ratio_a, zero_ratio_b)
    scores["zero"] = _bucket(max_zero, [(0.17, 10.0), (0.37, 7.0), (0.57, 4.0)], default=0.0)

    # 2) Coefficient of variation
    cv_leader = float(np.std(leader) / (np.mean(leader) + 1.0)) if np.mean(leader) > 0 else 0.0
    cv_follower = float(np.std(follower) / (np.mean(follower) + 1.0)) if np.mean(follower) > 0 else 0.0
    max_cv = max(cv_leader, cv_follower)
    scores["coefficient_of_variation"] = _bucket(max_cv, [(0.75, 10.0), (1.40, 7.0), (2.70, 4.0)], default=0.0)

    # 3) Spike
    leader_changes = np.abs(np.diff(leader))
    follower_changes = np.abs(np.diff(follower))
    if leader_changes.size and follower_changes.size:
        spike_leader = leader_changes.max() / (np.median(leader_changes) + 1.0)
        spike_follower = follower_changes.max() / (np.median(follower_changes) + 1.0)
        max_spike = max(spike_leader, spike_follower)
        scores["spike"] = _bucket(max_spike, [(4.5, 10.0), (9.5, 7.0), (17.0, 4.0)], default=0.0)
    else:
        scores["spike"] = 5.0

    # 4) Stability — 전반/후반 상관 일관성
    half = len(leader) // 2
    if half >= 12 and half + lag < len(follower):
        first_corr = np.corrcoef(leader[:half], follower[lag : half + lag])[0, 1]
        second_corr = np.corrcoef(leader[half:], follower[half + lag :])[0, 1]
        if not (np.isnan(first_corr) or np.isnan(second_corr)):
            consistency = 1 - abs(first_corr - second_corr)
            scores["stability"] = _bucket(consistency, [(0.52, 15.0), (0.35, 10.0), (0.22, 6.0)], default=3.0, ascending=False)
        else:
            scores["stability"] = 5.0
    else:
        scores["stability"] = 5.0

    # 5) Trend
    if leader.size > 1 and follower.size > 1:
        leader_slope = np.polyfit(np.arange(len(leader)), leader, 1)[0] if np.std(leader) > 0 else 0.0
        follower_slope = np.polyfit(np.arange(len(follower)), follower, 1)[0] if np.std(follower) > 0 else 0.0
        if leader_slope * follower_slope > 0:
            ratio = abs(leader_slope) / (abs(follower_slope) + 1e-3)
            scores["trend"] = _bucket(ratio, [(0.38, 10.0), (4.20, 6.0)], default=3.0, between=(0.38, 2.7, 0.25, 4.2))
        elif leader_slope == 0 or follower_slope == 0:
            scores["trend"] = 5.0
        else:
            scores["trend"] = 2.0
    else:
        scores["trend"] = 4.0

    # 6) Direction match
    if leader_changes.size > lag and follower_changes.size > lag:
        match = float((np.sign(leader_changes[:-lag]) == np.sign(follower_changes[lag:])).mean())
        scores["direction"] = _bucket(match, [(0.64, 10.0), (0.54, 7.0), (0.44, 4.0)], default=1.0, ascending=False)
    else:
        scores["direction"] = 4.0

    # 7) Significance
    if len(leader) > lag:
        _, p = pearson_with_pvalue(leader[:-lag], follower[lag:])
        scores["significance"] = _bucket(p, [(0.007, 10.0), (0.025, 7.0), (0.07, 4.0)], default=1.0)
    else:
        scores["significance"] = 4.0

    # 8) Autocorrelation of follower
    if len(follower) > 1:
        autocorr = np.corrcoef(follower[:-1], follower[1:])[0, 1]
        if not np.isnan(autocorr):
            if 0.24 < autocorr < 0.86:
                scores["autocorr"] = 10.0
            elif 0.09 < autocorr < 0.91:
                scores["autocorr"] = 7.0
            else:
                scores["autocorr"] = 3.0
        else:
            scores["autocorr"] = 4.0
    else:
        scores["autocorr"] = 4.0

    scores["total"] = float(sum(scores.values()))
    return scores


def _bucket(
    value: float,
    thresholds: list[tuple[float, float]],
    default: float,
    *,
    ascending: bool = True,
    between: tuple[float, float, float, float] | None = None,
) -> float:
    """`thresholds` 는 (cutoff, score) 튜플의 리스트."""

    if between is not None:
        low_in, high_in, low_out, high_out = between
        if low_in < value < high_in:
            return thresholds[0][1]
        if low_out < value < high_out:
            return thresholds[-1][1]
        return default

    if ascending:
        for cutoff, score in thresholds:
            if value < cutoff:
                return score
        return default

    for cutoff, score in thresholds:
        if value > cutoff:
            return score
    return default
