# 산업 지표 Lead-Lag 예측 경진대회

[![Python](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org) [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

산업 품목의 월간 무역 시계열에서 선행·후행 관계를 찾고, 후행 품목의 다음 달 무역량을 예측했습니다. 선후행 쌍 판별과 회귀를 함께 다뤄 예선 상위 5.2%를 기록했습니다.

## 개요

| 항목 | 내용 |
| --- | --- |
| 대회 | 제3회 국민대학교 AI빅데이터 분석 경진대회 |
| 기간 | 2025.11.10 – 2025.11.28 (예선) |
| 주최 / 주관 | 국민대학교 경영대학원·한국기계산업진흥회 / 국민대학교 경영대학·경영대학원 (운영: 데이콘) |
| 평가지표 | 0.6·F1 + 0.4·(1−NMAE) |
| 결과 | 예선 상위 5.2% |
| 과제 | 선후행 쌍 판별 + 후행 품목 다음 달(2025.8) 무역량 예측 |
| 모델 | LightGBM |

## 접근

- 품목별 월간 pivot을 만들고 lag 1–9 구간에서 상관, p-value, 방향성, 안정성을 스캔했습니다.
- zero ratio, 변동계수, spike, 추세, autocorrelation 등 품질 점수로 우연한 상관을 걸렀습니다.
- 선행→후행 방향 그래프를 만들고 PageRank와 community 정보로 쌍 점수를 보정했습니다.
- 후행 품목의 lag·rolling 피처와 선택된 선행 품목의 lag-shifted 값을 결합해 LightGBM 회귀 모델을 학습했습니다.
- 마지막 3개월을 hold-out으로 두고 NMAE와 MAE를 확인했습니다.

## 저장소 구성

```text
.
├── configs/default.yaml
├── scripts/
│   ├── discover_pairs.py
│   ├── train.py
│   └── infer.py
├── src/lead_lag_forecasting/
│   ├── leadlag/
│   ├── graph.py
│   ├── features.py
│   ├── models.py
│   ├── pipeline.py
│   └── cli.py
├── pyproject.toml
└── requirements.txt
```

## 실행

`train.csv`, `sample_submission.csv`를 `data/`에 둡니다.

```bash
pip install -e .
python -m lead_lag_forecasting.cli discover --config configs/default.yaml
python -m lead_lag_forecasting.cli train --config configs/default.yaml
python -m lead_lag_forecasting.cli infer --config configs/default.yaml
```

## 공개 범위

대회 원본 데이터, 생성한 pair table, 모델 파일, 제출 파일은 포함하지 않았습니다. 데이터는 대회 참가자에게만 제공되어 그대로 재현하기는 어렵고, 저장소에는 선후행 탐색과 예측 파이프라인만 남겼습니다.

## 링크

- [대회 페이지](https://dacon.io/competitions/official/236619/overview/description)
- [최종 리더보드](https://dacon.io/competitions/official/236619/leaderboard)

## 라이선스

Apache License 2.0. 자세한 내용은 [LICENSE](LICENSE)에 있습니다.
