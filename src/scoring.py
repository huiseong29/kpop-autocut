"""Score calculation and component normalization."""

from typing import Dict, List

import numpy as np

from src.config import PipelineConfig, clamp


def calculate_score(
    p_score: float,
    m_score: float,
    q_score: float,
    temporal_penalty: float,
    config: PipelineConfig,
) -> float:
    score = (
        config.alpha * p_score
        - config.beta * m_score
        + config.gamma * q_score
        - config.delta * temporal_penalty
    )
    return clamp(score, 0.0, 1.0)


def compute_temporal_penalty(elapsed_sec: float, min_scene_len_sec: float) -> float:
    return clamp(1.0 - elapsed_sec / min_scene_len_sec, 0.0, 1.0)


def robust_minmax_normalize(values: List[float], eps: float = 1e-8) -> List[float]:
    if not values:
        return values

    arr = np.array(values, dtype=np.float32)
    lo = float(np.percentile(arr, 5))
    hi = float(np.percentile(arr, 95))

    if hi - lo < eps:
        return [0.0 for _ in values]

    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return [float(x) for x in arr]


def normalize_score_components(score_components: List[Dict]) -> List[Dict]:
    """p/m/q를 카메라별로 robust min-max normalization."""
    if not score_components:
        return score_components

    n_cams = len(score_components[0]["p"])
    normalized = [
        {"p": list(c["p"]), "m": list(c["m"]), "q": list(c["q"])}
        for c in score_components
    ]

    for key in ["p", "m", "q"]:
        for cam_idx in range(n_cams):
            series = [c[key][cam_idx] for c in score_components]
            norm_series = robust_minmax_normalize(series)
            for t_idx, value in enumerate(norm_series):
                normalized[t_idx][key][cam_idx] = value

    return normalized


def rebuild_scores_from_components(
    score_components: List[Dict],
    config: PipelineConfig,
) -> List[List[float]]:
    return [
        [
            calculate_score(c["p"][j], c["m"][j], c["q"][j], 0.0, config)
            for j in range(len(c["p"]))
        ]
        for c in score_components
    ]
