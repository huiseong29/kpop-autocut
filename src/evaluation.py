"""GT loading, metrics, baselines, ablation, and grid search."""

import csv
import json
import os
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    GRID_ALPHA_VALUES,
    GRID_BETA_VALUES,
    GRID_DELTA_VALUES,
    GRID_GAMMA_VALUES,
    GT_CUTS_PATH,
    REPORT_DIR,
    PipelineConfig,
)
from src.scoring import rebuild_scores_from_components
from src.selection import find_pose_cut_candidates, select_camera_topk


def load_gt_cuts(path: str = GT_CUTS_PATH) -> Optional[List[Tuple[float, int]]]:
    """GT 컷 파일을 불러온다.

    지원:
    - CSV: time_sec, camera(optional)
    - JSON: [{"time_sec": 10.0, "camera": 1}, ...]
    """
    if not os.path.exists(path):
        print(f"[Evaluation] GT file not found: {path}. Evaluation MAE/acc will be skipped.")
        return None

    ext = os.path.splitext(path)[1].lower()
    gt: List[Tuple[float, int]] = []

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            gt.append((float(item["time_sec"]), int(item.get("camera", -1))))
        return gt

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_key = "time_sec" if "time_sec" in row else "time"
            cam_key = "camera" if "camera" in row else "cam"
            gt.append((float(row[time_key]), int(row.get(cam_key, -1) or -1)))

    return gt


def evaluate(
    chosen: List[int],
    times: List[float],
    gt_cuts: Optional[List[Tuple[float, int]]] = None,
    fps: float = 30.0,
) -> Dict:
    n_cuts = sum(1 for i in range(1, len(chosen)) if chosen[i] != chosen[i - 1])
    result = {"n_cuts": n_cuts, "total_steps": len(chosen)}

    if gt_cuts is None:
        result.update({"MAE_frames": None, "acc_15f": None, "acc_30f": None})
        return result

    pred_cuts = [
        (times[i], chosen[i])
        for i in range(1, len(chosen))
        if chosen[i] != chosen[i - 1]
    ]

    errors_sec = []
    for gt_t, _ in gt_cuts:
        if not pred_cuts:
            errors_sec.append(float("inf"))
            continue
        errors_sec.append(min(abs(pt - gt_t) for pt, _ in pred_cuts))

    errors_f = [e * fps for e in errors_sec]
    mae = float(np.mean(errors_f)) if errors_f else float("nan")
    acc_15f = float(np.mean([e <= 15 for e in errors_f])) if errors_f else float("nan")
    acc_30f = float(np.mean([e <= 30 for e in errors_f])) if errors_f else float("nan")

    result.update(
        {
            "MAE_frames": round(mae, 2),
            "acc_15f": round(acc_15f, 4),
            "acc_30f": round(acc_30f, 4),
        }
    )
    return result


def compute_topk_hit_rate(chosen: List[int], scores: List[List[float]], k: int = 3) -> float:
    hits = 0

    for i, cam in enumerate(chosen):
        top_k = sorted(range(len(scores[i])), key=lambda x: scores[i][x], reverse=True)[:k]
        if cam in top_k:
            hits += 1

    rate = hits / max(len(chosen), 1)
    print(f"[Evaluation] Top-{k} Hit Rate: {rate:.4f} ({hits}/{len(chosen)})")
    return rate


def run_baseline_random(
    times: List[float],
    n_cams: int,
    config: PipelineConfig,
    seed: int = 42,
) -> List[int]:
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(0, n_cams))]
    last_sw = times[0]

    for i in range(1, len(times)):
        if times[i] - last_sw >= config.min_scene_len_sec:
            chosen.append(int(rng.integers(0, n_cams)))
            last_sw = times[i]
        else:
            chosen.append(chosen[-1])

    return chosen


def run_baseline_uniform(times: List[float], n_cams: int, config: PipelineConfig) -> List[int]:
    chosen = [0]
    last_sw = times[0]
    cam = 0

    for i in range(1, len(times)):
        if times[i] - last_sw >= config.min_scene_len_sec:
            cam = (cam + 1) % n_cams
            last_sw = times[i]
            chosen.append(cam)
        else:
            chosen.append(chosen[-1])

    return chosen


def run_baseline_pose_only(
    times: List[float],
    score_components: List[Dict],
    pose_cut_flags: List[bool],
    config: PipelineConfig,
) -> List[int]:
    chosen = [int(np.argmax(score_components[0]["p"]))]
    cur = chosen[0]
    last_sw = times[0]

    for i in range(1, len(times)):
        p_scores = score_components[i]["p"]
        best = int(np.argmax(p_scores))
        elapsed = times[i] - last_sw

        if (
            best != cur
            and elapsed >= config.min_scene_len_sec
            and pose_cut_flags[i]
            and (p_scores[best] - p_scores[cur]) > config.switch_threshold
        ):
            cur = best
            last_sw = times[i]

        chosen.append(cur)

    return chosen


def run_baseline_quality_only(
    times: List[float],
    score_components: List[Dict],
    pose_cut_flags: List[bool],
    config: PipelineConfig,
) -> List[int]:
    chosen = [int(np.argmax(score_components[0]["q"]))]
    cur = chosen[0]
    last_sw = times[0]

    for i in range(1, len(times)):
        q_scores = score_components[i]["q"]
        best = int(np.argmax(q_scores))
        elapsed = times[i] - last_sw

        if (
            best != cur
            and elapsed >= config.min_scene_len_sec
            and pose_cut_flags[i]
            and (q_scores[best] - q_scores[cur]) > config.switch_threshold
        ):
            cur = best
            last_sw = times[i]

        chosen.append(cur)

    return chosen


def run_ablation(
    times: List[float],
    score_components: List[Dict],
    inter_cam_sims: List[float],
    pose_history: List[List[Optional[np.ndarray]]],
    config: PipelineConfig,
    gt_cuts: Optional[List[Tuple[float, int]]] = None,
) -> Dict[str, Dict]:
    """Full / w/o Motion / w/o Quality / w/o Temporal 비교."""
    ablation_configs = {
        "Full": config,
        "w/o Motion": replace(config, beta=0.0, alpha=0.333, gamma=0.389, delta=0.278),
        "w/o Quality": replace(config, gamma=0.0, alpha=0.462, beta=0.154, delta=0.384),
        "w/o Temporal": replace(config, delta=0.0, alpha=0.400, beta=0.133, gamma=0.467),
    }

    pose_cut_flags = find_pose_cut_candidates(times, inter_cam_sims, config)
    results = {}

    for name, cfg in ablation_configs.items():
        rebuilt = rebuild_scores_from_components(score_components, cfg)
        chosen = select_camera_topk(times, rebuilt, pose_cut_flags, cfg, pose_history)
        metrics = evaluate(chosen, times, gt_cuts)
        metrics["top3_hit_rate"] = compute_topk_hit_rate(chosen, rebuilt, k=cfg.top_k)
        results[name] = metrics
        print(f"[Evaluation Ablation] {name}: {metrics}")

    return results


def iter_grid_configs(config: PipelineConfig) -> List[PipelineConfig]:
    configs = []

    for alpha in GRID_ALPHA_VALUES:
        for beta in GRID_BETA_VALUES:
            for gamma in GRID_GAMMA_VALUES:
                for delta in GRID_DELTA_VALUES:
                    if abs((alpha + beta + gamma + delta) - 1.0) <= 1e-6:
                        configs.append(
                            replace(
                                config,
                                alpha=alpha,
                                beta=beta,
                                gamma=gamma,
                                delta=delta,
                            )
                        )

    return configs


def grid_validation_score(metrics: Dict, total_sec: float) -> float:
    """GT 없을 때는 너무 많은 컷/너무 적은 컷을 피하는 휴리스틱 사용."""
    if metrics.get("MAE_frames") is not None:
        mae = metrics["MAE_frames"]
        acc_30 = metrics.get("acc_30f") or 0.0
        acc_15 = metrics.get("acc_15f") or 0.0
        return -mae + 10.0 * acc_30 + 5.0 * acc_15

    # K-pop 기준 185초에서 대략 35~45컷 정도가 발표용으로 무난
    target_cuts = max(total_sec / 4.8, 1.0)
    cut_penalty = abs(metrics["n_cuts"] - target_cuts) / target_cuts

    return metrics.get("top3_hit_rate", 0.0) - 0.08 * cut_penalty


def run_grid_search(
    times: List[float],
    score_components: List[Dict],
    inter_cam_sims: List[float],
    pose_history: List[List[Optional[np.ndarray]]],
    config: PipelineConfig,
    gt_cuts: Optional[List[Tuple[float, int]]] = None,
    report_dir: str = REPORT_DIR,
) -> Tuple[PipelineConfig, List[Dict]]:
    candidates = iter_grid_configs(config)
    pose_cut_flags = find_pose_cut_candidates(times, inter_cam_sims, config)
    total_sec = times[-1] - times[0] if times else 0.0
    rows = []

    print(f"[Evaluation] GridSearch: {len(candidates)} candidates")

    for idx, cfg in enumerate(candidates, start=1):
        rebuilt = rebuild_scores_from_components(score_components, cfg)
        chosen = select_camera_topk(times, rebuilt, pose_cut_flags, cfg, pose_history)
        metrics = evaluate(chosen, times, gt_cuts)
        metrics["top3_hit_rate"] = compute_topk_hit_rate(chosen, rebuilt, k=cfg.top_k)

        validation_score = grid_validation_score(metrics, total_sec)

        rows.append(
            {
                "rank": 0,
                "candidate": idx,
                "alpha": cfg.alpha,
                "beta": cfg.beta,
                "gamma": cfg.gamma,
                "delta": cfg.delta,
                "n_cuts": metrics["n_cuts"],
                "total_steps": metrics["total_steps"],
                "MAE_frames": metrics["MAE_frames"],
                "acc_15f": metrics["acc_15f"],
                "acc_30f": metrics["acc_30f"],
                "top3_hit_rate": round(metrics["top3_hit_rate"], 6),
                "validation_score": round(validation_score, 6),
            }
        )

    rows.sort(key=lambda r: r["validation_score"], reverse=True)

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    os.makedirs(report_dir, exist_ok=True)
    log_path = os.path.join(report_dir, "grid_search_log.csv")

    if rows:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        best = rows[0]
        best_config = replace(
            config,
            alpha=best["alpha"],
            beta=best["beta"],
            gamma=best["gamma"],
            delta=best["delta"],
        )

        print(
            "[Evaluation] GridSearch best: "
            f"α={best_config.alpha} β={best_config.beta} "
            f"γ={best_config.gamma} δ={best_config.delta} "
            f"score={best['validation_score']}"
        )
    else:
        best_config = config
        print("[Evaluation] GridSearch skipped: no valid weight combinations.")

    print(f"[Evaluation] GridSearch log saved: {log_path}")
    return best_config, rows
