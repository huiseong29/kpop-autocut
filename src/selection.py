"""Pose-aware Top-K camera selection and segment creation."""

from typing import List, Optional

import numpy as np

from src.config import MIN_SWITCH_POSE_SIM, POSE_SWITCH_BONUS, PipelineConfig
from src.features import pose_cosine_similarity
from src.scoring import compute_temporal_penalty


def compute_inter_cam_pose_similarity(
    pose_landmarks_per_cam: List[Optional[np.ndarray]],
) -> float:
    valid = [lm for lm in pose_landmarks_per_cam if lm is not None]

    if len(valid) < 2:
        return 0.0

    sims = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            sims.append(pose_cosine_similarity(valid[i], valid[j]))

    return float(np.mean(sims)) if sims else 0.0


def find_pose_cut_candidates(
    times: List[float],
    inter_cam_sims: List[float],
    config: PipelineConfig,
) -> List[bool]:
    arr = np.array(inter_cam_sims, dtype=np.float32)

    if len(arr) == 0:
        return []

    flags = (arr >= config.pose_cut_sim_threshold).tolist()
    n_cands = sum(flags)
    fallback = False

    # 후보가 너무 적으면 상위 percentile 기반으로 완화
    if n_cands / max(len(flags), 1) * 100 < config.pose_cut_fallback_percentile:
        cutoff = float(np.percentile(arr, 100.0 - config.pose_cut_fallback_percentile))
        flags = (arr >= cutoff).tolist()
        n_cands = sum(flags)
        fallback = True

    print(
        f"[Selection] PoseCutCandidates: {n_cands}/{len(flags)} "
        f"({100*n_cands/max(len(flags),1):.1f}%) | "
        f"sim min={arr.min():.3f} mean={arr.mean():.3f} max={arr.max():.3f} | "
        f"threshold={config.pose_cut_sim_threshold} | "
        f"fallback={'YES' if fallback else 'NO'}"
    )

    return flags


def switch_pose_similarity_at_time(
    pose_history: Optional[List[List[Optional[np.ndarray]]]],
    time_idx: int,
    from_cam: int,
    to_cam: int,
) -> float:
    if pose_history is None:
        return 0.5
    if time_idx < 0 or time_idx >= len(pose_history):
        return 0.5

    poses = pose_history[time_idx]

    if from_cam >= len(poses) or to_cam >= len(poses):
        return 0.5

    return pose_cosine_similarity(poses[from_cam], poses[to_cam])


def select_camera_topk(
    times: List[float],
    scores: List[List[float]],
    pose_cut_flags: List[bool],
    config: PipelineConfig,
    pose_history: Optional[List[List[Optional[np.ndarray]]]] = None,
) -> List[int]:
    if not scores:
        raise ValueError("No score data.")

    chosen: List[int] = []
    cur = int(np.argmax(scores[0]))
    last_switch_t = times[0]
    chosen.append(cur)

    total_attempted = 0
    pose_gated = 0

    for i in range(1, len(times)):
        row = scores[i]
        top_k = sorted(range(len(row)), key=lambda x: row[x], reverse=True)[: config.top_k]

        # 현재 카메라 유지 점수
        elapsed = times[i] - last_switch_t
        temporal_penalty = config.delta * compute_temporal_penalty(
            elapsed,
            config.min_scene_len_sec,
        )

        current_score = row[cur] + POSE_SWITCH_BONUS * 1.0

        # 후보 카메라 점수 = 기본 score + 현재 카메라와의 pose similarity bonus
        adjusted_candidates = []
        for cand in top_k:
            sim = switch_pose_similarity_at_time(pose_history, i, cur, cand)
            adjusted = row[cand] + POSE_SWITCH_BONUS * sim
            adjusted_candidates.append((adjusted, cand, sim))

        adjusted_best, best_cam, switch_sim = max(adjusted_candidates, key=lambda x: x[0])

        wants_switch = (
            best_cam != cur
            and elapsed >= config.min_scene_len_sec
            and (adjusted_best - temporal_penalty - current_score) > config.switch_threshold
        )

        if wants_switch:
            total_attempted += 1

            # similarity가 낮으면 의상/시점이 달라도 포즈 연결이 어색할 수 있으므로 컷 억제
            if pose_cut_flags[i] and switch_sim >= MIN_SWITCH_POSE_SIM:
                cur = best_cam
                last_switch_t = times[i]
            else:
                pose_gated += 1

        chosen.append(cur)

    scene_lens = []
    s, c = times[0], chosen[0]

    for i in range(1, len(chosen)):
        if chosen[i] != c:
            scene_lens.append(times[i] - s)
            s, c = times[i], chosen[i]

    scene_lens.append(times[-1] - s)

    print(
        f"[Selection] CutSelection: attempted={total_attempted} "
        f"actual={total_attempted - pose_gated} pose_gated={pose_gated} "
        f"avg_scene={np.mean(scene_lens):.2f}s"
    )

    return chosen


def make_segments(times: List[float], chosen: List[int], total_sec: float):
    if not times or not chosen:
        raise ValueError("times or chosen is empty.")

    segments = []
    start = times[0]
    cam = chosen[0]

    for i in range(1, len(times)):
        if chosen[i] != cam:
            segments.append((cam, start, times[i]))
            start = times[i]
            cam = chosen[i]

    segments.append((cam, start, total_sec))
    return segments
