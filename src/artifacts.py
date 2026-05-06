"""CSV, JSON, summary, and plot artifacts for pipeline checkpoints."""

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import PipelineConfig, VideoMeta
from src.visualization import plot_score_heatmap, plot_timeline


def _artifact_dir(report_dir: str, dirname: str) -> str:
    path = os.path.join(report_dir, dirname)
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, ensure_ascii=False, indent=2)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_csv(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_preprocess_artifacts(
    report_dir: str,
    config: PipelineConfig,
    metas: List[VideoMeta],
    total_sec: float,
    sync_map: Dict[str, float],
) -> None:
    artifact_dir = _artifact_dir(report_dir, "01_preprocess")

    meta_rows = [
        {
            "path": meta.path,
            "fps": round(meta.fps, 6),
            "frame_count": meta.frame_count,
            "duration_sec": round(meta.duration_sec, 6),
            "width": meta.width,
            "height": meta.height,
            "offset_sec": round(meta.offset_sec, 6),
            "usable_duration_sec": round(meta.usable_duration_sec, 6),
        }
        for meta in metas
    ]
    _write_csv(
        os.path.join(artifact_dir, "video_metadata.csv"),
        [
            "path",
            "fps",
            "frame_count",
            "duration_sec",
            "width",
            "height",
            "offset_sec",
            "usable_duration_sec",
        ],
        meta_rows,
    )

    _write_json(
        os.path.join(artifact_dir, "preprocess_summary.json"),
        {
            "video_count": len(metas),
            "analysis_step_sec": config.analysis_step_sec,
            "target_resolution": config.target_resolution,
            "common_timeline_sec": round(total_sec, 6),
            "offsets_sec": sync_map,
            "audio_waveform_sync": "excluded",
        },
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [os.path.splitext(os.path.basename(meta.path))[0] for meta in metas]
    x = np.arange(len(metas))
    ax.bar(x - 0.18, [meta.duration_sec for meta in metas], width=0.36, label="duration")
    ax.bar(x + 0.18, [meta.usable_duration_sec for meta in metas], width=0.36, label="usable")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title("Input Duration / Usable Timeline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "timeline_duration.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Artifacts] preprocess saved: {artifact_dir}")


def save_feature_artifacts(
    report_dir: str,
    times: List[float],
    score_components: List[Dict],
    inter_cam_sims: List[float],
    pose_history,
) -> None:
    artifact_dir = _artifact_dir(report_dir, "02_features")
    n_cams = len(score_components[0]["p"]) if score_components else 0

    feature_rows = []
    for i, t in enumerate(times):
        for cam in range(n_cams):
            pose_detected = (
                pose_history[i][cam] is not None
                if i < len(pose_history) and cam < len(pose_history[i])
                else False
            )
            feature_rows.append(
                {
                    "time_sec": round(t, 6),
                    "camera": cam,
                    "pose_score": round(float(score_components[i]["p"][cam]), 6),
                    "motion_score": round(float(score_components[i]["m"][cam]), 6),
                    "quality_score": round(float(score_components[i]["q"][cam]), 6),
                    "pose_detected": int(pose_detected),
                }
            )

    _write_csv(
        os.path.join(artifact_dir, "feature_components.csv"),
        ["time_sec", "camera", "pose_score", "motion_score", "quality_score", "pose_detected"],
        feature_rows,
    )

    sim_rows = [
        {"time_sec": round(t, 6), "inter_cam_pose_similarity": round(float(sim), 6)}
        for t, sim in zip(times, inter_cam_sims)
    ]
    _write_csv(
        os.path.join(artifact_dir, "inter_camera_pose_similarity.csv"),
        ["time_sec", "inter_cam_pose_similarity"],
        sim_rows,
    )

    if score_components:
        p_mean = [float(np.mean(row["p"])) for row in score_components]
        m_mean = [float(np.mean(row["m"])) for row in score_components]
        q_mean = [float(np.mean(row["q"])) for row in score_components]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times, p_mean, label="pose")
        ax.plot(times, m_mean, label="motion")
        ax.plot(times, q_mean, label="quality")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Component")
        ax.set_title("Mean Feature Components")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(artifact_dir, "feature_means_over_time.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        values = [
            [v for row in score_components for v in row["p"]],
            [v for row in score_components for v in row["m"]],
            [v for row in score_components for v in row["q"]],
        ]
        for ax, title, vals in zip(axes, ["Pose", "Motion", "Quality"], values):
            ax.hist(vals, bins=20)
            ax.set_title(title)
            ax.set_xlim(0, 1)
        plt.suptitle("Feature Distributions")
        plt.tight_layout()
        plt.savefig(os.path.join(artifact_dir, "feature_distributions.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[Artifacts] features saved: {artifact_dir}")


def save_scoring_artifacts(
    report_dir: str,
    config: PipelineConfig,
    times: List[float],
    scores: List[List[float]],
    score_components: List[Dict],
) -> None:
    artifact_dir = _artifact_dir(report_dir, "03_scoring")
    n_cams = len(scores[0]) if scores else 0

    rows = []
    for i, t in enumerate(times):
        for cam in range(n_cams):
            rows.append(
                {
                    "time_sec": round(t, 6),
                    "camera": cam,
                    "pose_score": round(float(score_components[i]["p"][cam]), 6),
                    "motion_score": round(float(score_components[i]["m"][cam]), 6),
                    "quality_score": round(float(score_components[i]["q"][cam]), 6),
                    "total_score": round(float(scores[i][cam]), 6),
                }
            )

    _write_csv(
        os.path.join(artifact_dir, "score_table.csv"),
        ["time_sec", "camera", "pose_score", "motion_score", "quality_score", "total_score"],
        rows,
    )
    _write_json(
        os.path.join(artifact_dir, "scoring_config.json"),
        {
            "alpha": config.alpha,
            "beta": config.beta,
            "gamma": config.gamma,
            "delta": config.delta,
            "formula": "alpha*pose - beta*motion + gamma*quality - delta*temporal",
        },
    )
    if scores:
        plot_score_heatmap(
            times,
            scores,
            n_cams,
            save_path=os.path.join(artifact_dir, "score_heatmap.png"),
        )

    print(f"[Artifacts] scoring saved: {artifact_dir}")


def save_selection_artifacts(
    report_dir: str,
    times: List[float],
    chosen: List[int],
    pose_cut_flags: List[bool],
    inter_cam_sims: List[float],
    segments: List[Tuple[int, float, float]],
    n_cams: int,
) -> None:
    artifact_dir = _artifact_dir(report_dir, "04_selection")

    selection_rows = [
        {
            "time_sec": round(t, 6),
            "chosen_camera": cam,
            "pose_cut_candidate": int(pose_cut_flags[i]) if i < len(pose_cut_flags) else 0,
            "inter_cam_pose_similarity": round(float(inter_cam_sims[i]), 6)
            if i < len(inter_cam_sims)
            else "",
        }
        for i, (t, cam) in enumerate(zip(times, chosen))
    ]
    _write_csv(
        os.path.join(artifact_dir, "camera_selection_timeline.csv"),
        ["time_sec", "chosen_camera", "pose_cut_candidate", "inter_cam_pose_similarity"],
        selection_rows,
    )

    segment_rows = [
        {
            "camera": cam,
            "start_sec": round(start_sec, 6),
            "end_sec": round(end_sec, 6),
            "duration_sec": round(end_sec - start_sec, 6),
        }
        for cam, start_sec, end_sec in segments
    ]
    _write_csv(
        os.path.join(artifact_dir, "segments.csv"),
        ["camera", "start_sec", "end_sec", "duration_sec"],
        segment_rows,
    )

    if times and chosen:
        plot_timeline(
            times,
            chosen,
            n_cams,
            title="Camera Selection Timeline",
            save_path=os.path.join(artifact_dir, "selection_timeline.png"),
        )

    _write_json(
        os.path.join(artifact_dir, "selection_summary.json"),
        {
            "steps": len(times),
            "segments": len(segments),
            "cuts": max(len(segments) - 1, 0),
            "pose_cut_candidates": int(sum(pose_cut_flags)),
        },
    )

    print(f"[Artifacts] selection saved: {artifact_dir}")


def save_evaluation_artifacts(
    report_dir: str,
    our_eval: Dict,
    baseline_results: Dict[str, Dict],
    ablation_results: Dict[str, Dict],
    grid_rows: List[Dict],
    gt_cuts: Optional[List[Tuple[float, int]]],
) -> None:
    artifact_dir = _artifact_dir(report_dir, "05_evaluation")

    _write_json(os.path.join(artifact_dir, "ours_evaluation.json"), our_eval)

    baseline_rows = [
        {"method": name, **result}
        for name, result in baseline_results.items()
    ]
    if baseline_rows:
        _write_csv(
            os.path.join(artifact_dir, "baseline_metrics.csv"),
            list(baseline_rows[0].keys()),
            baseline_rows,
        )

    ablation_rows = [
        {"variant": name, **result}
        for name, result in ablation_results.items()
    ]
    if ablation_rows:
        _write_csv(
            os.path.join(artifact_dir, "ablation_metrics.csv"),
            list(ablation_rows[0].keys()),
            ablation_rows,
        )

    if grid_rows:
        _write_csv(
            os.path.join(artifact_dir, "grid_search_top10.csv"),
            list(grid_rows[0].keys()),
            grid_rows[:10],
        )

    gt_rows = [
        {"time_sec": round(float(time_sec), 6), "camera": int(camera)}
        for time_sec, camera in (gt_cuts or [])
    ]
    _write_csv(
        os.path.join(artifact_dir, "gt_cuts_loaded.csv"),
        ["time_sec", "camera"],
        gt_rows,
    )

    _write_json(
        os.path.join(artifact_dir, "evaluation_summary.json"),
        {
            "gt_cut_count": len(gt_cuts or []),
            "gt_has_camera_labels": any(camera >= 0 for _, camera in (gt_cuts or [])),
            "ours": our_eval,
            "best_grid": grid_rows[0] if grid_rows else None,
        },
    )

    print(f"[Artifacts] evaluation saved: {artifact_dir}")
