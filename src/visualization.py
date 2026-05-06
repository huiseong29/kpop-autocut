"""Plots, report PDF, and pipeline summary outputs."""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from src.config import MIN_SWITCH_POSE_SIM, POSE_SWITCH_BONUS, PipelineConfig, VideoMeta


def plot_timeline(
    times: List[float],
    chosen: List[int],
    n_cams: int,
    title: str = "Camera Selection Timeline",
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 3))
    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = [cmap(c / max(n_cams - 1, 1)) for c in chosen]

    ax.scatter(times, chosen, c=colors, s=8, zorder=3)

    for cut_time in [times[i] for i in range(1, len(chosen)) if chosen[i] != chosen[i - 1]]:
        ax.axvline(cut_time, color="red", alpha=0.4, linewidth=0.8, linestyle="--")

    ax.set_yticks(range(n_cams))
    ax.set_yticklabels([f"Cam {i+1}" for i in range(n_cams)])
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_xlim(times[0], times[-1])

    patches = [
        mpatches.Patch(color=cmap(i / max(n_cams - 1, 1)), label=f"Cam {i+1}")
        for i in range(n_cams)
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()
    print(f"[Visualization] Timeline saved: {save_path}")


def plot_score_heatmap(
    times: List[float],
    scores: List[List[float]],
    n_cams: int,
    save_path: Optional[str] = None,
) -> None:
    arr = np.array(scores).T
    fig, ax = plt.subplots(figsize=(14, 3))

    im = ax.imshow(
        arr,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], -0.5, n_cams - 0.5],
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )

    ax.set_yticks(range(n_cams))
    ax.set_yticklabels([f"Cam {i+1}" for i in range(n_cams)])
    ax.set_xlabel("Time (s)")
    ax.set_title("Score Heatmap per Camera")

    plt.colorbar(im, ax=ax, label="Score")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()
    print(f"[Visualization] Heatmap saved: {save_path}")


def plot_baseline_comparison(
    baseline_results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> None:
    names = list(baseline_results.keys())
    n_cuts = [baseline_results[n].get("n_cuts", 0) for n in names]
    mae = [baseline_results[n].get("MAE_frames") for n in names]
    acc_30 = [baseline_results[n].get("acc_30f") for n in names]

    has_gt = any(v is not None for v in mae)
    ncols = 3 if has_gt else 1

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    if ncols == 1:
        axes = [axes]

    x = np.arange(len(names))
    axes[0].bar(x, n_cuts)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    axes[0].set_title("n_cuts Comparison")
    axes[0].set_ylabel("# Cuts")

    if has_gt:
        axes[1].bar(x, [v or 0 for v in mae])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        axes[1].set_title("MAE (frames)")

        axes[2].bar(x, [v or 0 for v in acc_30])
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        axes[2].set_title("acc_30f")
        axes[2].set_ylim(0, 1)

    plt.suptitle("Baseline vs Ours Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()
    print(f"[Visualization] Baseline comparison saved: {save_path}")


def plot_ablation(ablation_results: Dict[str, Dict], save_path: Optional[str] = None) -> None:
    names = list(ablation_results.keys())
    hit_rates = [ablation_results[n].get("top3_hit_rate", 0) for n in names]
    n_cuts = [ablation_results[n].get("n_cuts", 0) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(names))

    axes[0].bar(x, hit_rates)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    axes[0].set_title("Top-3 Hit Rate (Ablation)")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Hit Rate")

    axes[1].bar(x, n_cuts)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    axes[1].set_title("n_cuts (Ablation)")
    axes[1].set_ylabel("# Cuts")

    plt.suptitle("Ablation Study", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()
    print(f"[Visualization] Ablation graph saved: {save_path}")


def export_report_pdf(
    report_dir: str,
    baseline_results: Dict[str, Dict],
    ablation_results: Dict[str, Dict],
    grid_rows: List[Dict],
    save_path: Optional[str] = None,
) -> None:
    if save_path is None:
        save_path = os.path.join(report_dir, "summary_report.pdf")

    image_paths = [
        os.path.join(report_dir, "timeline.png"),
        os.path.join(report_dir, "score_heatmap.png"),
        os.path.join(report_dir, "baseline_comparison.png"),
        os.path.join(report_dir, "ablation.png"),
    ]

    with PdfPages(save_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        lines = [
            "K-Pop Multicam Editing Pipeline Report",
            "",
            "Baseline Results",
        ]

        for name, result in baseline_results.items():
            lines.append(f"- {name}: {result}")

        lines.extend(["", "Ablation Results"])

        for name, result in ablation_results.items():
            lines.append(f"- {name}: {result}")

        if grid_rows:
            best = grid_rows[0]
            lines.extend(
                [
                    "",
                    "Grid Search Best",
                    (
                        f"- rank=1 alpha={best['alpha']} beta={best['beta']} "
                        f"gamma={best['gamma']} delta={best['delta']} "
                        f"validation_score={best['validation_score']}"
                    ),
                ]
            )

        ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", fontsize=9, wrap=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for image_path in image_paths:
            if not os.path.exists(image_path):
                continue

            img = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(os.path.basename(image_path), fontsize=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[Visualization] PDF report saved: {save_path}")


def write_pipeline_summary(
    report_dir: str,
    config: PipelineConfig,
    metas: List[VideoMeta],
    total_sec: float,
    segments,
    grid_rows: List[Dict],
) -> None:
    os.makedirs(report_dir, exist_ok=True)
    save_path = os.path.join(report_dir, "pipeline_summary.txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Full pipeline executed\n")
        f.write(f"output_path: {config.output_path}\n")
        f.write(f"videos: {len(config.video_paths)}\n")
        f.write(f"common_timeline_sec: {total_sec:.3f}\n")
        f.write(f"segments: {len(segments)}\n")
        f.write(
            f"weights: alpha={config.alpha} beta={config.beta} "
            f"gamma={config.gamma} delta={config.delta}\n"
        )
        f.write(f"min_scene_len_sec: {config.min_scene_len_sec}\n")
        f.write(f"switch_threshold: {config.switch_threshold}\n")
        f.write(f"pose_switch_bonus: {POSE_SWITCH_BONUS}\n")
        f.write(f"min_switch_pose_sim: {MIN_SWITCH_POSE_SIM}\n")

        if grid_rows:
            best = grid_rows[0]
            f.write(
                "grid_best: "
                f"alpha={best['alpha']} beta={best['beta']} "
                f"gamma={best['gamma']} delta={best['delta']} "
                f"validation_score={best['validation_score']}\n"
            )

        for meta in metas:
            f.write(
                f"{meta.path}: fps={meta.fps:.3f}, "
                f"offset={meta.offset_sec:.3f}, usable={meta.usable_duration_sec:.3f}\n"
            )

    print(f"[Visualization] Pipeline summary saved: {save_path}")
