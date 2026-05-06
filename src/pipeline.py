"""CLI orchestration for the modular K-Pop multicam pipeline."""

import os
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.artifacts import (
    save_evaluation_artifacts,
    save_feature_artifacts,
    save_preprocess_artifacts,
    save_scoring_artifacts,
    save_selection_artifacts,
)
from src.config import (
    COMPONENT_NORMALIZE,
    GT_CUTS_PATH,
    REPORT_DIR,
    SELECTION_DEBUG_DURATION_SEC,
    PipelineConfig,
    VideoMeta,
    ensure_report_dir,
    format_elapsed,
    load_pipeline_config,
    validate_config,
)
from src.evaluation import (
    compute_topk_hit_rate,
    evaluate,
    load_gt_cuts,
    run_ablation,
    run_baseline_pose_only,
    run_baseline_quality_only,
    run_baseline_random,
    run_baseline_uniform,
    run_grid_search,
)
from src.features import (
    combine_motion_score,
    extract_pose_landmarks,
    extract_pose_score,
    extract_quality_score,
    init_pose_detector,
    normalize_pose_keypoints,
    optical_flow_motion_score,
)
from src.preprocess import (
    build_audio_sync_map,
    build_debug_times,
    build_timeline,
    collect_video_meta,
    common_timeline_duration,
    load_validated_context,
    maybe_resample_fps,
    print_preprocess_summary,
    read_synced_frame,
)
from src.rendering import build_video
from src.scoring import calculate_score, normalize_score_components, rebuild_scores_from_components
from src.selection import (
    compute_inter_cam_pose_similarity,
    find_pose_cut_candidates,
    make_segments,
    select_camera_topk,
)
from src.visualization import (
    export_report_pdf,
    plot_ablation,
    plot_baseline_comparison,
    plot_score_heatmap,
    plot_timeline,
    write_pipeline_summary,
)


def analyze(config: PipelineConfig, metas: List[VideoMeta]):
    total_sec = common_timeline_duration(metas)
    timeline = build_timeline(total_sec, config.analysis_step_sec)
    caps = [cv2.VideoCapture(meta.path) for meta in metas]

    times: List[float] = []
    scores: List[List[float]] = []
    inter_cam_sims: List[float] = []
    score_components: List[Dict] = []
    pose_history: List[List[Optional[np.ndarray]]] = []

    pose_det = init_pose_detector()

    prev_grays = [None] * len(metas)
    prev_poses = [None] * len(metas)

    print("pose detector ready:", pose_det is not None)

    try:
        for t in timeline:
            print(f"analyzing time: {round(t, 2)}", flush=True)

            row = []
            curr_poses_t = []
            comp_row = {"p": [], "m": [], "q": []}

            for idx, (cap, meta) in enumerate(zip(caps, metas)):
                frame = read_synced_frame(cap, meta, t, config.target_resolution)

                if frame is None:
                    row.append(0.0)
                    curr_poses_t.append(None)
                    comp_row["p"].append(0.0)
                    comp_row["m"].append(0.0)
                    comp_row["q"].append(0.0)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                q_score = extract_quality_score(gray)

                p_score, curr_pose, _temporal_sim, pose_motion = extract_pose_score(
                    frame,
                    pose_det,
                    prev_poses[idx],
                )

                flow_motion = optical_flow_motion_score(prev_grays[idx], gray)
                m_score = combine_motion_score(pose_motion, flow_motion, idx)

                score = calculate_score(p_score, m_score, q_score, 0.0, config)

                prev_grays[idx] = gray
                prev_poses[idx] = curr_pose

                row.append(score)
                curr_poses_t.append(curr_pose)
                comp_row["p"].append(p_score)
                comp_row["m"].append(m_score)
                comp_row["q"].append(q_score)

            times.append(t)
            scores.append(row)
            inter_cam_sims.append(compute_inter_cam_pose_similarity(curr_poses_t))
            score_components.append(comp_row)
            pose_history.append(curr_poses_t)

    finally:
        if pose_det is not None and hasattr(pose_det, "close"):
            pose_det.close()
        for cap in caps:
            cap.release()

    if COMPONENT_NORMALIZE:
        score_components = normalize_score_components(score_components)
        scores = rebuild_scores_from_components(score_components, config)
        print("[Scoring] Component normalization applied: p/m/q camera-wise robust min-max.")

    return times, scores, inter_cam_sims, score_components, pose_history, total_sec


def run_pose_debug() -> None:
    run_start = time.perf_counter()
    print("pose debug started")

    config, metas, total_sec = load_validated_context()
    print_preprocess_summary(config, metas, total_sec)

    sample_times = build_debug_times(total_sec)
    caps = [cv2.VideoCapture(meta.path) for meta in metas]
    pose_det = init_pose_detector()

    checked = 0
    detected = 0

    try:
        for t in sample_times:
            print(f"time={t:.2f}s")
            for cam_idx, (cap, meta) in enumerate(zip(caps, metas)):
                checked += 1
                frame = read_synced_frame(cap, meta, t, config.target_resolution)
                kp = extract_pose_landmarks(frame, pose_det) if frame is not None else None

                if kp is None:
                    print(f"  cam={cam_idx} detected=False")
                    continue

                detected += 1
                norm = normalize_pose_keypoints(kp)

                print(
                    f"  cam={cam_idx} detected=True shape={kp.shape} "
                    f"mean_visibility={float(np.mean(kp[:, 3])):.3f} "
                    f"normalized={norm is not None}"
                )

    finally:
        if pose_det is not None and hasattr(pose_det, "close"):
            pose_det.close()
        for cap in caps:
            cap.release()

    rate = detected / checked if checked else 0.0
    print(f"pose detection: {detected}/{checked} ({rate:.1%})")
    print("pose debug elapsed:", format_elapsed(time.perf_counter() - run_start))


def run_similarity_debug() -> None:
    run_start = time.perf_counter()
    print("similarity debug started")

    config, metas, total_sec = load_validated_context()
    print_preprocess_summary(config, metas, total_sec)

    sample_times = build_debug_times(total_sec)
    caps = [cv2.VideoCapture(meta.path) for meta in metas]
    pose_det = init_pose_detector()

    sims = []

    try:
        for t in sample_times:
            poses = []
            for cap, meta in zip(caps, metas):
                frame = read_synced_frame(cap, meta, t, config.target_resolution)
                poses.append(extract_pose_landmarks(frame, pose_det) if frame is not None else None)

            sim = compute_inter_cam_pose_similarity(poses)
            sims.append(sim)
            print(f"time={t:.2f}s inter_cam_pose_similarity={sim:.3f}")

    finally:
        if pose_det is not None and hasattr(pose_det, "close"):
            pose_det.close()
        for cap in caps:
            cap.release()

    if sims:
        print(f"sim min={min(sims):.3f} mean={float(np.mean(sims)):.3f} max={max(sims):.3f}")

    print("similarity debug elapsed:", format_elapsed(time.perf_counter() - run_start))


def run_selection_debug() -> None:
    run_start = time.perf_counter()
    print("selection debug started")

    config, metas, total_sec = load_validated_context()
    print_preprocess_summary(config, metas, total_sec)
    sync_map = build_audio_sync_map(metas)
    save_preprocess_artifacts(REPORT_DIR, config, metas, total_sec, sync_map)

    debug_total_sec = min(total_sec, SELECTION_DEBUG_DURATION_SEC)
    debug_metas = [
        replace(meta, usable_duration_sec=min(meta.usable_duration_sec, debug_total_sec))
        for meta in metas
    ]
    print(f"selection debug duration: {debug_total_sec:.2f}s")

    times, scores, inter_cam_sims, score_components, pose_history, total_sec = analyze(
        config,
        debug_metas,
    )
    pose_cut_flags = find_pose_cut_candidates(times, inter_cam_sims, config)
    chosen = select_camera_topk(times, scores, pose_cut_flags, config, pose_history)
    segments = make_segments(times, chosen, total_sec)
    save_feature_artifacts(
        REPORT_DIR,
        times,
        score_components,
        inter_cam_sims,
        pose_history,
    )
    save_scoring_artifacts(REPORT_DIR, config, times, scores, score_components)
    save_selection_artifacts(
        REPORT_DIR,
        times,
        chosen,
        pose_cut_flags,
        inter_cam_sims,
        segments,
        len(config.video_paths),
    )

    print("segments:")
    for cam, start_sec, end_sec in segments:
        print(f"  cam={cam} start={start_sec:.2f}s end={end_sec:.2f}s")

    print("selection debug elapsed:", format_elapsed(time.perf_counter() - run_start))


def main() -> None:
    run_start = time.perf_counter()
    print("main started")

    ensure_report_dir(REPORT_DIR)

    config, metas, total_sec = load_validated_context()
    print_preprocess_summary(config, metas, total_sec)
    sync_map = build_audio_sync_map(metas)
    save_preprocess_artifacts(REPORT_DIR, config, metas, total_sec, sync_map)

    gt_cuts = load_gt_cuts(GT_CUTS_PATH)

    times, scores, inter_cam_sims, score_components, pose_history, total_sec = analyze(config, metas)
    print("analysis finished")
    save_feature_artifacts(
        REPORT_DIR,
        times,
        score_components,
        inter_cam_sims,
        pose_history,
    )
    save_scoring_artifacts(REPORT_DIR, config, times, scores, score_components)

    pose_cut_flags = find_pose_cut_candidates(times, inter_cam_sims, config)

    chosen = select_camera_topk(times, scores, pose_cut_flags, config, pose_history)
    print("camera selection finished")

    segments = make_segments(times, chosen, total_sec)
    print(f"segments: {len(segments)}")
    save_selection_artifacts(
        REPORT_DIR,
        times,
        chosen,
        pose_cut_flags,
        inter_cam_sims,
        segments,
        len(config.video_paths),
    )

    our_eval = evaluate(chosen, times, gt_cuts)
    our_eval["top3_hit_rate"] = compute_topk_hit_rate(chosen, scores, k=config.top_k)
    print("ours eval:", our_eval)

    ablation_results = run_ablation(
        times,
        score_components,
        inter_cam_sims,
        pose_history,
        config,
        gt_cuts=gt_cuts,
    )

    best_grid_config, grid_rows = run_grid_search(
        times,
        score_components,
        inter_cam_sims,
        pose_history,
        config,
        gt_cuts=gt_cuts,
    )

    n_cams = len(config.video_paths)

    baseline_results = {
        "Random": evaluate(run_baseline_random(times, n_cams, config), times, gt_cuts),
        "Uniform": evaluate(run_baseline_uniform(times, n_cams, config), times, gt_cuts),
        "Pose Only": evaluate(
            run_baseline_pose_only(times, score_components, pose_cut_flags, config),
            times,
            gt_cuts,
        ),
        "Quality Only": evaluate(
            run_baseline_quality_only(times, score_components, pose_cut_flags, config),
            times,
            gt_cuts,
        ),
        "Ours": our_eval,
    }
    save_evaluation_artifacts(
        REPORT_DIR,
        our_eval,
        baseline_results,
        ablation_results,
        grid_rows,
        gt_cuts,
    )

    plot_timeline(times, chosen, n_cams, save_path=os.path.join(REPORT_DIR, "timeline.png"))
    plot_score_heatmap(times, scores, n_cams, save_path=os.path.join(REPORT_DIR, "score_heatmap.png"))
    plot_baseline_comparison(
        baseline_results,
        save_path=os.path.join(REPORT_DIR, "baseline_comparison.png"),
    )
    plot_ablation(ablation_results, save_path=os.path.join(REPORT_DIR, "ablation.png"))
    export_report_pdf(REPORT_DIR, baseline_results, ablation_results, grid_rows)

    print("video rendering started")
    build_video(segments, config, total_sec)

    write_pipeline_summary(REPORT_DIR, config, metas, total_sec, segments, grid_rows)

    print("total elapsed:", format_elapsed(time.perf_counter() - run_start))
    print("done:", config.output_path)


def print_usage() -> None:
    print("usage: python main_all.py [pose-debug|similarity-debug|selection-debug|full]")
    print("  pose-debug: Lite Pose Landmarker keypoint extraction debug only")
    print("  similarity-debug: Hip-normalized pose similarity debug only")
    print("  selection-debug: analyze + camera selection debug without rendering")
    print("  full: full analyze + report + render pipeline")


def run_cli() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "full"

    commands = {
        "pose-debug": run_pose_debug,
        "similarity-debug": run_similarity_debug,
        "selection-debug": run_selection_debug,
        "full": main,
    }

    if command not in commands:
        print_usage()
        raise SystemExit(f"unknown command: {command}")

    commands[command]()
