"""Input validation, metadata, timeline, and synced frame reads."""

import os
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from moviepy import VideoFileClip

from src.config import (
    FPS_RESAMPLING_APPLY,
    MIN_SWITCH_POSE_SIM,
    POSE_SWITCH_BONUS,
    PREPROCESS_DIR,
    PipelineConfig,
    VideoMeta,
    load_pipeline_config,
    validate_config,
)


def read_video_meta(path: str, offset_sec: float) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid FPS: {path}")
    if frame_count <= 0:
        raise ValueError(f"Invalid frame count: {path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resolution: {path}")

    duration_sec = frame_count / fps
    usable_duration_sec = duration_sec - offset_sec

    if usable_duration_sec <= 0:
        raise ValueError(
            f"Usable duration <= 0. path={path}, "
            f"duration={duration_sec:.2f}, offset={offset_sec:.2f}"
        )

    return VideoMeta(
        path=path,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        width=width,
        height=height,
        offset_sec=offset_sec,
        usable_duration_sec=usable_duration_sec,
    )


def collect_video_meta(config: PipelineConfig) -> List[VideoMeta]:
    metas = []
    for path in config.video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        metas.append(read_video_meta(path, config.offsets_sec[path]))

    fps_list = [m.fps for m in metas]
    ref_fps = fps_list[0]
    max_fps_diff = max(abs(f - ref_fps) for f in fps_list)

    for m in metas:
        if abs(m.fps - ref_fps) > 0.5:
            print(
                f"[WARNING] FPS mismatch: {m.path} fps={m.fps:.3f} "
                f"vs ref={ref_fps:.3f}. Consider re-encoding."
            )

    print(f"[Preprocess] Reference FPS: {ref_fps:.3f} | Max FPS diff: {max_fps_diff:.3f}")
    return metas


def maybe_resample_fps(
    config: PipelineConfig,
    metas: List[VideoMeta],
    tolerance: float = 0.5,
) -> Tuple[PipelineConfig, List[VideoMeta]]:
    ref_fps = metas[0].fps
    needs_resample = any(abs(m.fps - ref_fps) > tolerance for m in metas)

    if not needs_resample:
        print("[Preprocess] FPS resampling skipped: all inputs are within tolerance.")
        return config, metas

    if not FPS_RESAMPLING_APPLY:
        print(
            "[Preprocess] FPS mismatch detected, but automatic resampling is disabled "
            "to preserve original source timing."
        )
        return config, metas

    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    new_paths = []
    new_offsets = {}

    print(f"[Preprocess] FPS resampling started: target_fps={ref_fps:.3f}")

    for meta in metas:
        base, _ = os.path.splitext(os.path.basename(meta.path))
        out_path = os.path.join(PREPROCESS_DIR, f"{base}_fps{ref_fps:.3f}.mp4")

        if abs(meta.fps - ref_fps) > tolerance:
            if not os.path.exists(out_path):
                clip = VideoFileClip(meta.path)
                try:
                    clip.write_videofile(
                        out_path,
                        fps=ref_fps,
                        codec="libx264",
                        audio_codec="aac",
                    )
                finally:
                    clip.close()
            print(f"[Preprocess] FPS resampled: {meta.path} -> {out_path}")
            new_path = out_path
        else:
            new_path = meta.path

        new_paths.append(new_path)
        new_offsets[new_path] = config.offsets_sec[meta.path]

    new_config = replace(config, video_paths=new_paths, offsets_sec=new_offsets)
    new_metas = [read_video_meta(path, new_config.offsets_sec[path]) for path in new_paths]
    return new_config, new_metas


def common_timeline_duration(metas: List[VideoMeta]) -> float:
    total = min(m.usable_duration_sec for m in metas)
    if total <= 0:
        raise ValueError("Common timeline duration is <= 0.")
    return total


def build_timeline(total_sec: float, step_sec: float) -> List[float]:
    times = []
    t = 0.0
    while t < total_sec:
        times.append(t)
        t += step_sec
    if not times:
        raise ValueError("Generated timeline is empty.")
    return times


def normalize_frame_resolution(
    frame: np.ndarray,
    target_resolution: Optional[Tuple[int, int]],
) -> np.ndarray:
    if target_resolution is None:
        return frame
    tw, th = target_resolution
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def read_synced_frame(
    cap: cv2.VideoCapture,
    meta: VideoMeta,
    timeline_t_sec: float,
    target_resolution: Optional[Tuple[int, int]],
) -> Optional[np.ndarray]:
    src_t = timeline_t_sec + meta.offset_sec
    frame_idx = int(round(src_t * meta.fps))
    frame_idx = max(0, min(frame_idx, meta.frame_count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ok, frame = cap.read()
    if not ok:
        return None
    return normalize_frame_resolution(frame, target_resolution)


def build_audio_sync_map(metas: List[VideoMeta], ref_idx: int = 0) -> Dict[str, float]:
    ref_offset = metas[ref_idx].offset_sec
    sync_map = {}

    for m in metas:
        sync_map[m.path] = m.offset_sec
        drift = m.offset_sec - ref_offset
        print(
            f"[Preprocess] AudioSync: {m.path} "
            f"offset={m.offset_sec:.3f}s drift_from_ref={drift:+.3f}s"
        )
    return sync_map


def print_preprocess_summary(config: PipelineConfig, metas: List[VideoMeta], total_sec: float) -> None:
    print("=== Preprocess Summary ===")
    print(f"videos:              {len(config.video_paths)}")
    print(f"analysis_step_sec:   {config.analysis_step_sec}")
    print(f"target_resolution:   {config.target_resolution}")
    print(f"common_timeline_sec: {round(total_sec, 2)}")
    print(f"weights: α={config.alpha} β={config.beta} γ={config.gamma} δ={config.delta}")
    print(f"top_k:               {config.top_k}")
    print(f"min_scene_len_sec:   {config.min_scene_len_sec}")
    print(f"switch_threshold:    {config.switch_threshold}")
    print(f"pose_cut_threshold:  {config.pose_cut_sim_threshold}")
    print(f"pose_switch_bonus:   {POSE_SWITCH_BONUS}")
    print(f"min_switch_pose_sim: {MIN_SWITCH_POSE_SIM}")

    for meta in metas:
        print(
            f"  {meta.path}: fps={meta.fps:.3f}, res={meta.width}x{meta.height}, "
            f"offset={meta.offset_sec:.2f}s, usable={meta.usable_duration_sec:.2f}s"
        )


def load_validated_context():
    config = load_pipeline_config()
    validate_config(config)

    metas = collect_video_meta(config)
    config, metas = maybe_resample_fps(config, metas)
    print("[Preprocess] AudioSync: using fixed manual offsets from OFFSETS_SEC.")

    validate_config(config)
    metas = [read_video_meta(path, config.offsets_sec[path]) for path in config.video_paths]
    total_sec = common_timeline_duration(metas)

    return config, metas, total_sec


def build_debug_times(total_sec: float, sample_count: int = 12) -> List[float]:
    if sample_count <= 0:
        raise ValueError("sample_count must be > 0.")

    end = max(0.0, total_sec - 1.0)

    if sample_count == 1:
        return [0.0]

    return [float(t) for t in np.linspace(0.0, end, sample_count)]
