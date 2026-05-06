"""Configuration, dataclasses, and small shared utilities."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


VIDEO_PATHS = [
    "stage1.mp4",
    "stage2.mp4",
    "stage3.mp4",
    "stage4.mp4",
]

OUTPUT_PATH = "edited_output.mp4"
POSE_MODEL_PATH = "pose_landmarker_lite.task"

OFFSETS_SEC = {
    "stage1.mp4": 3.5,
    "stage2.mp4": 29.5,
    "stage3.mp4": 0.1,
    "stage4.mp4": 0.0,
}

ANALYSIS_STEP_SEC = 0.5
MIN_SCENE_LEN_SEC = 1.7
SWITCH_THRESHOLD = 0.10

TARGET_RESOLUTION: Optional[Tuple[int, int]] = (1280, 720)

ALPHA = 0.30
BETA = 0.10
GAMMA = 0.35
DELTA = 0.25

TOP_K = 3

POSE_CUT_SIM_THRESHOLD = 0.72
POSE_CUT_FALLBACK_PERCENTILE = 40.0
MIN_SWITCH_POSE_SIM = 0.42
POSE_SWITCH_BONUS = 0.25

POSE_MOTION_SCALE = 1.25
OPTICAL_FLOW_FALLBACK_SCALE = 30.0
MOTION_SMOOTH_WINDOW = 5

QUALITY_SHARPNESS_SCALE = 500.0

COMPONENT_NORMALIZE = True
SELECTION_DEBUG_DURATION_SEC = 4.0

REPORT_DIR = "report"
PREPROCESS_DIR = os.path.join(REPORT_DIR, "preprocessed")

FPS_RESAMPLING_APPLY = False

GT_CUTS_PATH = "gt_cuts.csv"

GRID_ALPHA_VALUES = [0.25, 0.30, 0.35, 0.40]
GRID_BETA_VALUES = [0.05, 0.10, 0.15, 0.20]
GRID_GAMMA_VALUES = [0.25, 0.30, 0.35, 0.40]
GRID_DELTA_VALUES = [0.15, 0.20, 0.25, 0.30]



@dataclass(frozen=True)
class PipelineConfig:
    video_paths: List[str]
    offsets_sec: Dict[str, float]
    analysis_step_sec: float
    target_resolution: Optional[Tuple[int, int]]
    min_scene_len_sec: float
    switch_threshold: float
    output_path: str
    alpha: float = ALPHA
    beta: float = BETA
    gamma: float = GAMMA
    delta: float = DELTA
    top_k: int = TOP_K
    pose_cut_sim_threshold: float = POSE_CUT_SIM_THRESHOLD
    pose_cut_fallback_percentile: float = POSE_CUT_FALLBACK_PERCENTILE


@dataclass(frozen=True)
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int
    offset_sec: float
    usable_duration_sec: float


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def format_elapsed(seconds: float) -> str:
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes}m {remain:.1f}s"


def ensure_report_dir(report_dir: str = REPORT_DIR) -> None:
    os.makedirs(report_dir, exist_ok=True)


def load_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        video_paths=VIDEO_PATHS,
        offsets_sec=OFFSETS_SEC,
        analysis_step_sec=ANALYSIS_STEP_SEC,
        target_resolution=TARGET_RESOLUTION,
        min_scene_len_sec=MIN_SCENE_LEN_SEC,
        switch_threshold=SWITCH_THRESHOLD,
        output_path=OUTPUT_PATH,
    )


def validate_config(config: PipelineConfig) -> None:
    if not config.video_paths:
        raise ValueError("video_paths is empty.")
    if len(config.video_paths) < 4 or len(config.video_paths) > 6:
        raise ValueError("Multicam input must contain 4~6 videos.")
    if config.analysis_step_sec <= 0:
        raise ValueError("analysis_step_sec must be > 0.")
    if config.min_scene_len_sec <= 0:
        raise ValueError("min_scene_len_sec must be > 0.")
    if not (0 <= config.switch_threshold <= 1):
        raise ValueError("switch_threshold must be in [0, 1].")

    for path in config.video_paths:
        if path not in config.offsets_sec:
            raise KeyError(f"Missing offset for video: {path}")
        if config.offsets_sec[path] < 0:
            raise ValueError(f"Offset must be >= 0: {path}")

    if config.target_resolution is not None:
        w, h = config.target_resolution
        if w <= 0 or h <= 0:
            raise ValueError("target_resolution values must be > 0.")

    weight_sum = config.alpha + config.beta + config.gamma + config.delta
    if not (0.99 < weight_sum < 1.01):
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.3f}")
    if not (0 <= config.pose_cut_sim_threshold <= 1):
        raise ValueError("pose_cut_sim_threshold must be in [0, 1].")
    if not (0 <= config.pose_cut_fallback_percentile <= 100):
        raise ValueError("pose_cut_fallback_percentile must be in [0, 100].")
