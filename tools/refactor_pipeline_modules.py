"""Split src/pipeline.py into phase modules.

This is a mechanical project-local refactor helper. It extracts the already
working functions from the current monolithic pipeline and rewrites the module
files with explicit imports. The script is intentionally narrow and should be
removed after the split is verified.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "src" / "pipeline.py"


source = PIPELINE.read_text(encoding="utf-8")
tree = ast.parse(source)


def get_node_source(name: str) -> str:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise KeyError(name)


CONFIG_CONSTANTS = """VIDEO_PATHS = [
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
"""


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text.rstrip() + "\n", encoding="utf-8")


def join_blocks(names: list[str]) -> str:
    return "\n\n\n".join(get_node_source(name) for name in names)


write(
    "src/config.py",
    f'''"""Configuration, dataclasses, and small shared utilities."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


{CONFIG_CONSTANTS}


{join_blocks([
    "PipelineConfig",
    "VideoMeta",
    "clamp",
    "format_elapsed",
    "ensure_report_dir",
    "load_pipeline_config",
    "validate_config",
])}
''',
)

write(
    "src/rendering.py",
    f'''"""Video rendering helpers."""

from typing import Tuple

from moviepy import VideoFileClip, concatenate_videoclips

from src.config import PipelineConfig


{join_blocks(["clip_subclip", "clip_resize", "clip_with_audio", "build_video"])}
''',
)

write(
    "src/preprocess.py",
    f'''"""Input validation, metadata, timeline, and synced frame reads."""

import os
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from moviepy import VideoFileClip

from src.config import (
    FPS_RESAMPLING_APPLY,
    PREPROCESS_DIR,
    PipelineConfig,
    VideoMeta,
)


{join_blocks([
    "read_video_meta",
    "collect_video_meta",
    "maybe_resample_fps",
    "common_timeline_duration",
    "build_timeline",
    "normalize_frame_resolution",
    "read_synced_frame",
    "build_audio_sync_map",
    "print_preprocess_summary",
    "load_validated_context",
    "build_debug_times",
])}
''',
)

write(
    "src/features.py",
    f'''"""Pose, motion, and quality feature extraction."""

from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

from src.config import (
    MOTION_SMOOTH_WINDOW,
    OPTICAL_FLOW_FALLBACK_SCALE,
    POSE_MODEL_PATH,
    POSE_MOTION_SCALE,
    QUALITY_SHARPNESS_SCALE,
    clamp,
)


{join_blocks([
    "init_pose_detector",
    "extract_pose_landmarks",
    "normalize_pose_keypoints",
    "pose_cosine_similarity",
    "pose_velocity_score",
    "extract_pose_score",
    "extract_quality_score",
])}


_motion_history: Dict[int, List[float]] = {{}}


{join_blocks(["optical_flow_motion_score", "combine_motion_score"])}
''',
)

write(
    "src/scoring.py",
    f'''"""Score calculation and component normalization."""

from typing import Dict, List

import numpy as np

from src.config import PipelineConfig, clamp


{join_blocks([
    "calculate_score",
    "compute_temporal_penalty",
    "robust_minmax_normalize",
    "normalize_score_components",
    "rebuild_scores_from_components",
])}
''',
)

write(
    "src/selection.py",
    f'''"""Pose-aware Top-K camera selection and segment creation."""

from typing import List, Optional

import numpy as np

from src.config import MIN_SWITCH_POSE_SIM, POSE_SWITCH_BONUS, PipelineConfig
from src.features import pose_cosine_similarity
from src.scoring import compute_temporal_penalty


{join_blocks([
    "compute_inter_cam_pose_similarity",
    "find_pose_cut_candidates",
    "switch_pose_similarity_at_time",
    "select_camera_topk",
    "make_segments",
])}
''',
)

write(
    "src/evaluation.py",
    f'''"""GT loading, metrics, baselines, ablation, and grid search."""

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


{join_blocks([
    "load_gt_cuts",
    "evaluate",
    "compute_topk_hit_rate",
    "run_baseline_random",
    "run_baseline_uniform",
    "run_baseline_pose_only",
    "run_baseline_quality_only",
    "run_ablation",
    "iter_grid_configs",
    "grid_validation_score",
    "run_grid_search",
])}
''',
)

write(
    "src/visualization.py",
    f'''"""Plots, report PDF, and pipeline summary outputs."""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from src.config import MIN_SWITCH_POSE_SIM, POSE_SWITCH_BONUS, PipelineConfig, VideoMeta


{join_blocks([
    "plot_timeline",
    "plot_score_heatmap",
    "plot_baseline_comparison",
    "plot_ablation",
    "export_report_pdf",
    "write_pipeline_summary",
])}
''',
)

write(
    "src/pipeline.py",
    '''"""CLI orchestration for the modular K-Pop multicam pipeline."""

import os
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional

import cv2
import numpy as np

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


''' + join_blocks([
        "analyze",
        "run_pose_debug",
        "run_similarity_debug",
        "run_selection_debug",
        "main",
        "print_usage",
        "run_cli",
    ]),
)
