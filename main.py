import os
import sys
import time
import csv
import json
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from moviepy import VideoFileClip, concatenate_videoclips


# =========================================================
# K-Pop Multicam Auto Editing - Final Version with GT Evaluation
# 엔딩요정 강제 고정 없음: 원래 자동 편집 방식 유지
# =========================================================
# 핵심 개선점
# 1) MediaPipe Tasks Pose Landmarker Lite 사용
# 2) 33 keypoints 추출
# 3) Hip 기준 정렬 + scale normalization
# 4) pose cosine similarity를 컷 후보/카메라 전환 점수에 직접 반영
# 5) Motion은 optical flow보다 안정적인 keypoint velocity 중심으로 계산
# 6) 카메라별 p/m/q component robust normalization
# 7) Baseline, Ablation, Grid Search, Visualization, Report까지 단일 실행
# =========================================================


# ── User Settings ─────────────────────────────────────────

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

# 0.5초 단위 분석이면 185초 영상에서 약 371개 step 생성
ANALYSIS_STEP_SEC = 0.5

# 기존 fixed 결과가 5.3초 평균 컷으로 조금 안정적이었으므로,
# 최종본은 K-pop 리듬감을 위해 살짝 더 다이나믹하게 조정
MIN_SCENE_LEN_SEC = 1.7
SWITCH_THRESHOLD = 0.10

TARGET_RESOLUTION: Optional[Tuple[int, int]] = (1280, 720)

# Grid Search 결과를 반영한 최종 기본 가중치
# 이전 결과: alpha=0.3 beta=0.1 gamma=0.35 delta=0.25가 best
ALPHA = 0.30   # Pose
BETA = 0.10    # Motion energy penalty
GAMMA = 0.35   # Quality
DELTA = 0.25   # Temporal penalty

TOP_K = 3

# Pose similarity gating
POSE_CUT_SIM_THRESHOLD = 0.72
POSE_CUT_FALLBACK_PERCENTILE = 40.0
MIN_SWITCH_POSE_SIM = 0.42
POSE_SWITCH_BONUS = 0.25

# Motion: pose keypoint velocity 중심
POSE_MOTION_SCALE = 1.25
OPTICAL_FLOW_FALLBACK_SCALE = 30.0
MOTION_SMOOTH_WINDOW = 5

# Quality: sharpness + contrast + exposure
QUALITY_SHARPNESS_SCALE = 500.0

COMPONENT_NORMALIZE = True
SELECTION_DEBUG_DURATION_SEC = 4.0

REPORT_DIR = "report"
PREPROCESS_DIR = os.path.join(REPORT_DIR, "preprocessed")

FPS_RESAMPLING_APPLY = False

GT_CUTS_PATH = "gt_cuts.csv"  # 없으면 자동 skip

GRID_ALPHA_VALUES = [0.25, 0.30, 0.35, 0.40]
GRID_BETA_VALUES = [0.05, 0.10, 0.15, 0.20]
GRID_GAMMA_VALUES = [0.25, 0.30, 0.35, 0.40]
GRID_DELTA_VALUES = [0.15, 0.20, 0.25, 0.30]


# ── Data Classes ──────────────────────────────────────────

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


# ── Utility ───────────────────────────────────────────────

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


# ── MoviePy Compatibility ─────────────────────────────────

def clip_subclip(clip, start: float, end: float):
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start, end)
    return clip.subclip(start, end)


def clip_resize(clip, size: Tuple[int, int]):
    if hasattr(clip, "resized"):
        return clip.resized(new_size=size)
    return clip.resize(newsize=size)


def clip_with_audio(clip, audio):
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio)
    return clip.set_audio(audio)


# ── Phase 1: Video Meta / Timeline ─────────────────────────

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

    print(f"[Phase 1] Reference FPS: {ref_fps:.3f} | Max FPS diff: {max_fps_diff:.3f}")
    return metas


def maybe_resample_fps(
    config: PipelineConfig,
    metas: List[VideoMeta],
    tolerance: float = 0.5,
) -> Tuple[PipelineConfig, List[VideoMeta]]:
    ref_fps = metas[0].fps
    needs_resample = any(abs(m.fps - ref_fps) > tolerance for m in metas)

    if not needs_resample:
        print("[Phase 1] FPS resampling skipped: all inputs are within tolerance.")
        return config, metas

    if not FPS_RESAMPLING_APPLY:
        print(
            "[Phase 1] FPS mismatch detected, but automatic resampling is disabled "
            "to preserve original source timing."
        )
        return config, metas

    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    new_paths = []
    new_offsets = {}

    print(f"[Phase 1] FPS resampling started: target_fps={ref_fps:.3f}")

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
            print(f"[Phase 1] FPS resampled: {meta.path} -> {out_path}")
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


# ── Phase 1: Audio Sync Structure ──────────────────────────

def build_audio_sync_map(metas: List[VideoMeta], ref_idx: int = 0) -> Dict[str, float]:
    ref_offset = metas[ref_idx].offset_sec
    sync_map = {}

    for m in metas:
        sync_map[m.path] = m.offset_sec
        drift = m.offset_sec - ref_offset
        print(
            f"[Phase 1] AudioSync: {m.path} "
            f"offset={m.offset_sec:.3f}s drift_from_ref={drift:+.3f}s"
        )
    return sync_map


# ── Phase 2: Pose Track ───────────────────────────────────

def init_pose_detector():
    """MediaPipe Tasks Pose Landmarker Lite 사용.

    같은 폴더에 pose_landmarker_lite.task 파일이 있어야 한다.
    """
    if not os.path.exists(POSE_MODEL_PATH):
        print(f"[Pose] model missing: {POSE_MODEL_PATH}")
        print("[Pose] download pose_landmarker_lite.task and place it next to this script.")
        return None

    base_options = mp_tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_pose_landmarks(frame: np.ndarray, pose_det) -> Optional[np.ndarray]:
    """33개 keypoints 추출. shape=(33,4), columns=x,y,z,visibility."""
    if pose_det is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    results = pose_det.detect(image)

    if not results.pose_landmarks:
        return None

    keypoints = []
    for landmark in results.pose_landmarks[0]:
        visibility = getattr(landmark, "visibility", 1.0)
        keypoints.append((landmark.x, landmark.y, landmark.z, visibility))

    return np.array(keypoints, dtype=np.float32)


def normalize_pose_keypoints(keypoints: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Hip 기준 정렬 + scale normalization."""
    if keypoints is None or keypoints.shape != (33, 4):
        return None

    normalized = keypoints.copy()

    # 23, 24 = left/right hip
    hip_center = normalized[[23, 24], :3].mean(axis=0)
    normalized[:, :3] -= hip_center

    # scale: shoulder/hip landmark의 최대 거리 기준
    shoulders = normalized[[11, 12], :3]
    hips = normalized[[23, 24], :3]
    torso_points = np.vstack([shoulders, hips])
    scale = float(np.max(np.linalg.norm(torso_points, axis=1)))

    if scale <= 1e-6:
        return None

    normalized[:, :3] /= scale
    return normalized


def pose_cosine_similarity(
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
    visibility_threshold: float = 0.5,
) -> float:
    """정규화된 포즈 벡터 간 cosine similarity. 결과 0~1."""
    na = normalize_pose_keypoints(a)
    nb = normalize_pose_keypoints(b)

    if na is None or nb is None:
        return 0.5

    visible = (na[:, 3] >= visibility_threshold) & (nb[:, 3] >= visibility_threshold)
    if int(np.sum(visible)) < 8:
        return 0.5

    va = na[visible, :3].reshape(-1)
    vb = nb[visible, :3].reshape(-1)

    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-8:
        return 0.5

    sim = float(np.dot(va, vb) / denom)
    return clamp((sim + 1.0) / 2.0, 0.0, 1.0)


def pose_velocity_score(
    prev_pose: Optional[np.ndarray],
    curr_pose: Optional[np.ndarray],
    visibility_threshold: float = 0.5,
) -> float:
    """Optical flow 대신 keypoint velocity로 motion energy 계산.

    포즈가 빠르게 변하는 순간은 컷을 조금 더 보수적으로 만든다.
    """
    prev_norm = normalize_pose_keypoints(prev_pose)
    curr_norm = normalize_pose_keypoints(curr_pose)

    if prev_norm is None or curr_norm is None:
        return 0.0

    visible = (prev_norm[:, 3] >= visibility_threshold) & (curr_norm[:, 3] >= visibility_threshold)
    if int(np.sum(visible)) < 8:
        return 0.0

    diff = curr_norm[visible, :3] - prev_norm[visible, :3]
    velocity = float(np.mean(np.linalg.norm(diff, axis=1)))
    return clamp(velocity / POSE_MOTION_SCALE, 0.0, 1.0)


def extract_pose_score(
    frame: np.ndarray,
    pose_det,
    prev_landmarks: Optional[np.ndarray],
) -> Tuple[float, Optional[np.ndarray], float, float]:
    """pose_score, curr_pose, temporal_pose_sim, pose_motion 반환."""
    curr = extract_pose_landmarks(frame, pose_det)

    if curr is None:
        return 0.0, None, 0.5, 0.0

    key_idx = [11, 12, 23, 24]  # shoulders + hips
    vis_score = float(np.mean(curr[key_idx, 3]))

    norm_curr = normalize_pose_keypoints(curr)
    if norm_curr is None:
        return 0.0, curr, 0.5, 0.0

    # 원본 화면에서 몸 중심이 중앙에 가까운지
    torso_cx = float(np.mean(curr[key_idx, 0]))
    torso_cy = float(np.mean(curr[key_idx, 1]))
    dist = ((torso_cx - 0.5) ** 2 + (torso_cy - 0.5) ** 2) ** 0.5
    center_score = 1.0 - clamp(dist / 0.5, 0.0, 1.0)

    temporal_sim = pose_cosine_similarity(prev_landmarks, curr)
    pose_motion = pose_velocity_score(prev_landmarks, curr)

    pose_score = 0.45 * vis_score + 0.25 * center_score + 0.30 * temporal_sim
    return clamp(pose_score, 0.0, 1.0), curr, temporal_sim, pose_motion


# ── Phase 2: Quality Track ────────────────────────────────

def extract_quality_score(gray: np.ndarray) -> float:
    """sharpness + contrast + exposure 기반 quality score."""
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp_score = clamp(sharp / QUALITY_SHARPNESS_SCALE, 0.0, 1.0)

    contrast = float(np.std(gray))
    contrast_score = clamp(contrast / 64.0, 0.0, 1.0)

    mean_brightness = float(np.mean(gray))
    exposure_score = 1.0 - clamp(abs(mean_brightness - 127.5) / 127.5, 0.0, 1.0)

    quality = 0.60 * sharp_score + 0.25 * contrast_score + 0.15 * exposure_score
    return clamp(quality, 0.0, 1.0)


# ── Phase 2: Motion Track ─────────────────────────────────

_motion_history: Dict[int, List[float]] = {}


def optical_flow_motion_score(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
) -> float:
    if prev_gray is None:
        return 0.0

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
    return clamp(mag / OPTICAL_FLOW_FALLBACK_SCALE, 0.0, 1.0)


def combine_motion_score(
    pose_motion: float,
    flow_motion: float,
    cam_idx: int,
    smooth_window: int = MOTION_SMOOTH_WINDOW,
) -> float:
    # pose motion이 있으면 그것을 우선, flow는 보조
    motion = 0.75 * pose_motion + 0.25 * flow_motion

    history = _motion_history.setdefault(cam_idx, [])
    history.append(motion)

    if len(history) > smooth_window:
        history.pop(0)

    return clamp(float(np.mean(history)), 0.0, 1.0)


# ── Phase 3: Scoring ──────────────────────────────────────

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


# ── Phase 4: Cut Candidate / Selection ────────────────────

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
        f"[Phase 4] PoseCutCandidates: {n_cands}/{len(flags)} "
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
        f"[Phase 4] CutSelection: attempted={total_attempted} "
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


# ── Analysis Loop ─────────────────────────────────────────

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
        print("[Phase 3] Component normalization applied: p/m/q camera-wise robust min-max.")

    return times, scores, inter_cam_sims, score_components, pose_history, total_sec


# ── Phase 6: Evaluation / Baselines ───────────────────────

def load_gt_cuts(path: str = GT_CUTS_PATH) -> Optional[List[Tuple[float, int]]]:
    """GT 컷 파일을 불러온다.

    지원:
    - CSV: time_sec, camera(optional)
    - JSON: [{"time_sec": 10.0, "camera": 1}, ...]
    """
    if not os.path.exists(path):
        print(f"[Phase 6] GT file not found: {path}. Evaluation MAE/acc will be skipped.")
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
    print(f"[Phase 6] Top-{k} Hit Rate: {rate:.4f} ({hits}/{len(chosen)})")
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
        print(f"[Phase 6 Ablation] {name}: {metrics}")

    return results


# ── Phase 7: Grid Search ──────────────────────────────────

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

    print(f"[Phase 7] GridSearch: {len(candidates)} candidates")

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
            "[Phase 7] GridSearch best: "
            f"α={best_config.alpha} β={best_config.beta} "
            f"γ={best_config.gamma} δ={best_config.delta} "
            f"score={best['validation_score']}"
        )
    else:
        best_config = config
        print("[Phase 7] GridSearch skipped: no valid weight combinations.")

    print(f"[Phase 7] GridSearch log saved: {log_path}")
    return best_config, rows


# ── Phase 8: Visualization / Report ───────────────────────

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
    print(f"[Phase 8] Timeline saved: {save_path}")


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
    print(f"[Phase 8] Heatmap saved: {save_path}")


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
    print(f"[Phase 8] Baseline comparison saved: {save_path}")


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
    print(f"[Phase 8] Ablation graph saved: {save_path}")


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

    print(f"[Phase 8] PDF report saved: {save_path}")


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
        f.write("Phase 9 Done: full pipeline executed\n")
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

    print(f"[Phase 9] Pipeline summary saved: {save_path}")


# ── Phase 5: Video Rendering ──────────────────────────────

def build_video(segments, config: PipelineConfig, total_sec: float) -> None:
    clips = []
    output_clips = []
    final = None

    try:
        clips = [VideoFileClip(path) for path in config.video_paths]

        for cam, start_sec, end_sec in segments:
            path = config.video_paths[cam]
            offset_sec = config.offsets_sec[path]

            start = start_sec + offset_sec
            end = end_sec + offset_sec

            clip = clip_subclip(clips[cam], start, end)

            if config.target_resolution is not None:
                clip = clip_resize(clip, config.target_resolution)

            output_clips.append(clip)

        final = concatenate_videoclips(output_clips)

        base_path = config.video_paths[0]
        base_offset = config.offsets_sec[base_path]
        base_audio_clip = clip_subclip(clips[0], base_offset, base_offset + total_sec)

        if base_audio_clip.audio is not None:
            final = clip_with_audio(final, base_audio_clip.audio)

        final.write_videofile(config.output_path, codec="libx264", audio_codec="aac")

    finally:
        if final is not None:
            final.close()

        for c in output_clips:
            try:
                c.close()
            except Exception:
                pass

        for c in clips:
            try:
                c.close()
            except Exception:
                pass


# ── Summary / Context ─────────────────────────────────────

def print_preprocess_summary(config: PipelineConfig, metas: List[VideoMeta], total_sec: float) -> None:
    print("=== Preprocess Summary (Phase 1) ===")
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
    print("[Phase 1] AudioSync: using fixed manual offsets from OFFSETS_SEC.")

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


# ── Debug CLI ─────────────────────────────────────────────

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

    print("segments:")
    for cam, start_sec, end_sec in segments:
        print(f"  cam={cam} start={start_sec:.2f}s end={end_sec:.2f}s")

    print("selection debug elapsed:", format_elapsed(time.perf_counter() - run_start))


# ── Main ──────────────────────────────────────────────────

def main() -> None:
    run_start = time.perf_counter()
    print("main started")

    ensure_report_dir(REPORT_DIR)

    config, metas, total_sec = load_validated_context()
    print_preprocess_summary(config, metas, total_sec)
    build_audio_sync_map(metas)

    gt_cuts = load_gt_cuts(GT_CUTS_PATH)

    times, scores, inter_cam_sims, score_components, pose_history, total_sec = analyze(config, metas)
    print("analysis finished")

    pose_cut_flags = find_pose_cut_candidates(times, inter_cam_sims, config)

    chosen = select_camera_topk(times, scores, pose_cut_flags, config, pose_history)
    print("camera selection finished")

    segments = make_segments(times, chosen, total_sec)
    print(f"segments: {len(segments)}")

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


if __name__ == "__main__":
    run_cli()
