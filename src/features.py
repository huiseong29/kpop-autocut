"""Pose, motion, and quality feature extraction."""

import os
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
