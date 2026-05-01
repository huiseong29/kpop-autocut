import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from moviepy import VideoFileClip, concatenate_videoclips


VIDEO_PATHS = [
    "stage1.mp4",
    "stage2.mp4",
    "stage3.mp4",
    "stage4.mp4",
]

OUTPUT_PATH = "edited_output.mp4"
POSE_MODEL_PATH = "pose_landmarker_lite.task"

OFFSETS_SEC = {
    "stage1.mp4": 4.0,
    "stage2.mp4": 29.0,
    "stage3.mp4": 0.0,
    "stage4.mp4": 0.0,
}

# Analysis unit is parameterized in seconds.
ANALYSIS_STEP_SEC = 0.5
MIN_SCENE_LEN_SEC = 2.0
SWITCH_THRESHOLD = 0.15

# Preprocessing targets.
# If None, keep original. If tuple, normalize all analysis frames to this size.
TARGET_RESOLUTION: Optional[Tuple[int, int]] = (1280, 720)
DEBUG_SAMPLE_COUNT = 12


def init_face_detector():
    # mediapipe.solutions is not always available depending on mediapipe build/runtime.
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        return "mediapipe", detector

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade face detector.")
    return "haar", cascade


def init_pose_estimator():
    if not os.path.exists(POSE_MODEL_PATH):
        print(f"pose model missing: {POSE_MODEL_PATH}")
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


@dataclass(frozen=True)
class PipelineConfig:
    video_paths: List[str]
    offsets_sec: Dict[str, float]
    analysis_step_sec: float
    target_resolution: Optional[Tuple[int, int]]
    min_scene_len_sec: float
    switch_threshold: float
    output_path: str


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
        raise ValueError(f"Invalid FPS in video: {path}")
    if frame_count <= 0:
        raise ValueError(f"Invalid frame count in video: {path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resolution in video: {path}")

    duration_sec = frame_count / fps
    usable_duration_sec = duration_sec - offset_sec
    if usable_duration_sec <= 0:
        raise ValueError(
            f"Usable duration <= 0 after offset. path={path}, "
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
    return metas


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
    frame: np.ndarray, target_resolution: Optional[Tuple[int, int]]
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


def sharpness(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_pose_keypoints(frame: np.ndarray, pose_estimator) -> Optional[np.ndarray]:
    if pose_estimator is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    results = pose_estimator.detect(image)
    if not results.pose_landmarks:
        return None

    keypoints = []
    for landmark in results.pose_landmarks[0]:
        visibility = getattr(landmark, "visibility", 1.0)
        keypoints.append((landmark.x, landmark.y, landmark.z, visibility))
    return np.array(keypoints, dtype=np.float32)


def normalize_pose_keypoints(keypoints: np.ndarray) -> Optional[np.ndarray]:
    if keypoints.shape != (33, 4):
        return None

    normalized = keypoints.copy()
    hips = normalized[[23, 24], :3]
    center = hips.mean(axis=0)
    normalized[:, :3] -= center

    shoulders = normalized[[11, 12], :3]
    torso_points = np.vstack([hips - center, shoulders])
    scale = float(np.max(np.linalg.norm(torso_points, axis=1)))
    if scale <= 1e-6:
        return None

    normalized[:, :3] /= scale
    return normalized


def pose_similarity(a: np.ndarray, b: np.ndarray, normalize: bool = False) -> Optional[float]:
    if normalize:
        a = normalize_pose_keypoints(a)
        b = normalize_pose_keypoints(b)
        if a is None or b is None:
            return None

    visible = (a[:, 3] >= 0.5) & (b[:, 3] >= 0.5)
    if int(np.sum(visible)) < 8:
        return None

    va = a[visible, :3].reshape(-1)
    vb = b[visible, :3].reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-6:
        return None

    return float(np.dot(va, vb) / denom)


def build_debug_times(total_sec: float, sample_count: int = DEBUG_SAMPLE_COUNT) -> List[float]:
    if sample_count <= 0:
        raise ValueError("sample_count must be > 0.")
    if total_sec <= 0:
        raise ValueError("total_sec must be > 0.")

    end = max(0.0, total_sec - 1.0)
    if sample_count == 1:
        return [0.0]
    return [float(t) for t in np.linspace(0.0, end, sample_count)]


def load_validated_context():
    config = load_pipeline_config()
    validate_config(config)
    metas = collect_video_meta(config)
    total_sec = common_timeline_duration(metas)
    return config, metas, total_sec


def collect_pose_samples(
    config: PipelineConfig,
    metas: List[VideoMeta],
    sample_times: List[float],
) -> List[List[Optional[np.ndarray]]]:
    caps = [cv2.VideoCapture(meta.path) for meta in metas]
    pose_estimator = init_pose_estimator()
    print("pose estimator ready:", pose_estimator is not None)
    samples: List[List[Optional[np.ndarray]]] = []

    try:
        for t in sample_times:
            row: List[Optional[np.ndarray]] = []
            for cap, meta in zip(caps, metas):
                frame = read_synced_frame(cap, meta, t, config.target_resolution)
                if frame is None:
                    row.append(None)
                    continue
                row.append(extract_pose_keypoints(frame, pose_estimator))
            samples.append(row)
    finally:
        if pose_estimator is not None:
            pose_estimator.close()
        for cap in caps:
            cap.release()

    return samples


def detect_face_bboxes(frame: np.ndarray, detector_kind: str, detector) -> List[Tuple[float, float, float, float]]:
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return []

    if detector_kind == "mediapipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if not results.detections:
            return []

        bboxes = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            bboxes.append((bbox.xmin, bbox.ymin, bbox.width, bbox.height))
        return bboxes

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    bboxes = []
    for x, y, fw, fh in faces:
        bboxes.append((x / w, y / h, fw / w, fh / h))
    return bboxes


def score_frame(frame: np.ndarray, detector_kind: str, detector) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharp = sharpness(gray)
    sharp_score = clamp(sharp / 500.0, 0.0, 1.0)

    bboxes = detect_face_bboxes(frame, detector_kind, detector)
    if not bboxes:
        return 0.2 * sharp_score

    best = 0.0
    for xmin, ymin, bw, bh in bboxes:
        area = bw * bh
        size_score = clamp(area / 0.15, 0.0, 1.0)

        cx = xmin + bw / 2
        cy = ymin + bh / 2
        center_dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
        center_score = 1 - clamp(center_dist / 0.7, 0.0, 1.0)

        total = 0.5 * size_score + 0.3 * center_score + 0.2 * sharp_score
        best = max(best, total)
    return best


def analyze(config: PipelineConfig, metas: List[VideoMeta]):
    total_sec = common_timeline_duration(metas)
    timeline = build_timeline(total_sec, config.analysis_step_sec)
    caps = [cv2.VideoCapture(meta.path) for meta in metas]

    times: List[float] = []
    scores: List[List[float]] = []

    detector_kind, detector = init_face_detector()
    print("detector ready:", detector_kind)
    pose_estimator = init_pose_estimator()
    pose_enabled = pose_estimator is not None
    pose_checked = 0
    pose_detected = 0
    print("pose estimator ready:", pose_enabled)
    try:
        for t in timeline:
            print("analyzing time:", round(t, 2))
            row = []
            for cap, meta in zip(caps, metas):
                frame = read_synced_frame(cap, meta, t, config.target_resolution)
                if frame is None:
                    row.append(0.0)
                    continue

                pose_keypoints = extract_pose_keypoints(frame, pose_estimator)
                pose_checked += 1
                if pose_keypoints is not None:
                    pose_detected += 1

                row.append(score_frame(frame, detector_kind, detector))
            times.append(t)
            scores.append(row)
    finally:
        if detector_kind == "mediapipe" and hasattr(detector, "close"):
            detector.close()
        if pose_estimator is not None:
            pose_estimator.close()

    for cap in caps:
        cap.release()

    if pose_checked:
        rate = pose_detected / pose_checked
        print(f"pose detection: {pose_detected}/{pose_checked} ({rate:.1%})")

    return times, scores, total_sec


def select_camera(times: List[float], scores: List[List[float]], config: PipelineConfig):
    if not scores:
        raise ValueError("No score data generated.")

    chosen: List[int] = []
    cur = int(np.argmax(scores[0]))
    last_switch_t = times[0]
    chosen.append(cur)

    for i in range(1, len(times)):
        best = int(np.argmax(scores[i]))
        if (
            best != cur
            and (times[i] - last_switch_t) >= config.min_scene_len_sec
            and (scores[i][best] - scores[i][cur]) > config.switch_threshold
        ):
            cur = best
            last_switch_t = times[i]
        chosen.append(cur)

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
            cam = chosen[i]
            start = times[i]

    segments.append((cam, start, total_sec))
    return segments


def build_video(segments, config: PipelineConfig, total_sec: float):
    clips = [VideoFileClip(path) for path in config.video_paths]
    output_clips = []

    for cam, start_sec, end_sec in segments:
        path = config.video_paths[cam]
        offset_sec = config.offsets_sec[path]

        clip = clips[cam].subclipped(start_sec + offset_sec, end_sec + offset_sec)
        if config.target_resolution is not None:
            clip = clip.resized(new_size=config.target_resolution)
        output_clips.append(clip)

    final = concatenate_videoclips(output_clips)

    offset = config.offsets_sec[config.video_paths[0]]
    base_audio = clips[0].subclipped(
        offset,
        offset + total_sec,
    ).audio
    final = final.with_audio(base_audio)

    final.write_videofile(config.output_path, codec="libx264", audio_codec="aac")

    final.close()
    for clip in clips:
        clip.close()


def print_preprocess_summary(config: PipelineConfig, metas: List[VideoMeta], total_sec: float):
    print("=== Preprocess Summary ===")
    print("videos:", len(config.video_paths))
    print("analysis_step_sec:", config.analysis_step_sec)
    print("target_resolution:", config.target_resolution)
    print("common_timeline_sec:", round(total_sec, 2))
    for meta in metas:
        print(
            f"- {meta.path}: fps={meta.fps:.3f}, res={meta.width}x{meta.height}, "
            f"offset={meta.offset_sec:.2f}, usable={meta.usable_duration_sec:.2f}s"
        )


def format_elapsed(seconds: float) -> str:
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes}m {remain:.1f}s"


def main():
    run_start = time.perf_counter()
    print("main started")

    step_start = time.perf_counter()
    config = load_pipeline_config()
    validate_config(config)
    metas = collect_video_meta(config)
    total_sec = common_timeline_duration(metas)
    print_preprocess_summary(config, metas, total_sec)
    print("preprocess elapsed:", format_elapsed(time.perf_counter() - step_start))

    step_start = time.perf_counter()
    times, scores, total_sec = analyze(config, metas)
    print("analysis finished")
    print("analysis elapsed:", format_elapsed(time.perf_counter() - step_start))

    step_start = time.perf_counter()
    chosen = select_camera(times, scores, config)
    print("camera selection finished")
    print("camera selection elapsed:", format_elapsed(time.perf_counter() - step_start))

    step_start = time.perf_counter()
    segments = make_segments(times, chosen, total_sec)
    print("segment generation finished")
    print("segment generation elapsed:", format_elapsed(time.perf_counter() - step_start))

    step_start = time.perf_counter()
    print("video rendering started")
    build_video(segments, config, total_sec)
    print("video rendering elapsed:", format_elapsed(time.perf_counter() - step_start))

    total_elapsed = time.perf_counter() - run_start
    print("total elapsed:", format_elapsed(total_elapsed))
    print("done:", config.output_path)


def run_pose_debug():
    run_start = time.perf_counter()
    print("pose debug started")

    config, metas, total_sec = load_validated_context()
    print_preprocess_summary(config, metas, total_sec)
    sample_times = build_debug_times(total_sec)
    samples = collect_pose_samples(config, metas, sample_times)

    checked = 0
    detected = 0
    visibility_scores = []

    for t, row in zip(sample_times, samples):
        print(f"time={t:.2f}s")
        for cam_idx, keypoints in enumerate(row):
            checked += 1
            if keypoints is None:
                print(f"  cam={cam_idx} detected=False")
                continue

            detected += 1
            mean_visibility = float(np.mean(keypoints[:, 3]))
            visibility_scores.append(mean_visibility)
            print(
                f"  cam={cam_idx} detected=True "
                f"shape={keypoints.shape} mean_visibility={mean_visibility:.3f}"
            )

    rate = detected / checked if checked else 0.0
    avg_visibility = float(np.mean(visibility_scores)) if visibility_scores else 0.0
    print(f"pose detection: {detected}/{checked} ({rate:.1%})")
    print(f"average visibility: {avg_visibility:.3f}")
    print("pose debug elapsed:", format_elapsed(time.perf_counter() - run_start))


def run_similarity_debug():
    run_start = time.perf_counter()
    print("similarity debug started")

    config, metas, total_sec = load_validated_context()
    print_preprocess_summary(config, metas, total_sec)
    sample_times = build_debug_times(total_sec)
    samples = collect_pose_samples(config, metas, sample_times)

    positive_raw = []
    positive_norm = []
    negative_norm = []

    for row in samples:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if row[i] is None or row[j] is None:
                    continue
                raw = pose_similarity(row[i], row[j], normalize=False)
                norm = pose_similarity(row[i], row[j], normalize=True)
                if raw is not None:
                    positive_raw.append(raw)
                if norm is not None:
                    positive_norm.append(norm)

    for prev, cur in zip(samples, samples[1:]):
        for cam_idx in range(len(cur)):
            if prev[cam_idx] is None or cur[cam_idx] is None:
                continue
            norm = pose_similarity(prev[cam_idx], cur[cam_idx], normalize=True)
            if norm is not None:
                negative_norm.append(norm)

    def mean_or_none(values: List[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    positive_raw_mean = mean_or_none(positive_raw)
    positive_norm_mean = mean_or_none(positive_norm)
    negative_norm_mean = mean_or_none(negative_norm)
    gap = (
        positive_norm_mean - negative_norm_mean
        if positive_norm_mean is not None and negative_norm_mean is not None
        else None
    )

    print(f"positive_raw_pairs: {len(positive_raw)}")
    print(f"positive_normalized_pairs: {len(positive_norm)}")
    print(f"negative_normalized_pairs: {len(negative_norm)}")
    print(f"positive_raw_mean: {positive_raw_mean}")
    print(f"positive_normalized_mean: {positive_norm_mean}")
    print(f"negative_normalized_mean: {negative_norm_mean}")
    print(f"gap: {gap}")
    print("similarity debug elapsed:", format_elapsed(time.perf_counter() - run_start))


def run_selection_debug():
    run_start = time.perf_counter()
    print("selection debug started")

    config, metas, total_sec = load_validated_context()
    print_preprocess_summary(config, metas, total_sec)
    sample_times = build_debug_times(total_sec)

    caps = [cv2.VideoCapture(meta.path) for meta in metas]
    detector_kind, detector = init_face_detector()
    print("detector ready:", detector_kind)

    scores: List[List[float]] = []
    try:
        for t in sample_times:
            row = []
            for cap, meta in zip(caps, metas):
                frame = read_synced_frame(cap, meta, t, config.target_resolution)
                row.append(score_frame(frame, detector_kind, detector) if frame is not None else 0.0)
            scores.append(row)
            best = int(np.argmax(row))
            score_text = ", ".join(f"cam{i}={score:.3f}" for i, score in enumerate(row))
            print(f"time={t:.2f}s best_cam={best} scores=[{score_text}]")
    finally:
        if detector_kind == "mediapipe" and hasattr(detector, "close"):
            detector.close()
        for cap in caps:
            cap.release()

    chosen = select_camera(sample_times, scores, config)
    segments = make_segments(sample_times, chosen, sample_times[-1])
    print("selected camera sequence:", chosen)
    print("debug segments:")
    for cam, start_sec, end_sec in segments:
        print(f"  cam={cam} start={start_sec:.2f}s end={end_sec:.2f}s")
    print("selection debug elapsed:", format_elapsed(time.perf_counter() - run_start))


def print_usage():
    print("usage: python main.py [pose-debug|similarity-debug|selection-debug|full]")
    print("  pose-debug: pose keypoint extraction debug only")
    print("  similarity-debug: pose similarity debug only")
    print("  selection-debug: camera selection debug only")
    print("  full: full analyze + render pipeline")


def run_cli():
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
