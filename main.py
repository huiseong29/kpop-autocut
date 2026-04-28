import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips


VIDEO_PATHS = [
    "stage1.mp4",
    "stage2.mp4",
    "stage3.mp4",
    "stage4.mp4",
]

OUTPUT_PATH = "edited_output.mp4"

OFFSETS_SEC = {
    "stage1.mp4": 3.5,
    "stage2.mp4": 29.5,
    "stage3.mp4": 0.1,
    "stage4.mp4": 0.0,
}

# Phase 1: analysis unit is parameterized in seconds.
ANALYSIS_STEP_SEC = 0.5
MIN_SCENE_LEN_SEC = 2.0
SWITCH_THRESHOLD = 0.15

# Phase 1: preprocessing targets.
# If None, keep original. If tuple, normalize all analysis frames to this size.
TARGET_RESOLUTION: Optional[Tuple[int, int]] = (1280, 720)


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
    try:
        for t in timeline:
            print("analyzing time:", round(t, 2))
            row = []
            for cap, meta in zip(caps, metas):
                frame = read_synced_frame(cap, meta, t, config.target_resolution)
                row.append(0.0 if frame is None else score_frame(frame, detector_kind, detector))
            times.append(t)
            scores.append(row)
    finally:
        if detector_kind == "mediapipe" and hasattr(detector, "close"):
            detector.close()

    for cap in caps:
        cap.release()

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


def build_video(segments, config: PipelineConfig):
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

    base_audio = clips[0].subclipped(
        config.offsets_sec[config.video_paths[0]],
        config.offsets_sec[config.video_paths[0]] + segments[-1][2],
    ).audio
    final = final.with_audio(base_audio)

    final.write_videofile(config.output_path, codec="libx264", audio_codec="aac")

    final.close()
    for clip in clips:
        clip.close()


def print_preprocess_summary(config: PipelineConfig, metas: List[VideoMeta], total_sec: float):
    print("=== Preprocess Summary (Phase 1) ===")
    print("videos:", len(config.video_paths))
    print("analysis_step_sec:", config.analysis_step_sec)
    print("target_resolution:", config.target_resolution)
    print("common_timeline_sec:", round(total_sec, 2))
    for meta in metas:
        print(
            f"- {meta.path}: fps={meta.fps:.3f}, res={meta.width}x{meta.height}, "
            f"offset={meta.offset_sec:.2f}, usable={meta.usable_duration_sec:.2f}s"
        )


def main():
    print("main started")

    config = load_pipeline_config()
    validate_config(config)
    metas = collect_video_meta(config)
    total_sec = common_timeline_duration(metas)
    print_preprocess_summary(config, metas, total_sec)

    times, scores, total_sec = analyze(config, metas)
    print("analysis finished")

    chosen = select_camera(times, scores, config)
    print("camera selection finished")

    segments = make_segments(times, chosen, total_sec)
    print("segment generation finished")

    print("video rendering started")
    build_video(segments, config)
    print("done:", config.output_path)


if __name__ == "__main__":
    main()
