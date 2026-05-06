"""Video rendering helpers."""

from typing import Tuple

from moviepy import VideoFileClip, concatenate_videoclips

from src.config import PipelineConfig


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
