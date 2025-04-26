from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.fx import MultiplySpeed
import glob

def concatenate_videos(video_paths, output_path, method='compose', multiply_speed_factor=1):
    # 1. Load each clip
    clips = [VideoFileClip(path) for path in video_paths]
    # clips = [clip.with_fps(fps) if fps is not None else clip for clip in clips]
    multiply_speed = MultiplySpeed(multiply_speed_factor)
    clips[0].subclipped
    clips = [multiply_speed.apply(clip.subclipped(
        start_time=5/clip.fps,
    )) for clip in clips]
    # 2. Concatenate
    final = concatenate_videoclips(clips, method=method)

    # # 3. (Optional) force a specific FPS
    # if fps is not None:
    #     final = final.set_fps(fps)

    # 4. Write the result
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        fps=final.fps
    )

    # 5. Clean up
    for clip in clips:
        clip.close()
    final.close()

# Example usage:
if __name__ == "__main__":
    video_dir = "video/bimanual/ours"
    video_dir = "video/dish/ours"
    video_dir = "video/construction_site/ours"
    video_dir = "video/construction_site_hard/ours"
    video_dir = "video/dish_multiple/ours"
    video_dir = "video/bimanual_hard/ours"
    # video_dir = "video/dish/curobo"
    videos = glob.glob(f"{video_dir}/*.mp4")
    videos = [video for video in videos if "final" not in video]
    sorted_videos = sorted(videos, key=lambda path: int(path.split("/")[-1].split(".")[0]))
    print(sorted_videos)
    concatenate_videos(sorted_videos, f"{video_dir}/final.mp4", method="compose", multiply_speed_factor=1.5)
