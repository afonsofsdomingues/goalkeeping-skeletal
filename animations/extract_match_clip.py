"""
Extract a specific clip from the full match video based on game clock time (minute:second).

Usage:
    python extract_match_clip.py <minute> <second> <length_seconds> [--first_half] [--input_file INPUT_FILE]

Example:
    python extract_match_clip.py 74 20 10 --first_half
"""

from moviepy import VideoFileClip
from datetime import datetime
import argparse

INPUT = "../data/euro2024/2036198/20240629_SUI-2-0-ITA_2036198/3-2024-2036198-Cam 1.mp4"

def utc_to_ts(utc_str: str) -> float:
    """Convert ISO UTC time to unix timestamp."""
    return datetime.fromisoformat(utc_str.replace("Z","+00:00")).timestamp()

FIRSTHALF_START_UTC = utc_to_ts("2024-06-29T16:01:14.504Z") # Corresponds to OFFSET second in the video
FIRSTHALF_END_UTC = utc_to_ts("2024-06-29T16:54:37.155Z")
SECONDHALF_START_UTC = utc_to_ts("2024-06-29T17:05:36.824Z")
SECONDHALF_START_UTC_IN_VID = SECONDHALF_START_UTC - FIRSTHALF_START_UTC

OFFSET = 9

def video_extract(game_min: int, game_sec: int, first_half: bool, length_s: float, input_file: str = INPUT):
    """
    Extract a clip starting at game_min:game_sec (football match time) for length_s seconds.

    To run it from the command line, do the following:

    python extract_clip.py 74 20 10 --first_half

    The above command extracts a 10-second clip starting at 74'20" in the first half.

    If it's the second half, just omit --first_half: python extract_clip.py 74 20 10
    """
    if first_half:
        # Convert football time to absolute seconds from start of file
        start_time_s = OFFSET + game_min * 60 + game_sec
    else:
        start_time_s = OFFSET + SECONDHALF_START_UTC_IN_VID + (game_min - 45) * 60 + game_sec
    
    with VideoFileClip(input_file) as video:
        clip = video.subclipped(start_time_s, start_time_s + length_s)
        clip.write_videofile(f"../data/videos/clip_{game_min}_{game_sec}.mp4", codec="libx264")
        clip.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract football clip from video.")
    parser.add_argument("minutes", type=int, help="Match minute")
    parser.add_argument("seconds", type=int, help="Match second")
    parser.add_argument("length", type=float, help="Clip length in seconds")
    parser.add_argument("--first_half", action="store_true", help="Flag if clip is in first half")
    parser.add_argument("--input_file", type=str, default=INPUT, help="Path to input video file")

    args = parser.parse_args()
    video_extract(args.minutes, args.seconds, args.first_half, args.length, input_file=args.input_file)