#!/usr/bin/env python3
"""Turn demo/raw/demo_raw.webm + captions.json into demo.srt, demo.mp4, demo.gif.

Captions are burned in with ffmpeg's subtitles filter. ffmpeg comes from the
imageio-ffmpeg package, so no system install is needed.

    python demo/build_video.py
"""

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

DEMO = Path(__file__).resolve().parent
RAW = DEMO / "raw"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
STYLE = ("FontName=Arial,FontSize=11,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
         "BackColour=&H80000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=14,Alignment=2")


def ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def run(*args):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], check=True, cwd=RAW)


def main():
    data = json.loads((RAW / "captions.json").read_text(encoding="utf-8"))
    caps, trim, cut = data["captions"], data["trim"], data.get("cut")
    if cut:  # drop the LLM wait between cut[0] and cut[1]; shift later captions back
        gap = cut[1] - cut[0]
        caps = [c if c["start"] < cut[0] else {**c, "start": c["start"] - gap, "end": c["end"] - gap}
                for c in caps]
    caps.sort(key=lambda c: c["start"])
    srt = "\n".join(
        f"{i}\n{ts(c['start'])} --> {ts(c['end'])}\n{c['text']}\n"
        for i, c in enumerate(caps, 1)
    )
    (DEMO / "demo.srt").write_text(srt, encoding="utf-8")
    (RAW / "captions.srt").write_text(srt, encoding="utf-8")  # relative path for the filter

    run("-ss", str(trim), "-i", "demo_raw.webm",
        "-filter_complex", (
            f"[0:v]trim=0:{cut[0]},setpts=PTS-STARTPTS[a];"
            f"[0:v]trim=start={cut[1]},setpts=PTS-STARTPTS[b];"
            f"[a][b]concat=n=2:v=1[c];[c]subtitles=captions.srt:force_style='{STYLE}'"
        ) if cut else f"[0:v]subtitles=captions.srt:force_style='{STYLE}'",
        "-t", f"{caps[-1]['end'] + 0.5:.2f}",  # drop idle footage after the last caption
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-r", "25",
        "-movflags", "+faststart", "-an", str(DEMO / "demo.mp4"))

    # GIF: 800px wide, 8 fps, shared palette.
    run("-i", str(DEMO / "demo.mp4"),
        "-vf", "fps=8,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
               "[b][p]paletteuse=dither=bayer:bayer_scale=4",
        str(DEMO / "demo.gif"))
    for f in ("demo.mp4", "demo.gif"):
        print(f, round((DEMO / f).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
