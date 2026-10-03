"""Pull the guitar out of a full mix (Demucs `htdemucs_6s`, which has a dedicated guitar stem),
and optionally download the song first (yt-dlp)."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .audio import load_audio, save_audio, SR


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def download(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise SystemExit("downloading needs yt-dlp:  uv pip install yt-dlp   (or pip install 'tonematch[download]')")
    tmpl = str(out_dir / "source.%(ext)s")
    cmd = [sys.executable, "-m", "yt_dlp", "-f", "bestaudio/best", "-x", "--audio-format", "wav",
           "--audio-quality", "0", "-o", tmpl, "--no-playlist", url]
    subprocess.run(cmd, check=True)
    wavs = sorted(out_dir.glob("source.*"))
    if not wavs:
        raise SystemExit("yt-dlp produced no file")
    return wavs[0]


def separate_guitar(src: str | Path, out_dir: Path, start: Optional[float] = None,
                    duration: Optional[float] = None, model: str = "htdemucs_6s",
                    device: Optional[str] = None, stem: str = "guitar") -> Path:
    """Return the path of the separated guitar stem (mono wav at SR) for the chosen excerpt."""
    try:
        import demucs  # noqa: F401
    except ImportError:
        raise SystemExit("separation needs demucs + torch:  uv pip install demucs   "
                         "(or pip install 'tonematch[separate]'). On Apple silicon add --device mps.")
    out_dir.mkdir(parents=True, exist_ok=True)
    x = load_audio(src, SR, start=start, duration=duration)
    tmp = out_dir / "excerpt.wav"
    save_audio(tmp, x, SR)
    cmd = [sys.executable, "-m", "demucs", "-n", model, "--two-stems", stem, "-o", str(out_dir / "demucs"), str(tmp)]
    if device:
        cmd += ["-d", device]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    found = list((out_dir / "demucs").rglob(f"{stem}.wav"))
    if not found:
        raise SystemExit("demucs finished but no guitar stem was found")
    dst = out_dir / f"{stem}.wav"
    shutil.copy(found[0], dst)
    return dst
