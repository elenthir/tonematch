"""Audio I/O, normalisation and a synthetic guitar DI used by tests / self-test."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

SR = 44100


def load_audio(path: str | Path, sr: int = SR, start: float | None = None,
               duration: float | None = None) -> np.ndarray:
    """Load any file soundfile/librosa can read, as mono float32 at `sr`."""
    path = str(path)
    try:
        info = sf.info(path)
        offset = int((start or 0.0) * info.samplerate)
        frames = int(duration * info.samplerate) if duration else -1
        x, file_sr = sf.read(path, start=offset, frames=frames, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:  # mp3/m4a on old libsndfile, etc. -> librosa/audioread
        import librosa
        x, file_sr = librosa.load(path, sr=None, mono=True, offset=start or 0.0, duration=duration)
        x = x.astype(np.float32)
    if file_sr != sr:
        import librosa
        x = librosa.resample(x, orig_sr=file_sr, target_sr=sr).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def save_audio(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    sf.write(str(path), x.astype(np.float32), sr)


def rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(float(np.sqrt(np.mean(np.square(x)) + 1e-12)))


def frame_rms_db(x: np.ndarray, sr: int, win_s: float = 0.05) -> np.ndarray:
    n = max(1, int(win_s * sr))
    m = len(x) // n
    if m == 0:
        return np.array([rms_db(x)])
    f = x[: m * n].reshape(m, n)
    return 20.0 * np.log10(np.sqrt(np.mean(f * f, axis=1)) + 1e-9)


def active_rms_db(x: np.ndarray, sr: int, gate_db: float = 40.0) -> float:
    """RMS over the frames that are within `gate_db` of the loudest frame (ignores silence)."""
    f = frame_rms_db(x, sr)
    keep = f > (f.max() - gate_db)
    lin = 10 ** (f[keep] / 20.0)
    return 20.0 * math.log10(float(np.sqrt(np.mean(lin * lin)) + 1e-9))


def normalize_active_rms(x: np.ndarray, sr: int, target_db: float = -18.0) -> np.ndarray:
    g = 10 ** ((target_db - active_rms_db(x, sr)) / 20.0)
    return (x * g).astype(np.float32)


def trim_silence(x: np.ndarray, sr: int, thresh_db: float = -50.0, pad_s: float = 0.05) -> np.ndarray:
    f = frame_rms_db(x, sr, 0.02)
    idx = np.where(f > thresh_db)[0]
    if idx.size == 0:
        return x
    n = int(0.02 * sr)
    a = max(0, idx[0] * n - int(pad_s * sr))
    b = min(len(x), (idx[-1] + 1) * n + int(pad_s * sr))
    return x[a:b]


def fade(x: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    n = min(len(x) // 2, int(sr * ms / 1000))
    if n <= 0:
        return x
    y = x.copy()
    r = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y[:n] *= r
    y[-n:] *= r[::-1]
    return y


def excerpt(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Loudest contiguous `seconds` of x (so screening renders use the busiest part of the riff)."""
    n = int(seconds * sr)
    if n >= len(x):
        return x
    hop = int(0.25 * sr)
    best, best_e = 0, -1.0
    sq = np.square(x)
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    for a in range(0, len(x) - n, hop):
        e = cs[a + n] - cs[a]
        if e > best_e:
            best, best_e = a, e
    return fade(x[best: best + n], sr)


# ----------------------------------------------------------------------------- synthetic DI
_E_STANDARD = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63]


def _pluck(freq: float, seconds: float, sr: int, rng: np.random.Generator,
           brightness: float = 0.5, decay: float = 0.996) -> np.ndarray:
    """Karplus–Strong string with a pick transient."""
    n = int(seconds * sr)
    period = int(round(sr / freq))
    buf = rng.uniform(-1, 1, period).astype(np.float32)
    # pick position comb + lowpass the excitation for a less glassy attack
    buf = np.convolve(buf, np.ones(3) / 3, mode="same")
    out = np.zeros(n, dtype=np.float32)
    y_prev = 0.0
    a = 0.5 + 0.49 * brightness
    for i in range(n):
        v = buf[i % period]
        out[i] = v
        new = decay * (a * v + (1 - a) * y_prev)
        y_prev = v
        buf[i % period] = new
    return out


def synth_di(sr: int = SR, seconds: float = 8.0, seed: int = 0, tempo_bpm: float = 120.0) -> np.ndarray:
    """A plausible guitar DI: power chords + palm-muted eighths + a short lead line. Mono, ~-18 dBFS."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / tempo_bpm
    out = np.zeros(int(seconds * sr), dtype=np.float32)
    t = 0.0
    roots = [0, 0, 3, 5, 0, 7, 5, 3]  # semitones above E2
    k = 0
    while t < seconds - 0.01:
        root = _E_STANDARD[0] * 2 ** (roots[k % len(roots)] / 12)
        if k % 4 == 3:  # lead fill: single notes an octave up
            for j in range(4):
                f = root * 2 ** ((12 + [0, 3, 5, 7][j]) / 12)
                seg = _pluck(f, beat * 0.5, sr, rng, brightness=0.8) * 0.5
                s = int(t * sr)
                out[s: s + len(seg)] += seg[: len(out) - s]
                t += beat * 0.5
        elif k % 2 == 0:  # ringing power chord (root + fifth + octave)
            seg = np.zeros(int(beat * 2 * sr), dtype=np.float32)
            for mult, g in ((1.0, 0.8), (1.4983, 0.6), (2.0, 0.45)):
                seg += _pluck(root * mult, beat * 2, sr, rng, brightness=0.55, decay=0.9985) * g
            s = int(t * sr)
            out[s: s + len(seg)] += seg[: len(out) - s]
            t += beat * 2
        else:  # palm-muted eighths
            for _ in range(4):
                seg = _pluck(root, beat * 0.5, sr, rng, brightness=0.3, decay=0.985)
                seg += 0.6 * _pluck(root * 1.4983, beat * 0.5, sr, rng, brightness=0.3, decay=0.985)
                s = int(t * sr)
                out[s: s + len(seg)] += seg[: len(out) - s]
                t += beat * 0.5
        k += 1
    # pickup-ish: gentle lowpass + light body resonance
    from scipy.signal import butter, sosfilt
    sos = butter(2, 5500, btype="low", fs=sr, output="sos")
    out = sosfilt(sos, out).astype(np.float32)
    return normalize_active_rms(out, sr, -18.0)
