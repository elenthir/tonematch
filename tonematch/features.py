"""Performance-agnostic tone descriptors and the distance between two of them.

The target (a guitar stem from a record) and the candidate (your DI through a plugin chain) are
different performances, so the loss only compares *distributional* properties:

* ``ltas``   — long-term average log-mel spectrum, level-removed (EQ, cab, mic, voicing)
* ``pct``    — per-band 10th / 90th percentile spectra (how the spectrum breathes with dynamics)
* ``stats``  — quantiles of frame-wise spectral centroid / flatness / rolloff / HF ratio
                (saturation amount, fizz, harmonic density)
* ``dyn``    — quantiles of the short-term RMS envelope (compression / sustain from gain)
* ``mfcc``   — mean + std of MFCCs (compact timbre)

Optionally ``aligned`` — if the DI is the same riff as the target, a multi-resolution STFT
distance after cross-correlation alignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import librosa

from .audio import frame_rms_db, normalize_active_rms

N_FFT = 2048
HOP = 512
N_MELS = 96
FMIN = 40.0
FMAX = 14000.0
GATE_DB = 40.0
QUANTILES = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

DEFAULT_WEIGHTS: Dict[str, float] = {
    "ltas": 1.0,
    "pct": 0.5,
    "stats": 0.6,
    "dyn": 0.3,
    "mfcc": 0.25,
    "aligned": 0.0,
}


@dataclass
class ToneFeatures:
    sr: int
    ltas: np.ndarray            # (N_MELS,) dB, level removed
    p10: np.ndarray             # (N_MELS,) dB, same offset as ltas
    p90: np.ndarray
    stats: np.ndarray           # (4, len(QUANTILES)) centroid[oct], flatness[dB], rolloff[oct], hf ratio[dB]
    dyn: np.ndarray             # (len(QUANTILES)+1,) env quantiles rel. to median + crest factor
    mfcc: np.ndarray            # (2, n_mfcc-1) mean, std
    level_db: float
    signal: Optional[np.ndarray] = field(default=None, repr=False)  # kept for aligned mode

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("signal")
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


_mel_cache: Dict[Tuple[int, int, int], np.ndarray] = {}


def _mel_fb(sr: int) -> np.ndarray:
    key = (sr, N_FFT, N_MELS)
    if key not in _mel_cache:
        _mel_cache[key] = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    return _mel_cache[key]


def extract(x: np.ndarray, sr: int, keep_signal: bool = False) -> ToneFeatures:
    x = normalize_active_rms(x.astype(np.float32), sr, -18.0)
    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window="hann")) ** 2  # (F, T)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    frame_energy = S.sum(axis=0) + 1e-12
    frame_db = 10 * np.log10(frame_energy)
    active = frame_db > (frame_db.max() - GATE_DB)
    if active.sum() < 4:
        active = np.ones_like(active, dtype=bool)
    Sa = S[:, active]

    mel = _mel_fb(sr) @ Sa
    mel_db = 10 * np.log10(mel + 1e-10)                       # (N_MELS, T)
    ltas = mel_db.mean(axis=1)
    offset = ltas.mean()
    ltas = ltas - offset
    p10 = np.percentile(mel_db, 10, axis=1) - offset
    p90 = np.percentile(mel_db, 90, axis=1) - offset

    # frame-wise scalar descriptors
    centroid = (freqs[:, None] * Sa).sum(axis=0) / (Sa.sum(axis=0) + 1e-12)
    centroid_oct = np.log2(np.maximum(centroid, 20.0))
    band = (freqs >= FMIN) & (freqs <= FMAX)
    Sb = Sa[band] + 1e-12
    flatness_db = 10 * np.log10(np.exp(np.mean(np.log(Sb), axis=0)) / np.mean(Sb, axis=0))
    cum = np.cumsum(Sa, axis=0)
    roll_idx = np.argmax(cum >= 0.85 * cum[-1], axis=0)
    rolloff_oct = np.log2(np.maximum(freqs[roll_idx], 20.0))
    hf = Sa[freqs >= 4000].sum(axis=0)
    hf_ratio_db = 10 * np.log10((hf + 1e-12) / (Sa.sum(axis=0) + 1e-12))
    stats = np.stack([
        np.quantile(centroid_oct, QUANTILES),
        np.quantile(flatness_db, QUANTILES),
        np.quantile(rolloff_oct, QUANTILES),
        np.quantile(hf_ratio_db, QUANTILES),
    ])

    env = frame_rms_db(x, sr, 0.05)
    env = env[env > env.max() - GATE_DB]
    q = np.quantile(env, QUANTILES)
    peak_db = 20 * np.log10(np.max(np.abs(x)) + 1e-9)
    crest = peak_db - q[2]
    dyn = np.concatenate([q - q[2], [crest]])

    mf = librosa.feature.mfcc(S=librosa.power_to_db(mel + 1e-10), n_mfcc=20)[1:]
    mfcc = np.stack([mf.mean(axis=1), mf.std(axis=1)])

    return ToneFeatures(sr=sr, ltas=ltas, p10=p10, p90=p90, stats=stats, dyn=dyn, mfcc=mfcc,
                        level_db=float(offset), signal=x if keep_signal else None)


# scale of each stats row so that a "typical" difference is ~1 (octaves, dB, octaves, dB)
_STATS_SCALE = np.array([0.5, 3.0, 0.5, 3.0])[:, None]


def distance(a: ToneFeatures, b: ToneFeatures, weights: Optional[Dict[str, float]] = None,
             band: Tuple[float, float] = (60.0, 12000.0)) -> Tuple[float, Dict[str, float]]:
    """Weighted distance in roughly dB-like units. Lower is better. Returns (total, per-term)."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    mel_f = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    m = (mel_f >= band[0]) & (mel_f <= band[1])
    terms = {
        "ltas": float(np.mean(np.abs(a.ltas[m] - b.ltas[m]))),
        "pct": float(0.5 * (np.mean(np.abs(a.p10[m] - b.p10[m])) + np.mean(np.abs(a.p90[m] - b.p90[m])))),
        "stats": float(np.mean(np.abs(a.stats - b.stats) / _STATS_SCALE)),
        "dyn": float(np.mean(np.abs(a.dyn - b.dyn))),
        "mfcc": float(np.mean(np.abs(a.mfcc - b.mfcc)) / 2.0),
    }
    if w.get("aligned", 0) > 0 and a.signal is not None and b.signal is not None:
        terms["aligned"] = aligned_distance(a.signal, b.signal, a.sr)
    total = float(sum(w.get(k, 0.0) * v for k, v in terms.items()))
    return total, terms


def align_lag(y: np.ndarray, target: np.ndarray, sr: int, max_lag_s: float = 2.0) -> int:
    """Lag (samples) that best aligns `y` to `target` (y[lag:] ~ target), by normalised
    cross-correlation of 10 ms RMS envelopes."""
    hop = int(0.01 * sr)
    ey = frame_rms_db(y, sr, 0.01)
    et = frame_rms_db(target, sr, 0.01)
    ey = (ey - ey.mean()) / (ey.std() + 1e-9)
    et = (et - et.mean()) / (et.std() + 1e-9)
    n = min(len(ey), len(et))
    ey, et = ey[:n], et[:n]
    max_lag = min(n - 2, int(max_lag_s / 0.01))
    best, best_c = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = ey[lag:], et[: n - lag]
        else:
            a, b = ey[: n + lag], et[-lag:]
        c = float(np.dot(a, b)) / max(1, len(a))
        if c > best_c:
            best, best_c = lag, c
    return best * hop


def aligned_distance(y: np.ndarray, target: np.ndarray, sr: int) -> float:
    """Multi-resolution log-STFT L1 after alignment (same riff played on both sides)."""
    lag = align_lag(y, target, sr)
    if lag >= 0:
        y2, t2 = y[lag:], target
    else:
        y2, t2 = y, target[-lag:]
    n = min(len(y2), len(t2))
    y2, t2 = y2[:n], t2[:n]
    tot = 0.0
    for n_fft in (512, 2048):
        A = np.log(np.abs(librosa.stft(y2, n_fft=n_fft, hop_length=n_fft // 4)) + 1e-5)
        B = np.log(np.abs(librosa.stft(t2, n_fft=n_fft, hop_length=n_fft // 4)) + 1e-5)
        tot += float(np.mean(np.abs(A - B)))
    return tot / 2.0 * 8.686  # nepers -> dB


@dataclass
class TargetProfile:
    """Coarse description of the target used to pick starting points."""
    gain_class: str          # clean | crunch | high
    brightness: float        # centroid median in octaves (log2 Hz)
    hf_ratio_db: float
    flatness_db: float
    dynamics_db: float       # p90 - p10 of the envelope

    @staticmethod
    def from_features(f: ToneFeatures) -> "TargetProfile":
        flat = float(f.stats[1, 2])
        hf = float(f.stats[3, 2])
        dyn = float(f.dyn[4] - f.dyn[0])
        # saturated guitars: flatter spectra, more HF, squashed envelope
        score = (flat + 35) / 10 + (hf + 20) / 10 + (8 - dyn) / 6
        if score < 1.0:
            cls = "clean"
        elif score < 2.2:
            cls = "crunch"
        else:
            cls = "high"
        return TargetProfile(gain_class=cls, brightness=float(f.stats[0, 2]), hf_ratio_db=hf,
                             flatness_db=flat, dynamics_db=dyn)
