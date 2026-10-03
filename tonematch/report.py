"""Plots: target vs matched spectrum, loss over time."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import features as F


def write_report(run_dir: Path, target: F.ToneFeatures, best: F.ToneFeatures,
                 di: Optional[F.ToneFeatures] = None, improvements: Optional[list] = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa

    mel_f = librosa.mel_frequencies(n_mels=F.N_MELS, fmin=F.FMIN, fmax=F.FMAX)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1.3]})
    ax = axes[0]
    ax.fill_between(mel_f, target.p10, target.p90, color="#d9534f", alpha=0.15, label="target p10–p90")
    ax.plot(mel_f, target.ltas, color="#d9534f", lw=2, label="target (record)")
    ax.fill_between(mel_f, best.p10, best.p90, color="#1f77b4", alpha=0.15, label="match p10–p90")
    ax.plot(mel_f, best.ltas, color="#1f77b4", lw=2, label="best match (your DI → chain)")
    if di is not None:
        ax.plot(mel_f, di.ltas, color="#888", lw=1, ls="--", label="your DI (unprocessed)")
    ax.set_xscale("log")
    ax.set_xlim(50, 14000)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB (level removed)")
    ax.set_title("Long-term spectrum: target vs match")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    if improvements:
        t = [i["elapsed"] / 60 for i in improvements]
        l = [i["loss"] for i in improvements]
        ax.step(t, l, where="post", color="#1f77b4")
        ax.scatter(t, l, s=10, color="#1f77b4")
        ax.set_xlabel("minutes")
        ax.set_ylabel("best loss")
        ax.set_title("Search progress")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = run_dir / "report.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
