"""Run Neural Amp Modeler (.nam) captures natively (torch), no plugin needed.

Supports the classic export format produced by the official trainer (WaveNet "standard / lite /
feather / nano" and LSTM — what nearly every TONE3000 capture is), and, when the
``neural-amp-modeler`` package is installed, the newer A2 format through its own loader.

The classic WaveNet is a port of the MIT-licensed reference implementation
(Steven Atkinson, https://github.com/sdatkinson/neural-amp-modeler); the LSTM follows
NeuralAmpModelerCore's weight layout. Both are checked against reference outputs in the tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise SystemExit("NAM captures need torch:  uv pip install torch   (or pip install 'tonematch[nam]')") from e
    torch.set_grad_enabled(False)
    return torch


# ----------------------------------------------------------------------------- classic WaveNet
def _build_legacy_wavenet(config: Dict[str, Any], weights: np.ndarray):
    torch = _torch()
    nn = torch.nn

    class Conv1d(nn.Conv1d):
        def load(self, w, i):
            n = self.weight.numel()
            self.weight.data = torch.as_tensor(w[i: i + n], dtype=torch.float32).reshape(self.weight.shape)
            i += n
            if self.bias is not None:
                n = self.bias.numel()
                self.bias.data = torch.as_tensor(w[i: i + n], dtype=torch.float32).reshape(self.bias.shape)
                i += n
            return i

    class Layer(nn.Module):
        def __init__(self, condition_size, channels, kernel_size, dilation, activation, gated):
            super().__init__()
            mid = 2 * channels if gated else channels
            self.conv = Conv1d(channels, mid, kernel_size, dilation=dilation)
            self.input_mixer = Conv1d(condition_size, mid, 1, bias=False)
            self.act = getattr(nn, activation)()
            self.one_by_one = Conv1d(channels, channels, 1)
            self.gated, self.channels = gated, channels

        def load(self, w, i):
            i = self.conv.load(w, i)
            i = self.input_mixer.load(w, i)
            return self.one_by_one.load(w, i)

        def forward(self, x, h, out_length):
            zconv = self.conv(x)
            z1 = zconv + self.input_mixer(h)[:, :, -zconv.shape[2]:]
            if self.gated:
                post = self.act(z1[:, : self.channels]) * torch.sigmoid(z1[:, self.channels:])
            else:
                post = self.act(z1)
            return x[:, :, -post.shape[2]:] + self.one_by_one(post), post[:, :, -out_length:]

    class LayerArray(nn.Module):
        def __init__(self, input_size, condition_size, head_size, channels, kernel_size, dilations,
                     activation="Tanh", gated=True, head_bias=True):
            super().__init__()
            self.rechannel = Conv1d(input_size, channels, 1, bias=False)
            self.layers = nn.ModuleList([Layer(condition_size, channels, kernel_size, d, activation, gated) for d in dilations])
            self.head_rechannel = Conv1d(channels, head_size, 1, bias=head_bias)
            self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)

        def load(self, w, i):
            i = self.rechannel.load(w, i)
            for l in self.layers:
                i = l.load(w, i)
            return self.head_rechannel.load(w, i)

        def forward(self, x, c, head_input=None):
            out_length = x.shape[2] - (self.receptive_field - 1)
            x = self.rechannel(x)
            for l in self.layers:
                x, head_term = l(x, c, out_length)
                head_input = head_term if head_input is None else head_input[:, :, -out_length:] + head_term
            return self.head_rechannel(head_input), x

    class Head(nn.Module):
        def __init__(self, in_channels, channels, activation, num_layers, out_channels):
            super().__init__()
            self.blocks = nn.ModuleList()
            cin = in_channels
            for i in range(num_layers):
                cout = channels if i != num_layers - 1 else out_channels
                self.blocks.append(nn.Sequential(getattr(nn, activation)(), Conv1d(cin, cout, 1)))
                cin = channels

        def load(self, w, i):
            for b in self.blocks:
                i = b[1].load(w, i)
            return i

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    class WaveNet(nn.Module):
        def __init__(self, layers, head, head_scale):
            super().__init__()
            self.arrays = nn.ModuleList([LayerArray(**lc) for lc in layers])
            self.head = None if head is None else Head(**head)
            self.head_scale = head_scale
            self.receptive_field = 1 + sum(a.receptive_field - 1 for a in self.arrays)

        def load(self, w):
            i = 0
            for a in self.arrays:
                i = a.load(w, i)
            if self.head is not None:
                i = self.head.load(w, i)
            self.head_scale = float(w[i])       # the exporter appends head_scale as the last weight
            i += 1
            if i != len(w):
                raise ValueError(f"weight count mismatch: used {i} of {len(w)}")

        def forward(self, x):  # x: (N, L) -> (N, L)
            x = torch.cat([torch.zeros((x.shape[0], self.receptive_field - 1)), x], dim=1)[:, None]
            y, head_input = x, None
            for a in self.arrays:
                head_input, y = a(y, x, head_input)
            head_input = self.head_scale * head_input
            out = head_input if self.head is None else self.head(head_input)
            return out[:, 0]

    net = WaveNet(config["layers"], config.get("head"), config.get("head_scale", 1.0)).eval()
    net.load(np.asarray(weights, dtype=np.float32))
    return net


# ----------------------------------------------------------------------------- classic LSTM
def _build_legacy_lstm(config: Dict[str, Any], weights: np.ndarray):
    """NeuralAmpModelerCore layout: per layer W(4H x (in+H)) row-major, b(4H), h0(H), c0(H);
    then head weight (1 x H) and head bias (1). Gate order i, f, g, o — same as torch."""
    torch = _torch()
    nn = torch.nn
    H, L, in0 = int(config["hidden_size"]), int(config.get("num_layers", 1)), int(config.get("input_size", 1))
    w = np.asarray(weights, dtype=np.float32)
    i = 0
    cells, h0s, c0s = [], [], []
    for l in range(L):
        inp = in0 if l == 0 else H
        W = w[i: i + 4 * H * (inp + H)].reshape(4 * H, inp + H)
        i += 4 * H * (inp + H)
        b = w[i: i + 4 * H]
        i += 4 * H
        h0 = w[i: i + H]
        i += H
        c0 = w[i: i + H]
        i += H
        cell = nn.LSTM(inp, H, batch_first=True)
        cell.weight_ih_l0.data = torch.as_tensor(W[:, :inp].copy())
        cell.weight_hh_l0.data = torch.as_tensor(W[:, inp:].copy())
        cell.bias_ih_l0.data = torch.as_tensor(b.copy())
        cell.bias_hh_l0.data = torch.zeros(4 * H)
        cells.append(cell.eval())
        h0s.append(torch.as_tensor(h0.copy()))
        c0s.append(torch.as_tensor(c0.copy()))
    head_w = torch.as_tensor(w[i: i + H].copy())
    i += H
    head_b = float(w[i])
    i += 1
    if i != len(w):
        raise ValueError(f"weight count mismatch: used {i} of {len(w)}")

    class LSTMNet(nn.Module):
        receptive_field = 1

        def __init__(self):
            super().__init__()
            self.cells = nn.ModuleList(cells)

        def forward(self, x):  # (N, L) -> (N, L)
            h = x[:, :, None]
            for cell, h0, c0 in zip(self.cells, h0s, c0s):
                n = h.shape[0]
                h, _ = cell(h, (h0[None, None].expand(1, n, H).contiguous(), c0[None, None].expand(1, n, H).contiguous()))
            return h @ head_w + head_b

    return LSTMNet().eval()


# ----------------------------------------------------------------------------- public API
@dataclass
class NamModel:
    path: str
    sample_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    architecture: str = ""
    _net: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.metadata.get("name") or Path(self.path).stem

    @property
    def gear_type(self) -> str:
        return normalize_gear(self.metadata.get("gear_type") or self.metadata.get("gear") or "")

    @property
    def needs_cab(self) -> bool:
        return self.gear_type in ("amp", "preamp", "pedal", "")

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        torch = _torch()
        y = x.astype(np.float32)
        msr = int(round(self.sample_rate)) if self.sample_rate else sr
        if msr != sr:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=msr).astype(np.float32)
        out = self._net(torch.as_tensor(y)[None])[0].numpy().astype(np.float32)
        if msr != sr:
            import librosa
            out = librosa.resample(out, orig_sr=msr, target_sr=sr).astype(np.float32)
        n = len(x)
        if len(out) < n:
            out = np.concatenate([out, np.zeros(n - len(out), dtype=np.float32)])
        return out[:n]


def normalize_gear(g: str) -> str:
    """amp | amp-cab | full-rig | pedal | preamp | outboard | cab | '' """
    g = (g or "").lower().strip().replace("_", "-").replace(" ", "-")
    return {"fullrig": "full-rig", "amp+cab": "amp-cab", "ampcab": "amp-cab", "amp-and-cab": "amp-cab",
            "rig": "full-rig"}.get(g, g)


def is_legacy_config(nam: Dict[str, Any]) -> bool:
    arch = nam.get("architecture")
    if arch == "LSTM":
        return True
    if arch == "WaveNet":
        layers = nam.get("config", {}).get("layers") or []
        return bool(layers) and "head_size" in layers[0]
    return False


def sidecar_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".json")


def read_metadata(path: str | Path) -> Dict[str, Any]:
    """Metadata without building the model (fast): the .nam's own block, merged with a sidecar
    ``<file>.nam.json`` written by the TONE3000 downloader (tone title, gear, make, tags…)."""
    path = Path(path)
    meta: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            head = fp.read(65536)      # metadata sits near the top; don't parse megabytes of weights
        start = head.find('"metadata"')
        if start >= 0:
            depth, i = 0, head.index("{", start)
            for j in range(i, len(head)):
                if head[j] == "{":
                    depth += 1
                elif head[j] == "}":
                    depth -= 1
                    if depth == 0:
                        meta = json.loads(head[i: j + 1])
                        break
    except Exception:
        pass
    side = sidecar_path(path)
    if side.exists():
        try:
            meta.update(json.loads(side.read_text(encoding="utf-8")))
        except Exception:
            pass
    return meta


def load_nam(path: str | Path) -> NamModel:
    path = str(path)
    with open(path, "r", encoding="utf-8") as fp:
        nam = json.load(fp)
    meta = dict(nam.get("metadata") or {})
    side = sidecar_path(path)
    if side.exists():
        try:
            meta.update(json.loads(side.read_text(encoding="utf-8")))
        except Exception:
            pass
    sr = float(nam.get("sample_rate") or 48000.0)
    arch = nam.get("architecture", "")
    if is_legacy_config(nam):
        net = _build_legacy_wavenet(nam["config"], nam["weights"]) if arch == "WaveNet" \
            else _build_legacy_lstm(nam["config"], nam["weights"])
    else:
        try:
            import sys
            from unittest.mock import MagicMock
            for mod in ("tkinter", "tkinter.filedialog", "tkinter.messagebox", "tkinter.ttk"):
                sys.modules.setdefault(mod, MagicMock())
            from nam.models import init_from_nam
        except ImportError as e:
            raise SystemExit(f"{path}: new-format (A2) capture — needs the neural-amp-modeler package: "
                             "uv pip install neural-amp-modeler") from e
        torch = _torch()
        inner = init_from_nam(nam).eval()

        class Wrap(torch.nn.Module):
            def forward(self, x):
                return inner(x, pad_start=True)
        net = Wrap().eval()
    return NamModel(path=path, sample_rate=sr, metadata=meta, architecture=arch, _net=net)
