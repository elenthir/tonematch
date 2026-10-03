"""Chain specification + renderer (pedalboard-hosted VST3 / mock plugins)."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from .catalog import Catalog, PluginInfo


@dataclass
class Slot:
    plugin_id: str
    params: Dict[str, float] = field(default_factory=dict)   # param name -> normalised value


@dataclass
class ChainSpec:
    slots: List[Slot]

    @property
    def key(self) -> str:
        return " > ".join(s.plugin_id for s in self.slots)

    @property
    def short_id(self) -> str:
        return hashlib.sha1(self.key.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return {"slots": [asdict(s) for s in self.slots]}

    @staticmethod
    def from_dict(d: dict) -> "ChainSpec":
        return ChainSpec([Slot(s["plugin_id"], dict(s.get("params", {}))) for s in d["slots"]])

    def with_params(self, params: Dict[str, Dict[str, float]]) -> "ChainSpec":
        return ChainSpec([Slot(s.plugin_id, {**s.params, **params.get(s.plugin_id, {})}) for s in self.slots])

    def copy(self) -> "ChainSpec":
        return ChainSpec.from_dict(self.to_dict())


class _Vst3Handle:
    """A loaded external plugin with normalised get/set by *reported* parameter name."""

    def __init__(self, info: PluginInfo):
        from pedalboard import load_plugin
        self.info = info
        name = info.name if "::" in info.id else None
        self.plug = load_plugin(info.path, plugin_name=name) if name else load_plugin(info.path)
        self._by_name = {}
        for py_name, p in self.plug.parameters.items():
            try:
                self._by_name[str(p.name)] = p
            except Exception:
                pass
            self._by_name[py_name] = p

    def set_raw(self, name: str, v: float) -> None:
        p = self._by_name.get(name)
        if p is not None:
            p.raw_value = float(np.clip(v, 0.0, 1.0))

    def get_raw(self, name: str) -> float:
        p = self._by_name.get(name)
        return float(p.raw_value) if p is not None else float("nan")

    def display(self, name: str) -> str:
        p = self._by_name.get(name)
        try:
            return str(p.string_value) if p is not None else ""
        except Exception:
            return ""

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        self.plug.reset()
        y = self.plug(x, sr, reset=True)
        lat = int(getattr(self.plug, "reported_latency_samples", 0) or 0)
        if lat > 0 and lat < len(y):
            y = np.concatenate([y[lat:], np.zeros(lat, dtype=y.dtype)])
        return y

    def preset_data(self) -> Optional[bytes]:
        try:
            return bytes(self.plug.preset_data)
        except Exception:
            return None


class Renderer:
    """Renders a ChainSpec. Plugin instances are cached per thread (plugins are not thread-safe,
    but separate instances can run in parallel since pedalboard releases the GIL while processing)."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self._local = threading.local()
        self.n_renders = 0
        self._lock = threading.Lock()

    def _handles(self) -> Dict[str, object]:
        if not hasattr(self._local, "handles"):
            self._local.handles = {}
        return self._local.handles

    def handle(self, plugin_id: str):
        hs = self._handles()
        if plugin_id not in hs:
            info = self.catalog.plugins[plugin_id]
            if info.format == "mock":
                from .mock import MOCK_CLASSES
                hs[plugin_id] = MOCK_CLASSES[plugin_id]()
            else:
                hs[plugin_id] = _Vst3Handle(info)
        return hs[plugin_id]

    def apply(self, spec: ChainSpec) -> None:
        """Set every parameter of every slot (defaults for unspecified, fixed values honoured)."""
        for slot in spec.slots:
            h = self.handle(slot.plugin_id)
            info = self.catalog.plugins[slot.plugin_id]
            for p in info.params:
                if p.kind == "fixed" and p.fixed_value is not None:
                    h.set_raw(p.name, p.fixed_value)
                elif p.name in slot.params:
                    h.set_raw(p.name, slot.params[p.name])
                elif p.kind != "excluded":
                    h.set_raw(p.name, p.default)

    def render(self, spec: ChainSpec, x: np.ndarray, sr: int) -> np.ndarray:
        self.apply(spec)
        y = x.astype(np.float32)
        for slot in spec.slots:
            y = self.handle(slot.plugin_id).process(y, sr)
            if y.ndim > 1:
                y = y.mean(axis=0) if y.shape[0] <= 2 else y.mean(axis=1)
            y = np.ascontiguousarray(y, dtype=np.float32)
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y)
        with self._lock:
            self.n_renders += 1
        return y

    def describe(self, spec: ChainSpec) -> List[dict]:
        """Human-readable parameter listing of a spec (uses the plugin's own value strings)."""
        self.apply(spec)
        out = []
        for slot in spec.slots:
            h = self.handle(slot.plugin_id)
            info = self.catalog.plugins[slot.plugin_id]
            rows = []
            for p in info.params:
                if p.kind == "excluded":
                    continue
                v = slot.params.get(p.name, p.fixed_value if p.kind == "fixed" else p.default)
                disp = h.display(p.name) if hasattr(h, "display") else (
                    p.values[int(round(v * (p.n_values - 1)))] if p.values and p.n_values else f"{v:.3f}")
                rows.append({"name": p.name, "kind": p.kind, "value": float(v), "display": disp})
            out.append({"plugin_id": slot.plugin_id, "name": info.name, "path": info.path, "params": rows})
        return out
