"""Mock 'plugins' built from pedalboard's built-in effects.

They expose the same surface as a scanned VST3 (a PluginInfo with normalised parameters) so the
whole search / export pipeline can be exercised — and tested — on a machine without the real
plugins. Parameters are deliberately amp-like so the knob map tags them like a real amp sim.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from pedalboard import (Pedalboard, Gain, Distortion, LowShelfFilter, PeakFilter, HighShelfFilter,
                        LowpassFilter, HighpassFilter, Compressor, Reverb, Delay, Clipping)

from .catalog import ParamInfo, PluginInfo


def _lerp(lo: float, hi: float, v: float) -> float:
    return lo + (hi - lo) * float(v)


class MockPlugin:
    """Base: `info` (PluginInfo), `set_raw`, `get_raw`, `board` (pedalboard chain), `reset`."""
    info: PluginInfo

    def __init__(self):
        self.values: Dict[str, float] = {p.name: p.default for p in self.info.params}
        self._build()

    def _build(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def set_raw(self, name: str, v: float) -> None:
        self.values[name] = float(np.clip(v, 0.0, 1.0))

    def get_raw(self, name: str) -> float:
        return self.values[name]

    def reset(self) -> None:
        self.board.reset()

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        self._apply()
        self.board.reset()
        return self.board(x, sr)

    def _apply(self):  # pragma: no cover - overridden
        raise NotImplementedError


_CABS = ["4x12 V30", "2x12 Greenback", "1x12 Jensen", "4x12 T75"]
_MICS = ["SM57", "R121", "MD421"]


class MockAmp(MockPlugin):
    """Gain, 3-band tone stack, presence, cab + mic choice, bright switch, master (excluded)."""
    info = PluginInfo(
        id="mock_amp", name="Mock Amp Suite", path="mock://amp", vendor="tonematch", format="mock",
        params=[
            ParamInfo("Amp Gain", "amp_gain", "float", 0.5),
            ParamInfo("Amp Bass", "amp_bass", "float", 0.5),
            ParamInfo("Amp Middle", "amp_middle", "float", 0.5),
            ParamInfo("Amp Treble", "amp_treble", "float", 0.5),
            ParamInfo("Amp Presence", "amp_presence", "float", 0.5),
            ParamInfo("Amp Bright", "amp_bright", "bool", 0.0, n_values=2, values=["Off", "On"], raw_centers=[0.0, 1.0]),
            ParamInfo("Cab Select", "cab_select", "str", 0.0, n_values=len(_CABS), values=_CABS,
                      raw_centers=list(np.linspace(0, 1, len(_CABS)))),
            ParamInfo("Mic Select", "mic_select", "str", 0.0, n_values=len(_MICS), values=_MICS,
                      raw_centers=list(np.linspace(0, 1, len(_MICS)))),
            ParamInfo("Reverb Mix", "reverb_mix", "float", 0.0),
            ParamInfo("Output Level", "output_level", "float", 0.5),
        ])

    def _build(self):
        self.pre_hp = HighpassFilter(80)
        self.bright = HighShelfFilter(2500, 0.0)
        self.pre = Gain(0)
        self.dist = Distortion(0)
        self.bass = LowShelfFilter(180, 0.0)
        self.mid = PeakFilter(700, 0.0, 0.8)
        self.treble = HighShelfFilter(2200, 0.0)
        self.pres = PeakFilter(4000, 0.0, 1.0)
        self.cab_lp = LowpassFilter(5000)
        self.cab_hp = HighpassFilter(90)
        self.cab_bump = PeakFilter(120, 0.0, 1.2)
        self.mic = PeakFilter(3000, 0.0, 1.5)
        self.verb = Reverb(room_size=0.4, wet_level=0.0, dry_level=1.0)
        self.out = Gain(0)
        self.board = Pedalboard([self.pre_hp, self.bright, self.pre, self.dist, self.bass, self.mid,
                                 self.treble, self.pres, self.cab_hp, self.cab_lp, self.cab_bump,
                                 self.mic, self.verb, self.out])

    def _apply(self):
        v = self.values
        self.bright.gain_db = 6.0 if v["Amp Bright"] >= 0.5 else 0.0
        self.pre.gain_db = _lerp(-6, 18, v["Amp Gain"])
        self.dist.drive_db = _lerp(0, 40, v["Amp Gain"])
        self.bass.gain_db = _lerp(-12, 12, v["Amp Bass"])
        self.mid.gain_db = _lerp(-12, 10, v["Amp Middle"])
        self.treble.gain_db = _lerp(-12, 10, v["Amp Treble"])
        self.pres.gain_db = _lerp(-8, 8, v["Amp Presence"])
        cab = int(round(v["Cab Select"] * (len(_CABS) - 1)))
        self.cab_lp.cutoff_frequency_hz = [5200, 4300, 6500, 4800][cab]
        self.cab_hp.cutoff_frequency_hz = [90, 110, 140, 80][cab]
        self.cab_bump.gain_db = [4.0, 2.0, 0.0, 6.0][cab]
        mic = int(round(v["Mic Select"] * (len(_MICS) - 1)))
        self.mic.cutoff_frequency_hz = [3500, 1800, 2500][mic]
        self.mic.gain_db = [5.0, -3.0, 2.0][mic]
        self.verb.wet_level = _lerp(0, 0.5, v["Reverb Mix"])
        self.verb.dry_level = 1.0
        self.out.gain_db = _lerp(-20, 0, v["Output Level"]) - 0.5 * _lerp(0, 40, v["Amp Gain"])


class MockDrive(MockPlugin):
    info = PluginInfo(
        id="mock_drive", name="Mock Overdrive", path="mock://drive", vendor="tonematch", format="mock",
        params=[
            ParamInfo("Active", "active", "bool", 1.0, n_values=2, values=["Off", "On"], raw_centers=[0.0, 1.0]),
            ParamInfo("Drive", "drive", "float", 0.3),
            ParamInfo("Tone", "tone", "float", 0.5),
            ParamInfo("Level", "level", "float", 0.5),
        ])

    def _build(self):
        self.hp = HighpassFilter(300)
        self.dist = Distortion(0)
        self.tone = LowpassFilter(3000)
        self.level = Gain(0)
        self.board = Pedalboard([self.hp, self.dist, self.tone, self.level])

    def _apply(self):
        v = self.values
        on = v["Active"] >= 0.5
        self.hp.cutoff_frequency_hz = 300 if on else 10
        self.dist.drive_db = _lerp(0, 30, v["Drive"]) if on else 0.0
        self.tone.cutoff_frequency_hz = _lerp(1500, 8000, v["Tone"]) if on else 20000
        self.level.gain_db = (_lerp(-12, 6, v["Level"]) - 0.4 * _lerp(0, 30, v["Drive"])) if on else 0.0


class MockEQ(MockPlugin):
    info = PluginInfo(
        id="mock_eq", name="Mock EQ", path="mock://eq", vendor="tonematch", format="mock",
        params=[
            ParamInfo("Low Gain", "low_gain", "float", 0.5),
            ParamInfo("Mid Gain", "mid_gain", "float", 0.5),
            ParamInfo("Mid Freq", "mid_freq", "float", 0.5),
            ParamInfo("High Gain", "high_gain", "float", 0.5),
        ])

    def _build(self):
        self.lo = LowShelfFilter(150, 0.0)
        self.mid = PeakFilter(800, 0.0, 1.0)
        self.hi = HighShelfFilter(3500, 0.0)
        self.board = Pedalboard([self.lo, self.mid, self.hi])

    def _apply(self):
        v = self.values
        self.lo.gain_db = _lerp(-10, 10, v["Low Gain"])
        self.mid.gain_db = _lerp(-10, 10, v["Mid Gain"])
        self.mid.cutoff_frequency_hz = 200 * 2 ** (_lerp(0, 4, v["Mid Freq"]))
        self.hi.gain_db = _lerp(-10, 10, v["High Gain"])


class MockReverb(MockPlugin):
    info = PluginInfo(
        id="mock_reverb", name="Mock Reverb", path="mock://reverb", vendor="tonematch", format="mock",
        params=[ParamInfo("Reverb Size", "reverb_size", "float", 0.5),
                ParamInfo("Reverb Mix", "reverb_mix", "float", 0.2)])

    def _build(self):
        self.verb = Reverb(room_size=0.5, wet_level=0.2, dry_level=1.0)
        self.board = Pedalboard([self.verb])

    def _apply(self):
        self.verb.room_size = self.values["Reverb Size"]
        self.verb.wet_level = _lerp(0, 0.6, self.values["Reverb Mix"])


MOCK_CLASSES = {c.info.id: c for c in (MockAmp, MockDrive, MockEQ, MockReverb)}


def mock_catalog():
    from .catalog import Catalog
    from .knobmap import KnobMap
    import copy
    cat = Catalog()
    km = KnobMap.load()
    for c in MOCK_CLASSES.values():
        info = copy.deepcopy(c.info)
        cat.plugins[info.id] = info
    cat.apply_knobmap(km)
    # roles the name heuristics cannot infer for the mocks
    cat.plugins["mock_amp"].role = "amp_suite"
    cat.plugins["mock_drive"].role = "drive"
    cat.plugins["mock_eq"].role = "eq"
    cat.plugins["mock_reverb"].role = "reverb"
    return cat
