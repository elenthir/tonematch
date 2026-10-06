"""Plugin catalog: scan VST3/AU plugins with pedalboard, record their parameters, tag them."""
from __future__ import annotations

import json
import re
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .knobmap import KnobMap


@dataclass
class ParamInfo:
    name: str                       # name as the plugin (and REAPER) reports it
    python_name: str                # pedalboard identifier
    type: str                       # float | bool | str
    default: float                  # normalised 0..1
    n_values: Optional[int] = None  # for discrete params
    values: Optional[List[str]] = None       # display strings for discrete params (<= 64 kept)
    raw_centers: Optional[List[float]] = None  # normalised value at the centre of each discrete step
    label: Optional[str] = None
    kind: str = "secondary"         # primary | secondary | ambience | fixed | excluded
    fixed_value: Optional[float] = None
    range: Optional[List[float]] = None       # narrowed normalised search range
    group: Optional[str] = None               # channel this knob belongs to ("Clean", "Lead" …)
    group_selector: Optional[str] = None      # the discrete param that selects the channel
    gates: List[str] = field(default_factory=list)   # on/off switches that must be on for this knob to matter


@dataclass
class PluginInfo:
    id: str                         # stable id (basename of the bundle + plugin name)
    name: str
    path: str
    vendor: str = ""
    category: str = ""
    format: str = "vst3"            # vst3 | au | mock
    role: str = "other"
    params: List[ParamInfo] = field(default_factory=list)
    latency: int = 0
    is_instrument: bool = False
    error: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)   # format-specific (NAM metadata, IR list)

    def param(self, name: str) -> Optional[ParamInfo]:
        for p in self.params:
            if p.name == name or p.python_name == name:
                return p
        return None

    def searchable(self, allow_ambience: bool = False) -> List[ParamInfo]:
        kinds = {"primary", "secondary"} | ({"ambience"} if allow_ambience else set())
        return [p for p in self.params if p.kind in kinds]


@dataclass
class Catalog:
    plugins: Dict[str, PluginInfo] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({k: asdict(v) for k, v in self.plugins.items()}, indent=1), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "Catalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cat = Catalog()
        for k, v in raw.items():
            v["params"] = [ParamInfo(**p) for p in v.get("params", [])]
            cat.plugins[k] = PluginInfo(**v)
        return cat

    def by_role(self, role: str) -> List[PluginInfo]:
        return [p for p in self.plugins.values() if p.role == role and not p.error]

    def find(self, needle: str) -> Optional[PluginInfo]:
        n = needle.lower()
        for p in self.plugins.values():
            if p.id.lower() == n or p.name.lower() == n:
                return p
        for p in self.plugins.values():
            if n in p.name.lower() or n in p.id.lower():
                return p
        return None

    def apply_knobmap(self, km: KnobMap) -> None:
        for p in self.plugins.values():
            tag_plugin(p, km)


TONEMATCH_HOME = Path(os.environ.get("TONEMATCH_HOME", Path.home() / ".tonematch"))
NAM_DIR = TONEMATCH_HOME / "nam"
IR_DIR = TONEMATCH_HOME / "ir"
IR_LOADER_ID = "irloader"


_RIG_RX = re.compile(r"full[ _-]?rig|amp[ _+&-]*cab|\b[124]x1[02]\b|\bsm ?57\b|\br ?121\b|\bmd ?421\b|\bcab(inet)?\b|\bir\b", re.I)


def nam_role(gear: str, hints: str = "") -> str:
    """Which chain slot a capture fills, from its gear type. `hints` (title, file name, tags)
    catches captures filed as plain "amp" on TONE3000 that are really full rigs with the cab."""
    if gear in ("amp-cab", "full-rig", "cab-amp"):
        return "amp_suite"
    if gear in ("pedal", "outboard"):
        return "drive"
    if gear in ("amp", "", "preamp") and _RIG_RX.search(hints or ""):
        return "amp_suite"
    return "amp"                       # amp / preamp / unknown: needs an IR after it


def scan_nam_library(nam_dirs: Optional[List[str | Path]] = None,
                     ir_dirs: Optional[List[str | Path]] = None) -> Dict[str, PluginInfo]:
    """NAM captures (.nam) and impulse responses (.wav) as catalog entries. Cheap: only metadata
    is read here, models are built when first rendered."""
    from .namengine import read_metadata, normalize_gear
    out: Dict[str, PluginInfo] = {}
    for d in [Path(x) for x in (nam_dirs or [NAM_DIR])]:
        if not d.exists():
            continue
        for f in sorted(d.rglob("*.nam")):
            meta = read_metadata(f)
            gear = normalize_gear(str(meta.get("gear_type") or meta.get("gear") or ""))
            title = meta.get("title") or meta.get("name") or f.stem
            bits = [b for b in (meta.get("gear_make"), meta.get("gear_model")) if b and b not in title]
            name = f"{title} ({' '.join(bits)})" if bits else title
            pid = "nam:" + str(f.relative_to(d)).replace("\\", "/")
            hints = " ".join([str(title), f.stem, " ".join(map(str, meta.get("tags") or []))])
            role = nam_role(gear, hints)
            out[pid] = PluginInfo(
                id=pid, name=f"NAM: {name}", path=str(f), vendor=str(meta.get("modeled_by") or meta.get("username") or "NAM"),
                format="nam", role=role,
                params=[ParamInfo("Input Gain", "input_gain", "float", 0.5, label="dB", kind="primary")],
                extra={"gear": "amp-cab" if role == "amp_suite" and gear not in ("amp-cab", "full-rig") else gear,
                       "gear_listed": gear, "tone_type": meta.get("tone_type"), "tags": meta.get("tags") or [],
                       "make": meta.get("make") or meta.get("gear_make"), "tone_url": meta.get("url"),
                       "size": meta.get("size")})
    irs: List[str] = []
    for d in [Path(x) for x in (ir_dirs or [IR_DIR])]:
        if d.exists():
            irs += [str(f) for f in sorted(d.rglob("*.wav")) + sorted(d.rglob("*.aif*"))]
    if irs:
        names = [Path(f).stem[:40] for f in irs]
        n = len(irs)
        centers = [i / max(1, n - 1) for i in range(n)] if n > 1 else [0.0]
        out[IR_LOADER_ID] = PluginInfo(
            id=IR_LOADER_ID, name="IR Loader", path="ir://", vendor="tonematch", format="ir", role="cab",
            params=[ParamInfo("IR", "ir", "str", 0.0, n_values=n, values=names if n <= 64 else None,
                              raw_centers=centers if n <= 64 else None, kind="primary"),
                    ParamInfo("Level", "level", "float", 0.5, kind="excluded")],
            extra={"irs": irs})
    return out


def default_plugin_dirs() -> List[Path]:
    sysname = platform.system()
    home = Path.home()
    if sysname == "Darwin":
        dirs = [Path("/Library/Audio/Plug-Ins/VST3"), home / "Library/Audio/Plug-Ins/VST3",
                Path("/Library/Audio/Plug-Ins/Components"), home / "Library/Audio/Plug-Ins/Components"]
    elif sysname == "Windows":
        dirs = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Common Files/VST3",
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Common/VST3"]
    else:
        dirs = [Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3"), home / ".vst3"]
    return [d for d in dirs if d.exists()]


def find_plugin_files(dirs: Optional[List[str | Path]] = None) -> List[Path]:
    out: List[Path] = []
    for d in dirs or default_plugin_dirs():
        d = Path(d)
        if d.suffix.lower() in (".vst3", ".component"):
            out.append(d)
            continue
        for ext in ("*.vst3", "*.component"):
            # a .vst3 *bundle* is a folder (Contents/x86_64-win/Plugin.vst3 inside it on Windows):
            # keep the outermost match only
            out.extend(sorted(p for p in d.rglob(ext)
                              if not any(q.suffix.lower() in (".vst3", ".component") for q in p.parents)))
    # de-dup nested matches (a .vst3 bundle contains no other bundles)
    seen, uniq = set(), []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    return uniq


def resolve_plugin_path(path: str | Path, system: Optional[str] = None) -> str:
    """A VST3 can be a single file or a bundle folder (``X.vst3/Contents/x86_64-win/X.vst3``).
    pedalboard on Windows / Linux wants the binary inside the bundle; macOS wants the bundle."""
    path = Path(path)
    system = system or platform.system()
    if not path.is_dir() or system == "Darwin":
        return str(path)
    arch_dirs = {"Windows": ("x86_64-win", "arm64-win", "x86-win"),
                 "Linux": ("x86_64-linux", "aarch64-linux")}.get(system, ())
    exts = (".vst3",) if system == "Windows" else (".so", ".vst3")
    contents = path / "Contents"
    for arch in arch_dirs:
        d = contents / arch
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    return str(f)
    return str(path)


# ----------------------------------------------------------------------------- probing
def probe_plugin_file(path: str | Path, timeout: float = 600.0) -> List[PluginInfo]:
    """Load the plugin in a *subprocess* (a crashing plugin must not take the scan down).

    The result travels through a temp file rather than stdout: plugins happily print licence
    banners or debug output to stdout while loading, which would corrupt the JSON."""
    import tempfile
    stem = Path(path).stem
    with tempfile.TemporaryDirectory() as td:
        out_file = Path(td) / "probe.json"
        cmd = [sys.executable, "-m", "tonematch.catalog", "--probe", str(path), "--out", str(out_file)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return [PluginInfo(id=stem, name=stem, path=str(path), error="timeout (a dialog waiting for a click?)")]
        if not out_file.exists():
            tail = " | ".join(((r.stderr or "") + (r.stdout or "")).strip().splitlines()[-3:])
            return [PluginInfo(id=stem, name=stem, path=str(path), error=f"load failed (exit {r.returncode}): {tail}")]
        try:
            raw = json.loads(out_file.read_text(encoding="utf-8"))
        except Exception as e:
            return [PluginInfo(id=stem, name=stem, path=str(path), error=f"unreadable probe result: {e}")]
    infos = []
    for d in raw:
        if d.get("error"):
            infos.append(PluginInfo(id=d.get("id") or stem, name=d.get("name") or stem, path=str(path), error=d["error"]))
            continue
        d["params"] = [ParamInfo(**p) for p in d["params"]]
        infos.append(PluginInfo(**d))
    return infos


def _probe_in_process(path: str) -> List[dict]:
    import pedalboard
    from pedalboard import load_plugin
    try:
        names = pedalboard.VST3Plugin.get_plugin_names_for_file(resolve_plugin_path(path))
    except Exception:
        names = [None]
    out = []
    for name in names or [None]:
        try:
            load_path = resolve_plugin_path(path)
            plug = load_plugin(load_path, plugin_name=name) if name else load_plugin(load_path)
            out.append(asdict(describe_loaded_plugin(plug, path)))
        except Exception as e:
            stem = Path(path).stem
            out.append({"id": f"{stem}::{name}" if name else stem, "name": name or stem, "path": path,
                        "error": f"{type(e).__name__}: {e}"})
    return out


def describe_loaded_plugin(plug, path: str, fmt: str = "vst3") -> PluginInfo:
    params: List[ParamInfo] = []
    for py_name, p in plug.parameters.items():
        try:
            name = str(p.name)          # forwarded to the C++ parameter: the name REAPER shows
        except Exception:
            name = py_name
        ptype = "float" if p.type is float else ("bool" if p.type is bool else "str")
        n_values = None
        values = None
        centers = None
        if ptype != "float" or (p.step_size and p.min_value is not None and p.max_value is not None
                                 and (p.max_value - p.min_value) / p.step_size <= 64):
            rngs = list(p.ranges.items())
            n_values = len(rngs)
            if n_values <= 64:
                values = [str(v) for _, v in rngs]
                centers = [float((a + b) / 2) for (a, b), _ in rngs]
                centers[0] = 0.0
                centers[-1] = 1.0 if n_values > 1 else 0.0
        try:
            default = float(p.raw_value)
        except Exception:
            default = 0.5
        params.append(ParamInfo(name=name, python_name=py_name, type=ptype, default=default,
                                n_values=n_values, values=values, raw_centers=centers, label=p.label))
    pname = getattr(plug, "name", None) or Path(path).stem
    return PluginInfo(id=f"{Path(path).stem}::{pname}" if pname != Path(path).stem else Path(path).stem,
                      name=pname, path=str(path), vendor=getattr(plug, "manufacturer_name", "") or "",
                      category=getattr(plug, "category", "") or "", format=fmt, params=params,
                      latency=int(getattr(plug, "reported_latency_samples", 0) or 0),
                      is_instrument=bool(getattr(plug, "is_instrument", False)))


def tag_plugin(p: PluginInfo, km: KnobMap) -> None:
    p.role = km.role_for(p.name, p.category, p.is_instrument, p.vendor)
    r = km.rule_for(p.name)
    if r and r.vendor and not p.vendor:
        p.vendor = r.vendor
    for prm in p.params:
        kind, fixed, rng = km.classify_param(p.name, prm.name, prm.n_values)
        # discrete params with huge value lists (e.g. 300 IRs) are only searched as primary
        if kind == "secondary" and prm.n_values and prm.n_values > 64:
            kind = "excluded"
        if prm.n_values == 1 and kind != "fixed":      # nothing to choose
            kind = "excluded"
        prm.kind, prm.fixed_value = kind, fixed
        prm.range = list(rng) if rng else None
    assign_channel_groups(p)
    assign_gates(p)


_SWITCH_SUFFIX_RX = re.compile(r"^(.+?)\s+(active|enabled?|on/off|on)$", re.I)      # "Overdrive Active"
_SECTION_RX = re.compile(r"^active\s+(.+?)\s+section$|^(.+?)\s+section\s+active$", re.I)   # "Active EQ Section" / "EQ Section Active"


def assign_gates(p: PluginInfo) -> None:
    """A two-state switch gates the knobs of its section, so knobs behind an OFF switch are not
    searched: "Overdrive Active" → "Overdrive Gain" (name prefix); "Active Pre FX Section" → every
    parameter listed after it up to the next "Active … Section" (sections are contiguous in the
    plugin's parameter order). Switches that are forced on (`fixed`) gate nothing."""
    for prm in p.params:
        prm.gates = []
    live = ("primary", "secondary", "ambience")
    # 1. positional sections
    section = None
    for prm in p.params:
        if prm.n_values == 2 and _SECTION_RX.match(prm.name):
            section = prm if prm.kind in live else None
            continue
        if section is not None and prm.kind != "fixed":
            prm.gates.append(section.name)
    # 2. "<Thing> Active" switches gate "<Thing> …" knobs (gate listed before the section's)
    for sw in p.params:
        if not (sw.n_values == 2 and sw.kind in live):
            continue
        m = _SWITCH_SUFFIX_RX.match(sw.name)
        if not m or _SECTION_RX.match(sw.name):
            continue
        prefix = m.group(1).strip().lower() + " "
        for prm in p.params:
            if prm is not sw and prm.kind != "fixed" and prm.name.lower().startswith(prefix) and sw.name not in prm.gates:
                prm.gates.insert(0, sw.name)


_SELECTOR_RX = re.compile(r"(amp|channel).*(type|select|model)|^channel$|^amp$|^mode$", re.I)


def assign_channel_groups(p: PluginInfo) -> None:
    """Multi-channel amps expose "Clean Gain", "Crunch Gain", "Lead Gain" … and a selector
    ("Amp Type" = Clean / Crunch / Lead). Tag each knob with the channel it belongs to so the
    search only touches the selected channel's knobs."""
    for prm in p.params:
        prm.group = prm.group_selector = None
    for sel in p.params:
        if sel.kind != "primary" or not sel.values or not (2 <= (sel.n_values or 0) <= 12):
            continue
        if not _SELECTOR_RX.search(sel.name):
            continue
        hits: Dict[str, int] = {}
        for v in sel.values:
            prefix = str(v).strip().lower() + " "
            if len(prefix) < 3:
                continue
            for prm in p.params:
                if prm is sel or prm.group is not None:
                    continue
                if prm.name.lower().startswith(prefix):
                    prm.group, prm.group_selector = str(v).strip(), sel.name
                    hits[str(v)] = hits.get(str(v), 0) + 1
        if len(hits) < 2:          # a coincidence, not a channel layout: undo
            for prm in p.params:
                if prm.group_selector == sel.name:
                    prm.group = prm.group_selector = None


def scan(dirs: Optional[List[str | Path]] = None, km: Optional[KnobMap] = None,
         progress=None, existing: Optional[Catalog] = None) -> Catalog:
    km = km or KnobMap.load()
    cat = existing or Catalog()
    files = find_plugin_files(dirs)
    for i, f in enumerate(files):
        if progress:
            progress(i, len(files), f)
        if existing and any(p.path == str(f) and not p.error for p in existing.plugins.values()):
            continue
        for info in probe_plugin_file(f):
            tag_plugin(info, km)
            cat.plugins[info.id] = info
    return cat


if __name__ == "__main__":  # subprocess probe entry point
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # keep the plugin's own chatter away from our result
    sys.stdout = sys.stderr
    result = _probe_in_process(a.probe)
    Path(a.out).write_text(json.dumps(result), encoding="utf-8")
