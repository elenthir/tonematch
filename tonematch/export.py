"""Write a run's results: JSON, audio, REAPER script, VST3 presets, report."""
from __future__ import annotations

import json
import re
import shutil
import time
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import save_audio
from .chain import ChainSpec, Renderer
from .search import Evaluation, SearchResult


def _lua_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "plugin"


def write_reaper_script(run_dir: Path, renderer: Renderer, spec: ChainSpec) -> Path:
    rdir = run_dir / "reaper"
    rdir.mkdir(parents=True, exist_ok=True)
    renderer.apply(spec)
    if (rdir / "nam").exists():          # a checkpoint of an earlier chain may have left other files
        shutil.rmtree(rdir / "nam")
    lua_slots = []
    nam_player = next((p for p in renderer.catalog.plugins.values() if p.role == "nam_player" and not p.error), None)
    for i, slot in enumerate(spec.slots):
        info = renderer.catalog.plugins[slot.plugin_id]
        h = renderer.handle(slot.plugin_id)
        if info.format in ("nam", "ir"):
            # the capture / IR is file state, not a knob: copy the file next to the script and tell the
            # user (and the ReaScript console) what to load into the NAM plugin
            ndir = rdir / "nam"
            ndir.mkdir(exist_ok=True)
            if info.format == "nam":
                src = Path(info.path)
                dst = ndir / f"{i + 1:02d}_{_safe(src.stem)}.nam"
                shutil.copy(src, dst)
                db = h.gain_db()
                lua_slots.append(
                    "  { name = %s, vendor = %s, preset = \"\", load_file = %s, note = %s,\n    params = { {\"Input\", %.6f} } }" % (
                        _lua_str(nam_player.name if nam_player else "Neural Amp Modeler"),
                        _lua_str(nam_player.vendor if nam_player else ""), _lua_str(str(dst.resolve())),
                        _lua_str(f"load this capture in the NAM plugin, input {db:+.1f} dB"),
                        min(1.0, max(0.0, (db + 20.0) / 40.0))))
            else:
                src = Path(h.current_path())
                dst = ndir / f"{i + 1:02d}_{_safe(src.stem)}{src.suffix}"
                shutil.copy(src, dst)
                lua_slots.append(
                    "  { name = \"\", vendor = \"\", preset = \"\", load_file = %s, note = %s, params = {} }" % (
                        _lua_str(str(dst.resolve())), _lua_str("load this IR in the NAM plugin's IR slot (or any IR loader)")))
            continue
        preset_path = ""
        if hasattr(h, "preset_data"):
            data = h.preset_data()
            if data:
                preset_path = str((rdir / f"{i + 1:02d}_{_safe(info.name)}.vstpreset").resolve())
                Path(preset_path).write_bytes(data)
        params = []
        for p in info.params:
            if p.kind == "excluded":
                continue
            v = slot.params.get(p.name, p.fixed_value if p.kind == "fixed" else None)
            if v is None:   # untouched secondary knob: export the plugin default so the chain is reproducible
                v = p.default
            params.append(f"{{{_lua_str(p.name)}, {float(v):.6f}}}")
        lua_slots.append(
            "  { name = %s, vendor = %s, preset = %s,\n    params = {\n      %s\n    } }" % (
                _lua_str(info.name), _lua_str(info.vendor or ""), _lua_str(preset_path), ",\n      ".join(params)))
    template = resources.files("tonematch").joinpath("reaper/apply_tone_template.lua").read_text(encoding="utf-8")
    lua = template.replace("__CHAIN__", "{\n" + ",\n".join(lua_slots) + "\n}")
    out = rdir / "apply_tone.lua"
    out.write_text(lua, encoding="utf-8")
    return out


def write_checkpoint(run_dir: Path, renderer: Renderer, ev: Evaluation, di: np.ndarray, sr: int,
                     target: Optional[np.ndarray] = None, result: Optional[SearchResult] = None,
                     write_audio: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    desc = renderer.describe(ev.spec)
    payload = {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "loss": ev.loss, "terms": ev.terms,
               "stage": ev.stage, "elapsed_s": ev.elapsed, "n_evals": ev.n_evals, "chain": ev.spec.key,
               "spec": ev.spec.to_dict(), "plugins": desc}
    if result is not None:
        payload["search"] = result.to_dict()
    (run_dir / "best.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    write_reaper_script(run_dir, renderer, ev.spec)
    if write_audio:
        y = renderer.render(ev.spec, di, sr)
        save_audio(run_dir / "best.wav", y, sr)
        if target is not None and not (run_dir / "target.wav").exists():
            save_audio(run_dir / "target.wav", target, sr)
        if not (run_dir / "di.wav").exists():
            save_audio(run_dir / "di.wav", di, sr)
    (run_dir / "best.md").write_text(describe_markdown(payload), encoding="utf-8")


def describe_markdown(payload: dict) -> str:
    lines = [f"# tonematch result — loss {payload['loss']:.2f} ({payload['stage']}, "
             f"{payload['elapsed_s'] / 60:.1f} min, {payload['n_evals']} renders)", ""]
    lines.append("Loss terms: " + ", ".join(f"{k} {v:.2f}" for k, v in payload["terms"].items()))
    lines.append("")
    for i, pl in enumerate(payload["plugins"]):
        lines.append(f"## {i + 1}. {pl['name']}")
        lines.append("")
        lines.append("| knob | value | normalised | kind |")
        lines.append("|---|---|---|---|")
        for p in pl["params"]:
            lines.append(f"| {p['name']} | {p['display']} | {p['value']:.3f} | {p['kind']} |")
        lines.append("")
    lines.append("Apply in REAPER: select the guitar track, then run `reaper/apply_tone.lua` "
                 "(Actions → Show action list → New action → Load ReaScript).")
    return "\n".join(lines)
