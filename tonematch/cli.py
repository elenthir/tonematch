"""tonematch command line."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import __version__
from .audio import SR, load_audio, normalize_active_rms, save_audio, synth_di, trim_silence
from .catalog import Catalog, default_plugin_dirs, scan, scan_nam_library, NAM_DIR, IR_DIR
from .chain import ChainSpec, Renderer, Slot
from . import features as F
from .knobmap import KnobMap
from .search import Evaluation, SearchConfig, ToneSearch, propose_chains

DEFAULT_CATALOG = Path(os.environ.get("TONEMATCH_HOME", Path.home() / ".tonematch")) / "catalog.json"


def _log(s: str) -> None:
    print(time.strftime("%H:%M:%S"), s, flush=True)


def load_catalog(args) -> Catalog:
    if getattr(args, "mock", False):
        from .mock import mock_catalog
        return mock_catalog()
    path = Path(args.catalog)
    cat = Catalog.load(path) if path.exists() else Catalog()
    cat.apply_knobmap(KnobMap.load(args.knobs))
    nam_dirs = [NAM_DIR] + [Path(d) for d in (getattr(args, "nam_dir", None) or [])]
    ir_dirs = [IR_DIR] + [Path(d) for d in (getattr(args, "ir_dir", None) or [])]
    cat.plugins.update(scan_nam_library(nam_dirs, ir_dirs))     # cheap, always fresh
    if not cat.plugins:
        raise SystemExit(f"no plugin catalog at {path} and no NAM captures in {NAM_DIR} — run `tonematch scan`, "
                         "`tonematch tone3000 fetch …`, or use --mock to try things out")
    return cat


# ----------------------------------------------------------------------------- commands
def cmd_scan(args) -> None:
    km = KnobMap.load(args.knobs)
    dirs = [Path(p) for p in args.paths] if args.paths else default_plugin_dirs()
    if not dirs:
        raise SystemExit("no plugin folders found — pass them explicitly: tonematch scan /path/to/VST3")
    existing = Catalog.load(args.catalog) if Path(args.catalog).exists() and not args.rescan else None

    def progress(i, n, f):
        _log(f"[{i + 1}/{n}] probing {f.name} …")

    cat = scan(dirs, km, progress, existing)
    cat.save(args.catalog)
    ok = [p for p in cat.plugins.values() if not p.error]
    bad = [p for p in cat.plugins.values() if p.error]
    _log(f"catalog: {len(ok)} plugins usable, {len(bad)} failed → {args.catalog}")
    for p in bad:
        _log(f"   failed: {p.name}: {p.error}")
    _print_catalog(cat)


def _print_catalog(cat: Catalog) -> None:
    from collections import defaultdict
    by_role = defaultdict(list)
    for p in cat.plugins.values():
        if not p.error:
            by_role[p.role].append(p)
    for role in ("amp_suite", "amp", "cab", "drive", "eq", "compressor", "gate", "reverb", "delay"):
        if by_role.get(role):
            print(f"\n{role}:")
            for p in sorted(by_role[role], key=lambda p: p.name):
                if p.format == "nam":
                    ex = p.extra or {}
                    print(f"  {p.name[:40]:40s} {p.vendor[:18]:18s} capture: {ex.get('gear') or '?'} {ex.get('tone_type') or ''} "
                          f"{'#' + ' #'.join(map(str, ex.get('tags')[:4])) if ex.get('tags') else ''}")
                    continue
                if p.format == "ir":
                    print(f"  {p.name:40s} {p.vendor[:18]:18s} {len(p.extra.get('irs', []))} impulse responses")
                    continue
                kinds = {k: sum(1 for q in p.params if q.kind == k) for k in ("primary", "secondary", "ambience", "fixed", "excluded")}
                print(f"  {p.name:40s} {p.vendor[:18]:18s} knobs: " + " ".join(f"{k}={v}" for k, v in kinds.items() if v))
    skipped = [(r, p) for r in ("nam_player", "bass", "instrument", "utility", "modulation", "other") for p in sorted(by_role.get(r, []), key=lambda p: p.name)]
    if skipped:
        print("\nnot used in guitar chains (bass → `run --instrument bass`; misfiled? give it a role in a --knobs YAML):")
        for r, p in skipped:
            why = {"other": "unknown kind (not an amp/cab/drive/EQ by name)",
                   "nam_player": "NAM plugin: used to load the matched capture in REAPER"}.get(r, r)
            print(f"  {p.name:40s} {p.vendor[:18]:18s} {why}")
    failed = [p for p in cat.plugins.values() if p.error]
    if failed:
        print("\nfailed to load:")
        for p in sorted(failed, key=lambda p: p.name):
            print(f"  {p.name:40s} {p.error[:90]}")


def cmd_list(args) -> None:
    cat = load_catalog(args)
    _print_catalog(cat)
    if args.plugin:
        p = cat.find(args.plugin)
        if not p:
            raise SystemExit(f"'{args.plugin}' not in catalog")
        print(f"\n{p.name}  ({p.path})")
        for q in p.params:
            extra = f" [{p and q.n_values} values]" if q.n_values else ""
            print(f"  {q.kind:9s} {q.name}{extra}" + (f" fixed={q.fixed_value}" if q.fixed_value is not None else ""))


def cmd_separate(args) -> None:
    from .separate import download, is_url, separate_guitar
    out = Path(args.out)
    src = args.source
    if is_url(src):
        src = download(src, out)
    stem = separate_guitar(src, out, args.start, args.duration, args.model, args.device, args.stem)
    _log(f"guitar stem → {stem}")


def _prepare_target(args, run_dir: Path) -> np.ndarray:
    tgt = args.target
    from .separate import is_url
    if is_url(tgt) or args.separate:
        from .separate import download, separate_guitar
        sep_dir = run_dir / "separation"
        if is_url(tgt):
            tgt = str(download(tgt, sep_dir))
        tgt = str(separate_guitar(tgt, sep_dir, args.start, args.duration, args.model, args.device))
        x = load_audio(tgt, SR)
    else:
        x = load_audio(tgt, SR, start=args.start, duration=args.duration)
    x = trim_silence(x, SR)
    if len(x) < SR * 2:
        raise SystemExit("target excerpt is shorter than 2 s — pick a longer section")
    return normalize_active_rms(x, SR, -18.0)


def cmd_run(args) -> None:
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, filename=str(run_dir / "run.log"))
    if args.tone3000:
        from .tone3000 import Tone3000
        _log(f"TONE3000: fetching up to {args.tone3000_limit} captures for '{args.tone3000}' …")
        Tone3000().fetch(args.tone3000, limit=args.tone3000_limit, log=_log)
    cat = load_catalog(args)
    renderer = Renderer(cat)

    _log("loading target …")
    target_x = _prepare_target(args, run_dir)
    if args.di:
        di = load_audio(args.di, SR, start=args.di_start, duration=args.di_duration)
    else:
        _log("no --di given: using a synthetic DI (fine for a test, use a real DI of your guitar for real work)")
        di = synth_di(SR, 12.0)
    di = normalize_active_rms(trim_silence(di, SR), SR, -18.0)
    if args.max_di_seconds and len(di) > args.max_di_seconds * SR:
        di = di[: int(args.max_di_seconds * SR)]
    _log(f"target {len(target_x) / SR:.1f}s, DI {len(di) / SR:.1f}s")

    weights = dict(F.DEFAULT_WEIGHTS)
    if args.aligned:
        weights["aligned"] = 0.6
    for kv in args.weight or []:
        k, v = kv.split("=")
        weights[k] = float(v)
    target = F.extract(target_x, SR, keep_signal=args.aligned)
    (run_dir / "target_features.json").write_text(json.dumps(target.to_dict()))

    cfg = SearchConfig(minutes=args.minutes, allow_ambience=args.allow_ambience, keep_chains=args.keep_chains,
                       max_chains=args.max_chains, workers=args.workers, seed=args.seed, weights=weights,
                       screen_seconds=args.screen_seconds, optimize_seconds=args.optimize_seconds,
                       refine_seconds=args.refine_seconds,
                       storage=f"sqlite:///{(run_dir / 'optuna.db').resolve()}" if args.resume else None)
    gain_class = F.TargetProfile.from_features(target).gain_class
    chains = propose_chains(cat, cfg, include=args.include, exclude=args.exclude, pinned=args.chain,
                            instrument=args.instrument, gain_class=gain_class, limit=max(cfg.max_chains, args.max_prescreen))
    _log("candidate chains:\n   " + "\n   ".join(" > ".join(cat.plugins[s.plugin_id].name for s in c.slots) for c in chains))

    from .export import write_checkpoint
    from .report import write_report
    last_ckpt = [0.0]

    def on_improvement(ev: Evaluation) -> None:
        now = time.time()
        heavy = now - last_ckpt[0] > 30
        write_checkpoint(run_dir, renderer, ev, di, SR, target_x, write_audio=heavy)
        if heavy:
            last_ckpt[0] = now
        _log(f"  ★ {ev.loss:.3f}  {ev.spec.key}  [{ev.stage}]")

    search = ToneSearch(cat, renderer, di, SR, target, cfg, on_improvement=on_improvement, on_progress=_log)
    t0 = time.time()
    result = search.run(chains)
    best = result.best
    write_checkpoint(run_dir, renderer, best, di, SR, target_x, result=result, write_audio=True)
    y = load_audio(run_dir / "best.wav", SR)
    write_report(run_dir, target, F.extract(y, SR), F.extract(di, SR), result.improvements)
    _log(f"done in {(time.time() - t0) / 60:.1f} min, {result.n_evals} renders. best loss {best.loss:.3f} with "
         f"{' > '.join(cat.plugins[s.plugin_id].name for s in best.spec.slots)}")
    _log(f"results in {run_dir}/ : best.md (knob values), best.wav, report.png, reaper/apply_tone.lua")


def cmd_export(args) -> None:
    run_dir = Path(args.run_dir)
    payload = json.loads((run_dir / "best.json").read_text())
    cat = load_catalog(args)
    spec = ChainSpec.from_dict(payload["spec"])
    from .export import write_reaper_script
    out = write_reaper_script(run_dir, Renderer(cat), spec)
    _log(f"wrote {out}")


def cmd_selftest(args) -> None:
    """Hide a random tone made with the mock plugins, then try to find it back. No real plugins needed."""
    from .mock import mock_catalog
    rng = np.random.default_rng(args.seed)
    cat = mock_catalog()
    renderer = Renderer(cat)
    hidden = ChainSpec([Slot("mock_drive", {"Active": 1.0, "Drive": float(rng.uniform(0.2, 0.9)),
                                            "Tone": float(rng.uniform()), "Level": 0.5}),
                        Slot("mock_amp", {"Amp Gain": float(rng.uniform(0.2, 0.95)), "Amp Bass": float(rng.uniform()),
                                          "Amp Middle": float(rng.uniform()), "Amp Treble": float(rng.uniform()),
                                          "Amp Presence": float(rng.uniform()), "Amp Bright": float(rng.integers(2)),
                                          "Cab Select": float(rng.integers(4)) / 3, "Mic Select": float(rng.integers(3)) / 2})])
    di = synth_di(SR, 10.0, seed=args.seed + 1)
    other_perf = synth_di(SR, 12.0, seed=args.seed + 2, tempo_bpm=104)
    target_x = renderer.render(hidden, other_perf, SR)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_audio(run_dir / "hidden_target.wav", target_x, SR)
    target = F.extract(target_x, SR)
    cfg = SearchConfig(minutes=args.minutes, keep_chains=2, seed=args.seed, workers=args.workers)
    chains = propose_chains(cat, cfg)
    search = ToneSearch(cat, renderer, di, SR, target, cfg, on_progress=_log)
    base = search.evaluate(ChainSpec([Slot("mock_drive"), Slot("mock_amp")]), 10, "baseline")[0]
    floor = search.evaluate(hidden, 10, "floor")[0]
    search.best = None
    search.improvements = []
    result = search.run(chains)
    from .export import write_checkpoint
    from .report import write_report
    write_checkpoint(run_dir, renderer, result.best, di, SR, target_x, result=result)
    y = load_audio(run_dir / "best.wav", SR)
    write_report(run_dir, target, F.extract(y, SR), F.extract(di, SR), result.improvements)
    print("\nhidden chain:")
    for s in hidden.slots:
        print("  ", s.plugin_id, {k: round(v, 2) for k, v in s.params.items()})
    print("found chain:")
    for s in result.best.spec.slots:
        print("  ", s.plugin_id, {k: round(v, 2) for k, v in s.params.items()})
    print(f"\nloss: defaults {base:.2f} → found {result.best.loss:.2f}   (hidden chain itself on your DI: {floor:.2f})")
    print(f"renders: {result.n_evals} in {result.elapsed / 60:.1f} min. Files in {run_dir}/")
    ok = result.best.loss < 0.5 * base
    print("SELFTEST", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


def cmd_tone3000_login(args) -> None:
    from .tone3000 import Tone3000, REDIRECT_URI
    if not args.client_id:
        raise SystemExit("need --client-id t3k_pub_… : create an API key on tone3000.com → Settings → API Keys and "
                         f"register the redirect URI {REDIRECT_URI}")
    Tone3000().login(args.client_id, open_browser=not args.no_browser, log=_log)
    _log("logged in to TONE3000")


def cmd_tone3000_search(args) -> None:
    from .tone3000 import Tone3000
    res = Tone3000().search(args.query, gears=args.gears, sizes=args.sizes, sort=args.sort, page_size=args.limit)
    print(f"{res.get('total', '?')} tones match '{args.query}'")
    for t in res.get("data", []):
        makes = ", ".join(x.get("name", "") for x in t.get("makes") or [])
        print(f"  #{t.get('id'):<7} {str(t.get('title'))[:50]:50s} {str(t.get('gear') or ''):9s} {makes[:20]:20s} "
              f"↓{t.get('downloads_count', 0)} ♥{t.get('favorites_count', 0)}")


def cmd_tone3000_fetch(args) -> None:
    from .tone3000 import Tone3000
    files = Tone3000().fetch(args.query, limit=args.limit, gears=args.gears, sizes=args.sizes, sort=args.sort,
                             with_irs=not args.no_irs, log=_log)
    _log(f"{len(files)} file(s) in the library ({NAM_DIR}, {IR_DIR}); `tonematch list` shows them")


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tonematch", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="plugin catalog JSON (default %(default)s)")
        p.add_argument("--knobs", action="append", default=[], help="extra knob-map YAML (repeatable)")
        p.add_argument("--mock", action="store_true", help="use the built-in mock plugins instead of the catalog")
        if p.prog.endswith(("list", "export", "scan")):
            p.add_argument("--nam-dir", action="append", help="extra folder(s) of .nam captures")
            p.add_argument("--ir-dir", action="append", help="extra folder(s) of impulse responses (.wav)")

    p = sub.add_parser("scan", help="scan your VST3/AU plugins and build the catalog")
    p.add_argument("paths", nargs="*", help="plugin folders or bundles (default: the OS standard folders)")
    p.add_argument("--rescan", action="store_true", help="re-probe plugins already in the catalog")
    common(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("list", help="show the catalog (roles, knob kinds)")
    p.add_argument("plugin", nargs="?", help="show every knob of this plugin")
    common(p)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("separate", help="extract the guitar stem from a song (demucs htdemucs_6s)")
    p.add_argument("source", help="audio file or URL")
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, help="excerpt start (s)")
    p.add_argument("--duration", type=float, help="excerpt length (s), 15–40 s of exposed guitar is ideal")
    p.add_argument("--model", default="htdemucs_6s")
    p.add_argument("--device", help="cpu | cuda | mps")
    p.add_argument("--stem", default="guitar")
    p.set_defaults(func=cmd_separate)

    p = sub.add_parser("run", help="match the tone (autonomous, time-budgeted)")
    p.add_argument("--target", required=True, help="guitar stem / isolated guitar recording, or a full song with --separate, or a URL")
    p.add_argument("--separate", action="store_true", help="target is a full mix: run demucs first")
    p.add_argument("--start", type=float)
    p.add_argument("--duration", type=float)
    p.add_argument("--model", default="htdemucs_6s")
    p.add_argument("--device")
    p.add_argument("--di", help="your DI recording (dry guitar). Strongly recommended.")
    p.add_argument("--di-start", type=float)
    p.add_argument("--di-duration", type=float)
    p.add_argument("--max-di-seconds", type=float, default=45.0)
    p.add_argument("--out", required=True, help="run folder")
    p.add_argument("--minutes", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=1, help="parallel renders (threads). 2–4 if your CPU allows")
    p.add_argument("--include", action="append", help="only chains using plugins whose name contains this (repeatable)")
    p.add_argument("--exclude", action="append", help="never use plugins whose name contains this")
    p.add_argument("--chain", nargs="+", help="use exactly this chain: plugin names in order")
    p.add_argument("--instrument", choices=["guitar", "bass"], default="guitar",
                   help="which plugins to build chains from (bass suites are never used for guitar and vice versa)")
    p.add_argument("--max-chains", type=int, default=12, help="chains that get a full screening run")
    p.add_argument("--max-prescreen", type=int, default=300, help="chains (e.g. NAM captures) ranked with one render first")
    p.add_argument("--tone3000", metavar="QUERY", help="fetch captures from TONE3000 for this query before matching")
    p.add_argument("--tone3000-limit", type=int, default=15)
    p.add_argument("--nam-dir", action="append", help="extra folder(s) of .nam captures")
    p.add_argument("--ir-dir", action="append", help="extra folder(s) of impulse responses (.wav)")
    p.add_argument("--keep-chains", type=int, default=3)
    p.add_argument("--allow-ambience", action="store_true", help="also search reverb/delay/modulation knobs")
    p.add_argument("--aligned", action="store_true", help="DI is the *same riff* as the target: add an aligned spectral loss")
    p.add_argument("--weight", action="append", help="override a loss weight, e.g. --weight stats=1.0")
    p.add_argument("--screen-seconds", type=float, default=6.0)
    p.add_argument("--optimize-seconds", type=float, default=10.0)
    p.add_argument("--refine-seconds", type=float, default=20.0)
    p.add_argument("--resume", action="store_true", help="keep optuna trials in <out>/optuna.db and resume them")
    p.add_argument("--seed", type=int, default=0)
    common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("tone3000", help="TONE3000: login, search and fetch NAM captures / IRs")
    t = p.add_subparsers(dest="t3k_cmd", required=True)
    q = t.add_parser("login", help="OAuth login (needs a publishable key from tone3000.com → Settings → API Keys)")
    q.add_argument("--client-id", default=os.environ.get("TONE3000_CLIENT_ID"), help="t3k_pub_… (or env TONE3000_CLIENT_ID)")
    q.add_argument("--no-browser", action="store_true")
    q.set_defaults(func=cmd_tone3000_login)
    q = t.add_parser("search", help="search tones")
    q.add_argument("query")
    q.add_argument("--gears", nargs="+", default=["amp", "amp-cab", "full-rig"])
    q.add_argument("--sizes", nargs="+")
    q.add_argument("--sort", default="downloads-all-time")
    q.add_argument("--limit", type=int, default=20)
    q.set_defaults(func=cmd_tone3000_search)
    q = t.add_parser("fetch", help="download the best-sized model of the top tones into the library")
    q.add_argument("query")
    q.add_argument("--limit", type=int, default=15)
    q.add_argument("--gears", nargs="+", default=["amp", "amp-cab", "full-rig"])
    q.add_argument("--sizes", nargs="+", help="e.g. standard lite")
    q.add_argument("--sort", default="downloads-all-time")
    q.add_argument("--no-irs", action="store_true")
    q.set_defaults(func=cmd_tone3000_fetch)

    p = sub.add_parser("export", help="re-generate the REAPER script / presets of a run")
    p.add_argument("run_dir")
    common(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("selftest", help="end-to-end check with mock plugins (no real plugins needed)")
    p.add_argument("--minutes", type=float, default=2.0)
    p.add_argument("--out", default="runs/selftest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_selftest)
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
