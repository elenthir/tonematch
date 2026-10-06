"""Time-budgeted search over plugin chains and their parameters.

    screen   every candidate chain gets a short Optuna run over its *primary* knobs on a short
             excerpt; chains are ranked by their best loss.
    optimize the best `keep_chains` survive; successive halving splits the time between them,
             primary + secondary knobs, longer excerpt.
    refine   the winner: continuous knobs are narrowed around the incumbent and polished with
             CMA-ES on a long excerpt; discrete choices are frozen.

Everything is driven by a wall-clock deadline, so a run always ends on time with a usable result,
and every improvement is reported through `on_improvement` so a checkpoint can be written.
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna

from .audio import excerpt
from .catalog import Catalog, ParamInfo, PluginInfo
from .chain import ChainSpec, Renderer, Slot
from .features import ToneFeatures, TargetProfile, distance, extract

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger("tonematch")

BAD_LOSS = 1e3


@dataclass
class SearchConfig:
    minutes: float = 30.0
    allow_ambience: bool = False
    prescreen_frac: float = 0.15     # only used when more chains than max_chains are proposed
    screen_frac: float = 0.22
    optimize_frac: float = 0.53
    refine_frac: float = 0.25
    screen_seconds: float = 6.0
    optimize_seconds: float = 10.0
    refine_seconds: float = 20.0
    final_seconds: float = 40.0
    screen_max_trials: int = 60
    keep_chains: int = 3
    max_chains: int = 12
    max_chains_per_core: int = 3     # chains sharing the same amp/suite plugin
    workers: int = 1
    seed: int = 0
    weights: Dict[str, float] = field(default_factory=dict)
    refine_halfwidth: float = 0.12
    storage: Optional[str] = None    # optuna storage url (sqlite:///...) for resumable runs


@dataclass
class Evaluation:
    spec: ChainSpec
    loss: float
    terms: Dict[str, float]
    stage: str
    elapsed: float
    n_evals: int

    def to_dict(self) -> dict:
        return {"chain": self.spec.key, "spec": self.spec.to_dict(), "loss": self.loss,
                "terms": self.terms, "stage": self.stage, "elapsed": self.elapsed, "n_evals": self.n_evals}


@dataclass
class SearchResult:
    best: Evaluation
    improvements: List[dict]
    chain_ranking: List[dict]
    profile: TargetProfile
    elapsed: float
    n_evals: int
    stages: List[dict]

    def to_dict(self) -> dict:
        return {"best": self.best.to_dict(), "improvements": self.improvements,
                "chain_ranking": self.chain_ranking, "profile": asdict(self.profile),
                "elapsed": self.elapsed, "n_evals": self.n_evals, "stages": self.stages}


# ----------------------------------------------------------------------------- chain proposals
SKIPPED_ROLES = ("bass", "instrument", "utility", "modulation", "nam_player", "other")   # never part of a guitar chain
_TONE_WORDS = {"clean": ("clean", "edge", "pristine", "jazz"),
               "crunch": ("crunch", "overdrive", "breakup", "blues", "rock", "plexi", "classic"),
               "high": ("high gain", "high-gain", "hi gain", "lead", "metal", "djent", "heavy", "modern", "rhythm")}


def prior_score(catalog: Catalog, ids: List[str], gain_class: Optional[str]) -> float:
    """Cheap relevance of a chain before any render: capture metadata vs the target's gain class,
    popularity from TONE3000 sidecars. Only used to order chains for the pre-screen."""
    score = 0.0
    for i in ids:
        p = catalog.plugins[i]
        if p.format != "nam":
            score += 1.0                       # real plugins: always worth a look
            continue
        ex = p.extra or {}
        hay = " ".join([str(ex.get("tone_type") or ""), " ".join(map(str, ex.get("tags") or [])), p.name]).lower()
        if gain_class and any(w in hay for w in _TONE_WORDS[gain_class]):
            score += 1.0
        elif gain_class and any(w in hay for c, ws in _TONE_WORDS.items() if c != gain_class for w in ws):
            score -= 0.5
        score += 0.2 * math.log10(1 + float(ex.get("downloads_count") or 0))
    return score


def propose_chains(catalog: Catalog, cfg: SearchConfig, include: Optional[List[str]] = None,
                   exclude: Optional[List[str]] = None, pinned: Optional[List[str]] = None,
                   instrument: str = "guitar", gain_class: Optional[str] = None,
                   limit: Optional[int] = None) -> List[ChainSpec]:
    """Candidate chains from plugin roles. `pinned` = exact list of plugin ids/names → one chain.
    `instrument="bass"` builds chains around the bass suites instead of the guitar ones."""
    if pinned:
        ids = []
        for n in pinned:
            p = catalog.find(n)
            if p is None:
                raise SystemExit(f"plugin '{n}' not found in catalog (run `tonematch scan`, or check `tonematch list`)")
            ids.append(p.id)
        return [ChainSpec([Slot(i) for i in ids])]

    def ok(p: PluginInfo) -> bool:
        if p.error:
            return False
        hay = f"{p.name} {p.id}".lower()
        if include and not any(s.lower() in hay for s in include):
            return False
        if exclude and any(s.lower() in hay for s in exclude):
            return False
        return True

    role = {r: [p for p in catalog.by_role(r) if ok(p)] for r in
            ("amp_suite", "amp", "cab", "drive", "eq", "compressor", "gate", "reverb", "delay", "bass")}
    if instrument == "bass":
        role["amp_suite"], role["amp"] = role["bass"], []
    chains: List[List[str]] = []
    # tier 1: a suite alone; an amp + a cab
    for s in role["amp_suite"]:
        chains.append([s.id])
    for a in role["amp"]:
        for c in role["cab"]:
            chains.append([a.id, c.id])
    # tier 2: a drive pedal in front
    for d in role["drive"][:3]:
        for s in role["amp_suite"]:
            chains.append([d.id, s.id])
        for a in role["amp"]:
            for c in role["cab"][:2]:
                chains.append([d.id, a.id, c.id])
    # tier 3: post EQ
    for e in role["eq"][:1]:
        for s in role["amp_suite"]:
            chains.append([s.id, e.id])
        for a in role["amp"]:
            for c in role["cab"][:1]:
                chains.append([a.id, c.id, e.id])
    if not chains:  # no amp-ish plugin at all: anything we have, drives first
        pool = role["drive"] + role["eq"] + role["compressor"]
        if not pool:
            raise SystemExit("no usable amp/drive/EQ plugin for this instrument in the catalog — see `tonematch list` "
                             "(a misfiled plugin can be given a role in a --knobs YAML)")
        chains = [[p.id] for p in pool] + [[a.id, b.id] for a in pool for b in pool if a is not b][:6]
    # diversity cap per core (amp/suite) plugin, then global cap (the pre-screen stage of the search
    # ranks by a real render; here we only order by a cheap prior so the cap keeps the likely ones)
    chains.sort(key=lambda ids: -prior_score(catalog, ids, gain_class))
    limit = limit if limit is not None else cfg.max_chains
    per_core: Dict[str, int] = {}
    out: List[ChainSpec] = []
    seen = set()
    for ids in chains:
        core = next((i for i in ids if catalog.plugins[i].role in ("amp_suite", "amp")), ids[-1])
        key = " > ".join(ids)
        if key in seen or per_core.get(core, 0) >= cfg.max_chains_per_core:
            continue
        seen.add(key)
        per_core[core] = per_core.get(core, 0) + 1
        out.append(ChainSpec([Slot(i) for i in ids]))
        if len(out) >= limit:
            break
    return out


# ----------------------------------------------------------------------------- the search
_GAIN_PRIOR = {"clean": (0.0, 0.5), "crunch": (0.25, 0.8), "high": (0.5, 1.0)}


def _is_gainish(p: ParamInfo) -> bool:
    n = p.name.lower()
    return p.type == "float" and ("gain" in n or "drive" in n) and not any(
        k in n for k in ("input", "output", "mic", "cab", "low", "mid", "high", "band"))


class ToneSearch:
    def __init__(self, catalog: Catalog, renderer: Renderer, di: np.ndarray, sr: int,
                 target: ToneFeatures, cfg: SearchConfig,
                 on_improvement: Optional[Callable[[Evaluation], None]] = None,
                 on_progress: Optional[Callable[[str], None]] = None):
        self.catalog, self.r, self.di, self.sr, self.target, self.cfg = catalog, renderer, di, sr, target, cfg
        self.on_improvement = on_improvement
        self.on_progress = on_progress or (lambda s: log.info(s))
        self.profile = TargetProfile.from_features(target)
        self.t0 = time.time()
        self.best: Optional[Evaluation] = None
        self.improvements: List[dict] = []
        self.n_evals = 0
        self.stages: List[dict] = []
        self._audio_cache: Dict[float, np.ndarray] = {}
        self._study_cache: Dict[str, optuna.Study] = {}
        self._rng = random.Random(cfg.seed)
        self._lock = __import__("threading").Lock()

    # ------------------------------------------------------------------ helpers
    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def audio(self, seconds: float) -> np.ndarray:
        if seconds not in self._audio_cache:
            self._audio_cache[seconds] = excerpt(self.di, self.sr, seconds)
        return self._audio_cache[seconds]

    def evaluate(self, spec: ChainSpec, seconds: float, stage: str) -> Tuple[float, Dict[str, float]]:
        x = self.audio(seconds)
        try:
            y = self.r.render(spec, x, self.sr)
            if not np.any(np.abs(y) > 1e-6):
                return BAD_LOSS, {"silent": 1.0}
            f = extract(y, self.sr, keep_signal=self.target.signal is not None)
            loss, terms = distance(f, self.target, self.cfg.weights or None)
        except Exception as e:  # a misbehaving plugin should cost a trial, not the run
            log.warning("render failed for %s: %s", spec.key, e)
            return BAD_LOSS, {"error": 1.0}
        if not math.isfinite(loss):
            return BAD_LOSS, terms
        with self._lock:
            self.n_evals += 1
            improved = self.best is None or loss < self.best.loss
            if improved:
                self.best = Evaluation(spec.copy(), loss, terms, stage, self.elapsed, self.n_evals)
                self.improvements.append({"elapsed": self.elapsed, "loss": loss, "stage": stage,
                                          "chain": spec.key, "n_evals": self.n_evals})
                best = self.best
        if improved and self.on_improvement:
            try:
                self.on_improvement(best)
            except Exception as e:  # pragma: no cover
                log.warning("checkpoint failed: %s", e)
        return loss, terms

    def space(self, chain: ChainSpec, kinds: set, prior: bool = False,
              narrow_around: Optional[ChainSpec] = None) -> List[dict]:
        """Searchable parameters of a chain → list of dicts describing an Optuna distribution."""
        out = []
        for si, slot in enumerate(chain.slots):
            info = self.catalog.plugins[slot.plugin_id]
            inc = narrow_around.slots[si].params if narrow_around else None
            for p in info.params:
                if p.kind not in kinds:
                    continue
                name = f"{si}|{p.name}"
                grp = {"group": p.group, "selector": p.group_selector} if p.group else {}
                if p.n_values and p.n_values > 1 and p.type != "float" or (p.n_values and p.n_values <= 12):
                    if inc is not None:    # refine: freeze discrete choices
                        continue
                    if p.n_values <= 64:
                        out.append({"name": name, "slot": si, "param": p.name, "kind": "cat", "choices": p.raw_centers,
                                    "values": p.values, **grp})
                    else:
                        out.append({"name": name, "slot": si, "param": p.name, "kind": "int", "n": p.n_values, **grp})
                    continue
                lo, hi = (p.range or (0.0, 1.0))
                if prior and _is_gainish(p):
                    glo, ghi = _GAIN_PRIOR[self.profile.gain_class]
                    lo, hi = max(lo, glo), min(hi, ghi)
                if inc is not None and p.name in inc:
                    c, hw = inc[p.name], self.cfg.refine_halfwidth
                    lo, hi = max(lo, c - hw), min(hi, c + hw)
                out.append({"name": name, "slot": si, "param": p.name, "kind": "float", "lo": lo, "hi": hi, **grp})
        # selectors first, so grouped knobs can be made conditional on them
        selectors = {(e["slot"], e["selector"]) for e in out if e.get("selector")}
        out.sort(key=lambda d: 0 if (d["slot"], d["param"]) in selectors else 1)
        return out

    def _selected_group(self, slot_idx: int, selector: str, chosen: Dict[tuple, float], base: ChainSpec) -> Optional[str]:
        """Display value of a channel selector: from this trial, else the base spec, else the default."""
        info = self.catalog.plugins[base.slots[slot_idx].plugin_id]
        sel = info.param(selector)
        if sel is None or not sel.values:
            return None
        raw = chosen.get((slot_idx, selector))
        if raw is None:
            raw = base.slots[slot_idx].params.get(selector, sel.fixed_value if sel.kind == "fixed" else sel.default)
        idx = int(round(float(raw) * (len(sel.values) - 1))) if len(sel.values) > 1 else 0
        return str(sel.values[max(0, min(idx, len(sel.values) - 1))]).strip()

    def _suggest(self, trial: optuna.Trial, space: List[dict], base: ChainSpec) -> ChainSpec:
        spec = base.copy()
        chosen: Dict[tuple, float] = {}
        for d in space:
            if d.get("group"):
                if self._selected_group(d["slot"], d["selector"], chosen, base) != d["group"]:
                    continue                      # knob of a channel that is not selected: untouched
            if d["kind"] == "float":
                v = trial.suggest_float(d["name"], d["lo"], d["hi"])
            elif d["kind"] == "cat":
                v = trial.suggest_categorical(d["name"], d["choices"])
            else:
                i = trial.suggest_int(d["name"], 0, d["n"] - 1)
                v = i / max(1, d["n"] - 1)
            spec.slots[d["slot"]].params[d["param"]] = float(v)
            chosen[(d["slot"], d["param"])] = float(v)
        return spec

    def _study(self, name: str, sampler) -> optuna.Study:
        if name not in self._study_cache:
            self._study_cache[name] = optuna.create_study(
                study_name=name, direction="minimize", sampler=sampler,
                storage=self.cfg.storage, load_if_exists=bool(self.cfg.storage))
        return self._study_cache[name]

    def _optimize(self, study: optuna.Study, space: List[dict], base: ChainSpec, seconds: float,
                  stage: str, timeout: float, n_trials: Optional[int] = None) -> None:
        if timeout <= 0.5:
            return

        def objective(trial: optuna.Trial) -> float:
            spec = self._suggest(trial, space, base)
            loss, terms = self.evaluate(spec, seconds, stage)
            trial.set_user_attr("terms", terms)
            return loss

        study.optimize(objective, timeout=timeout, n_trials=n_trials, n_jobs=max(1, self.cfg.workers),
                       gc_after_trial=False, catch=(Exception,))

    def _best_spec(self, study: optuna.Study, space: List[dict], base: ChainSpec) -> Optional[ChainSpec]:
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if not done:
            return None
        t = min(done, key=lambda t: t.value)
        spec = base.copy()
        for d in space:
            if d["name"] not in t.params:
                continue
            v = t.params[d["name"]]
            spec.slots[d["slot"]].params[d["param"]] = float(v / max(1, d["n"] - 1)) if d["kind"] == "int" else float(v)
        return spec

    def _enqueue_seeds(self, study: optuna.Study, space: List[dict], seeds: List[ChainSpec]) -> None:
        for s in seeds:
            params = {}
            for d in space:
                v = s.slots[d["slot"]].params.get(d["param"])
                if v is None:
                    continue
                if d["kind"] == "float":
                    params[d["name"]] = float(min(d["hi"], max(d["lo"], v)))
                elif d["kind"] == "cat":
                    params[d["name"]] = min(d["choices"], key=lambda c: abs(c - v))
                else:
                    params[d["name"]] = int(round(v * (d["n"] - 1)))
            if params:
                try:
                    study.enqueue_trial(params, skip_if_exists=True)
                except Exception:
                    pass

    def _noon_seed(self, chain: ChainSpec) -> ChainSpec:
        """Everything at noon, gain from the target's gain class — a sane place to start any amp."""
        spec = chain.copy()
        glo, ghi = _GAIN_PRIOR[self.profile.gain_class]
        for slot in spec.slots:
            for p in self.catalog.plugins[slot.plugin_id].params:
                if p.kind in ("primary", "secondary") and p.type == "float" and not (p.n_values and p.n_values <= 12):
                    slot.params[p.name] = (glo + ghi) / 2 if _is_gainish(p) else 0.5
        return spec

    # ------------------------------------------------------------------ stages
    def run(self, chains: List[ChainSpec]) -> SearchResult:
        cfg = self.cfg
        total = cfg.minutes * 60.0
        deadline = self.t0 + total
        kinds_b = {"primary", "secondary"} | ({"ambience"} if cfg.allow_ambience else set())
        self.on_progress(f"target looks {self.profile.gain_class} (flatness {self.profile.flatness_db:.1f} dB, "
                         f"HF {self.profile.hf_ratio_db:.1f} dB, dynamics {self.profile.dynamics_db:.1f} dB); "
                         f"{len(chains)} candidate chain(s), budget {cfg.minutes:.1f} min")

        # ---- stage 0: pre-screen when there are more chains than we can afford to screen
        # (typical with a library of NAM captures): one render each at the "noon" seed, keep the best.
        if len(chains) > cfg.max_chains:
            t_pre = total * cfg.prescreen_frac
            self.on_progress(f"[pre-screen] {len(chains)} chains, one render each, up to {t_pre:.0f}s")
            scored = []
            t_end = time.time() + t_pre
            for ci, chain in enumerate(chains):
                if time.time() > t_end:
                    self.on_progress(f"    out of time after {ci} chains; the rest are skipped")
                    break
                loss, _ = self.evaluate(self._noon_seed(chain), cfg.screen_seconds, "prescreen")
                scored.append((loss, chain))
            scored.sort(key=lambda t: t[0])
            self.stages.append({"stage": "prescreen", "elapsed": self.elapsed, "n_evals": self.n_evals,
                                "ranking": [{"chain": c.key, "loss": l} for l, c in scored]})
            chains = [c for l, c in scored[: cfg.max_chains] if l < BAD_LOSS]
            self.on_progress("    kept: " + ", ".join(f"{c.key} ({l:.2f})" for l, c in scored[: cfg.max_chains]))
            total_left = deadline - time.time()
        else:
            total_left = total
        # ---- stage A: screen
        t_screen = total_left * cfg.screen_frac if len(chains) > 1 else total_left * 0.12
        per_chain = t_screen / max(1, len(chains))
        screen_best: Dict[str, Tuple[float, ChainSpec]] = {}
        for ci, chain in enumerate(chains):
            space = self.space(chain, {"primary"}, prior=True)
            study = self._study(f"screen:{chain.short_id}", optuna.samplers.TPESampler(
                seed=cfg.seed + ci, n_startup_trials=max(6, len(space)), multivariate=True))
            self._enqueue_seeds(study, space, [self._noon_seed(chain), chain])
            self.on_progress(f"[screen {ci + 1}/{len(chains)}] {chain.key}: {len(space)} knobs, {per_chain:.0f}s")
            self._optimize(study, space, chain, cfg.screen_seconds, "screen",
                           min(per_chain, deadline - time.time()), n_trials=cfg.screen_max_trials)
            bs = self._best_spec(study, space, chain)
            bv = study.best_value if bs is not None else BAD_LOSS
            screen_best[chain.key] = (bv, bs or chain)
            self.on_progress(f"    best {bv:.2f}  ({len(study.trials)} trials, {self.n_evals} renders total)")
        ranking = sorted(screen_best.items(), key=lambda kv: kv[1][0])
        self.stages.append({"stage": "screen", "elapsed": self.elapsed, "n_evals": self.n_evals,
                            "ranking": [{"chain": k, "loss": v[0]} for k, v in ranking]})
        survivors = [ChainSpec.from_dict(v[1].to_dict()) for k, v in ranking[: cfg.keep_chains] if v[0] < BAD_LOSS]
        if not survivors:
            raise SystemExit("every candidate chain failed to render — check the plugins with `tonematch scan`")

        # ---- stage B: optimise survivors with successive halving
        t_opt_end = deadline - total_left * cfg.refine_frac
        rounds = max(1, math.ceil(math.log2(len(survivors))) + 1) if len(survivors) > 1 else 1
        studies: Dict[str, Tuple[optuna.Study, List[dict], ChainSpec]] = {}
        alive = list(survivors)
        for rnd in range(rounds):
            remaining = t_opt_end - time.time()
            if remaining <= 1:
                break
            share = remaining / (rounds - rnd) / max(1, len(alive))
            for chain in alive:
                if chain.key not in studies:
                    base = ChainSpec([Slot(s.plugin_id) for s in chain.slots])
                    space = self.space(base, kinds_b)
                    study = self._study(f"opt:{base.short_id}", optuna.samplers.TPESampler(
                        seed=cfg.seed + 100, n_startup_trials=max(8, len(space) // 2), multivariate=True))
                    self._enqueue_seeds(study, space, [chain, self._noon_seed(base)])
                    studies[chain.key] = (study, space, base)
                study, space, base = studies[chain.key]
                self.on_progress(f"[optimize r{rnd + 1}] {chain.key}: {len(space)} knobs, {share:.0f}s")
                self._optimize(study, space, base, cfg.optimize_seconds, "optimize", min(share, deadline - time.time()))
                self.on_progress(f"    best {study.best_value:.2f}  ({len(study.trials)} trials)")
            alive.sort(key=lambda c: studies[c.key][0].best_value)
            self.stages.append({"stage": f"optimize-r{rnd + 1}", "elapsed": self.elapsed, "n_evals": self.n_evals,
                                "ranking": [{"chain": c.key, "loss": studies[c.key][0].best_value} for c in alive]})
            if len(alive) > 1:
                alive = alive[: max(1, math.ceil(len(alive) / 2))]

        # ---- stage C: refine the winner
        winner_key = alive[0].key
        study, space, base = studies[winner_key]
        incumbent = self._best_spec(study, space, base) or alive[0]
        ref_space = self.space(base, kinds_b, narrow_around=incumbent)
        frozen = incumbent.copy()  # discrete choices stay as found
        t_ref = deadline - time.time()
        if t_ref > 2 and ref_space:
            sampler = optuna.samplers.CmaEsSampler(seed=cfg.seed + 7) \
                if all(d["kind"] == "float" for d in ref_space) else \
                optuna.samplers.TPESampler(seed=cfg.seed + 7, multivariate=True)
            rstudy = self._study(f"refine:{base.short_id}", sampler)
            self._enqueue_seeds(rstudy, ref_space, [incumbent])
            self.on_progress(f"[refine] {winner_key}: {len(ref_space)} knobs, {t_ref:.0f}s")
            self._optimize(rstudy, ref_space, frozen, cfg.refine_seconds, "refine", t_ref)
            rs = self._best_spec(rstudy, ref_space, frozen)
            if rs is not None:
                incumbent = rs
            self.on_progress(f"    best {rstudy.best_value:.2f}  ({len(rstudy.trials)} trials)")

        # ---- final: score the incumbent on a long excerpt and make sure `best` is consistent
        loss, terms = self.evaluate(incumbent, cfg.final_seconds, "final")
        final = Evaluation(incumbent, loss, terms, "final", self.elapsed, self.n_evals)
        if self.best is None or self.best.spec.key != incumbent.key or self.best.loss > loss * 1.5:
            self.best = final
        self.stages.append({"stage": "final", "elapsed": self.elapsed, "n_evals": self.n_evals, "loss": loss, "terms": terms})
        ranking_out = [{"chain": k, "screen_loss": v[0],
                        "optimized_loss": studies[k][0].best_value if k in studies else None} for k, v in ranking]
        return SearchResult(best=self.best, improvements=self.improvements, chain_ranking=ranking_out,
                            profile=self.profile, elapsed=self.elapsed, n_evals=self.n_evals, stages=self.stages)
