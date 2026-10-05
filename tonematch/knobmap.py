"""Knob-map rules (YAML) → per-plugin role and per-parameter kind."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional

import yaml

KINDS = ("primary", "secondary", "ambience", "fixed", "excluded")


@dataclass
class PluginRule:
    match: re.Pattern
    role: Optional[str] = None
    vendor: Optional[str] = None
    primary: List[re.Pattern] = field(default_factory=list)
    secondary: List[re.Pattern] = field(default_factory=list)
    ambience: List[re.Pattern] = field(default_factory=list)
    excluded: List[re.Pattern] = field(default_factory=list)
    fixed: Dict[re.Pattern, float] = field(default_factory=dict)
    ranges: Dict[re.Pattern, tuple] = field(default_factory=dict)   # name regex -> (lo, hi) normalised


def _rx(s: str) -> re.Pattern:
    try:
        return re.compile(s, re.I)
    except re.error:
        return re.compile(re.escape(s), re.I)


def _rxs(lst) -> List[re.Pattern]:
    return [_rx(str(s)) for s in (lst or [])]


@dataclass
class KnobMap:
    roles: Dict[str, List[re.Pattern]] = field(default_factory=dict)
    excluded: List[re.Pattern] = field(default_factory=list)
    ambience: List[re.Pattern] = field(default_factory=list)
    primary: List[re.Pattern] = field(default_factory=list)
    secondary: List[re.Pattern] = field(default_factory=list)
    plugins: List[PluginRule] = field(default_factory=list)

    # ------------------------------------------------------------------ loading
    @staticmethod
    def load(extra_files: Optional[List[str | Path]] = None, include_default: bool = True) -> "KnobMap":
        km = KnobMap()
        docs = []
        if include_default:
            docs.append(yaml.safe_load(resources.files("tonematch").joinpath("knobs/default.yaml").read_text()))
        for f in extra_files or []:
            docs.append(yaml.safe_load(Path(f).read_text()) or {})
        for d in docs:
            km._merge(d)
        return km

    def _merge(self, d: dict) -> None:
        for role, pats in (d.get("roles") or {}).items():
            self.roles.setdefault(role, []).extend(_rxs(pats))
        p = d.get("params") or {}
        self.excluded.extend(_rxs(p.get("excluded")))
        self.ambience.extend(_rxs(p.get("ambience")))
        self.primary.extend(_rxs(p.get("primary")))
        self.secondary.extend(_rxs(p.get("secondary")))
        new_rules = []
        for pr in d.get("plugins") or []:
            new_rules.append(PluginRule(
                match=_rx(pr["match"]), role=pr.get("role"), vendor=pr.get("vendor"),
                primary=_rxs(pr.get("primary")), secondary=_rxs(pr.get("secondary")),
                ambience=_rxs(pr.get("ambience")), excluded=_rxs(pr.get("excluded")),
                fixed={_rx(k): float(v) for k, v in (pr.get("fixed") or {}).items()},
                ranges={_rx(k): (float(v[0]), float(v[1])) for k, v in (pr.get("ranges") or {}).items()},
            ))
        self.plugins = new_rules + self.plugins   # later files take precedence

    # ------------------------------------------------------------------ queries
    def rule_for(self, plugin_name: str) -> Optional[PluginRule]:
        for r in self.plugins:
            if r.match.search(plugin_name):
                return r
        return None

    def role_for(self, plugin_name: str, category: str = "", is_instrument: bool = False) -> str:
        r = self.rule_for(plugin_name)
        if r and r.role:
            return r.role
        if is_instrument:
            return "instrument"
        hay = f"{plugin_name} {category}"
        for role, pats in self.roles.items():       # YAML order = priority
            if any(p.search(hay) for p in pats):
                return role
        return "other"

    def classify_param(self, plugin_name: str, param_name: str, n_values: Optional[int] = None):
        """-> (kind, fixed_value, range) for a parameter."""
        r = self.rule_for(plugin_name)
        lists = []
        if r:
            lists.append(("fixed", r.fixed))
            lists.append(("excluded", r.excluded))
            lists.append(("ambience", r.ambience))
            lists.append(("primary", r.primary))
            lists.append(("secondary", r.secondary))
        lists += [("excluded", self.excluded), ("ambience", self.ambience),
                  ("primary", self.primary), ("secondary", self.secondary)]
        rng = None
        if r:
            for pat, lohi in r.ranges.items():
                if pat.search(param_name):
                    rng = lohi
                    break
        for kind, pats in lists:
            if kind == "fixed":
                for pat, val in pats.items():
                    if pat.search(param_name):
                        return "fixed", val, rng
            elif any(p.search(param_name) for p in pats):
                return kind, None, rng
        return "secondary", None, rng
