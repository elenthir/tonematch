import json
import time

import numpy as np
import pytest

from tonematch.audio import SR, synth_di
from tonematch.chain import ChainSpec, Slot
from tonematch import features as F
from tonematch.search import SearchConfig, ToneSearch, propose_chains
from tonematch.export import write_checkpoint, write_reaper_script


def test_render_applies_params(renderer, di):
    a = renderer.render(ChainSpec([Slot("mock_amp", {"Amp Gain": 0.1})]), di, SR)
    b = renderer.render(ChainSpec([Slot("mock_amp", {"Amp Gain": 0.9})]), di, SR)
    assert a.shape == di.shape and b.shape == di.shape
    assert F.distance(F.extract(a, SR), F.extract(b, SR))[0] > 1.0
    # deterministic
    c = renderer.render(ChainSpec([Slot("mock_amp", {"Amp Gain": 0.1})]), di, SR)
    assert np.allclose(a, c)


def test_fixed_and_excluded_params_respected(renderer, catalog, di):
    amp = catalog.plugins["mock_amp"]
    amp.param("Reverb Mix").kind = "fixed"
    amp.param("Reverb Mix").fixed_value = 0.0
    renderer.render(ChainSpec([Slot("mock_amp", {"Reverb Mix": 0.9})]), di, SR)
    assert renderer.handle("mock_amp").get_raw("Reverb Mix") == 0.0
    amp.param("Reverb Mix").kind = "ambience"
    amp.param("Reverb Mix").fixed_value = None


def test_propose_chains(catalog):
    cfg = SearchConfig(max_chains=10)
    chains = propose_chains(catalog, cfg)
    keys = [c.key for c in chains]
    assert "mock_amp" in keys and "mock_drive > mock_amp" in keys and "mock_amp > mock_eq" in keys
    assert propose_chains(catalog, cfg, pinned=["Mock Overdrive", "mock_amp"])[0].key == "mock_drive > mock_amp"
    assert all("drive" not in k for k in (c.key for c in propose_chains(catalog, cfg, exclude=["overdrive"])))
    with pytest.raises(SystemExit):
        propose_chains(catalog, cfg, pinned=["does-not-exist"])


def test_search_recovers_tone(catalog, renderer, di):
    hidden = ChainSpec([Slot("mock_amp", {"Amp Gain": 0.8, "Amp Bass": 0.3, "Amp Middle": 0.2,
                                          "Amp Treble": 0.7, "Amp Presence": 0.5, "Cab Select": 1.0})])
    target = F.extract(renderer.render(hidden, synth_di(SR, 7.0, seed=11), SR), SR)
    cfg = SearchConfig(minutes=0.5, keep_chains=1, max_chains=2, screen_max_trials=15, seed=1)
    improvements = []
    s = ToneSearch(catalog, renderer, di, SR, target, cfg, on_improvement=lambda e: improvements.append(e.loss))
    baseline = s.evaluate(ChainSpec([Slot("mock_amp")]), 6.0, "b")[0]
    s.best, s.improvements = None, []
    t0 = time.time()
    res = s.run(propose_chains(catalog, cfg))
    assert time.time() - t0 < 60
    assert res.best.loss < 0.6 * baseline
    assert improvements == sorted(improvements, reverse=True)
    assert res.n_evals > 30 and res.stages[-1]["stage"] == "final"
    assert res.best.spec.key == "mock_amp" or res.best.loss < baseline * 0.5


def test_export_writes_everything(tmp_path, catalog, renderer, di):
    spec = ChainSpec([Slot("mock_drive", {"Drive": 0.4}), Slot("mock_amp", {"Amp Gain": 0.6, "Cab Select": 1 / 3})])
    from tonematch.search import Evaluation
    ev = Evaluation(spec, 1.23, {"ltas": 1.0}, "final", 10.0, 50)
    write_checkpoint(tmp_path, renderer, ev, di, SR, di)
    for f in ("best.json", "best.md", "best.wav", "di.wav", "target.wav", "reaper/apply_tone.lua"):
        assert (tmp_path / f).exists(), f
    payload = json.loads((tmp_path / "best.json").read_text())
    assert payload["chain"] == "mock_drive > mock_amp"
    amp = payload["plugins"][1]
    assert amp["name"] == "Mock Amp Suite"
    assert {p["name"]: p["display"] for p in amp["params"]}["Cab Select"] == "2x12 Greenback"
    lua = (tmp_path / "reaper/apply_tone.lua").read_text()
    assert '{"Amp Gain", 0.600000}' in lua and '"Mock Overdrive"' in lua
    assert "Output Level" not in lua          # excluded knobs are left alone
    lupa = pytest.importorskip("lupa")
    rt = lupa.LuaRuntime()
    rt.compile(lua)                           # syntax check without running (no `reaper` global here)
    chain = rt.eval(lua[lua.index("local CHAIN = ") + len("local CHAIN = "): lua.index("\nlocal function msg")])
    assert chain[2]["name"] == "Mock Amp Suite" and len(chain[1]["params"]) == 4


def test_cli_selftest_short(tmp_path):
    from tonematch.cli import main
    main(["selftest", "--minutes", "0.4", "--out", str(tmp_path / "st"), "--seed", "2"])
    assert (tmp_path / "st/report.png").exists()
