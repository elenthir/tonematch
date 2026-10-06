from tonematch.catalog import Catalog, ParamInfo, PluginInfo, tag_plugin
from tonematch.knobmap import KnobMap


def test_default_knobmap_roles_and_param_kinds():
    km = KnobMap.load()
    assert km.role_for("Archetype Nolly X") == "amp_suite"
    assert km.role_for("Archetype: Gojira X") == "amp_suite"
    assert km.role_for("ReaEQ") == "eq"
    assert km.role_for("Some IR Loader", "Fx|Cab") == "cab"
    assert km.role_for("TotallyUnknown") == "other"
    cls = lambda n: km.classify_param("Archetype Nolly X", n)[0]
    assert cls("Amp 2 Gain") == "primary"
    assert cls("Amp 2 Presence") == "primary"
    assert cls("Cab 1 Mic 1 Type") == "primary"
    assert cls("Delay Time") == "ambience"
    assert cls("Reverb Mix") == "ambience"
    assert cls("Output Gain") == "excluded"
    assert cls("Transpose Semitones") == "excluded"
    assert cls("Noise Gate Threshold") == "excluded"
    kind, val, _ = km.classify_param("Archetype Nolly X", "Noise Gate Active")
    assert kind == "fixed" and val == 1.0
    kind, val, _ = km.classify_param("Archetype Nolly X", "Doubler Active")
    assert kind == "fixed" and val == 0.0
    assert cls("Pedal 1 Level") == "secondary"


def test_user_knobmap_overrides(tmp_path):
    f = tmp_path / "mine.yaml"
    f.write_text("""
plugins:
  - match: "nolly"
    role: amp_suite
    excluded: ["Amp 1 .*"]           # only use amp 2
    ranges: {"Amp 2 Gain": [0.3, 0.7]}
    fixed: {"Amp Select": 1.0}
""")
    km = KnobMap.load([f])
    assert km.classify_param("Archetype Nolly X", "Amp 1 Gain")[0] == "excluded"
    kind, _, rng = km.classify_param("Archetype Nolly X", "Amp 2 Gain")
    assert kind == "primary" and rng == (0.3, 0.7)
    assert km.classify_param("Archetype Nolly X", "Amp Select")[:2] == ("fixed", 1.0)


def test_catalog_roundtrip_and_tagging(tmp_path):
    p = PluginInfo(id="x", name="Archetype Plini X", path="/x.vst3", params=[
        ParamInfo("Amp Gain", "amp_gain", "float", 0.5),
        ParamInfo("Cab IR", "cab_ir", "str", 0.0, n_values=200, values=None, raw_centers=None),
        ParamInfo("Pedal Mix Knob", "pedal_mix_knob", "float", 0.5),
        ParamInfo("Big List", "big_list", "str", 0.0, n_values=300),
    ])
    tag_plugin(p, KnobMap.load())
    assert p.role == "amp_suite" and p.vendor == "Neural DSP"
    kinds = {q.name: q.kind for q in p.params}
    assert kinds["Amp Gain"] == "primary"
    assert kinds["Cab IR"] == "primary"           # big discrete list, but primary by name
    assert kinds["Big List"] == "excluded"        # big discrete list with no role → not searched
    cat = Catalog({"x": p})
    cat.save(tmp_path / "c.json")
    back = Catalog.load(tmp_path / "c.json")
    assert back.plugins["x"].params[1].n_values == 200
    assert back.find("plini").id == "x"
    assert len(back.plugins["x"].searchable()) == 3


def test_bass_instrument_utility_roles_are_kept_out_of_guitar_chains():
    km = KnobMap.load()
    assert km.role_for("Darkglass Ultra") == "bass"
    assert km.role_for("Parallax X") == "bass"
    assert km.role_for("Ampeg SVT-VR") == "bass"
    assert km.role_for("Bass Amp Room") == "bass"
    assert km.role_for("Superior Drummer 3") == "instrument"
    assert km.role_for("NeuralNote") == "instrument"
    assert km.role_for("Some Synth", "Instrument|Synth") == "instrument"
    assert km.role_for("Nice Amp", "", is_instrument=True) == "instrument"
    assert km.role_for("Pro-L 2 Limiter") == "utility"
    assert km.role_for("ReaTune") == "utility"
    assert km.role_for("Archetype Nolly X") == "amp_suite"
    assert km.role_for("Neural Amp Modeler") == "nam_player"
    assert km.role_for("Nice Amp 2") == "amp"
    assert km.role_for("Soundshed Guitar") == "other"       # unknown: listed, never chained

    from tonematch.catalog import Catalog, PluginInfo, ParamInfo
    from tonematch.search import SearchConfig, propose_chains
    cat = Catalog()
    for name in ("Archetype Nolly X", "Darkglass Ultra", "Superior Drummer 3", "ReaTune", "Nice Amp 2", "Some Cab IR"):
        p = PluginInfo(id=name, name=name, path=name, params=[ParamInfo("Gain", "gain", "float", 0.5)])
        tag_plugin(p, km)
        cat.plugins[name] = p
    keys = [c.key for c in propose_chains(cat, SearchConfig())]
    assert "Archetype Nolly X" in keys and "Nice Amp 2 > Some Cab IR" in keys
    assert not any(("Darkglass" in k or "Drummer" in k or "ReaTune" in k) for k in keys)
    bass_keys = [c.key for c in propose_chains(cat, SearchConfig(), instrument="bass")]
    assert bass_keys == ["Darkglass Ultra"]


def test_probe_result_survives_stdout_noise(tmp_path, monkeypatch):
    """A plugin printing to stdout while loading must not corrupt the scan."""
    import json, sys, subprocess
    from tonematch import catalog as C
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        out = cmd[cmd.index("--out") + 1]
        open(out, "w").write(json.dumps([{"id": "X", "name": "X", "path": cmd[cmd.index("--probe") + 1],
                                          "vendor": "", "category": "", "format": "vst3", "role": "other",
                                          "params": [], "latency": 0, "is_instrument": False, "error": None}]))
        return subprocess.CompletedProcess(cmd, 0, stdout="LICENSE OK [garbage\n", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    infos = C.probe_plugin_file(tmp_path / "X.vst3")
    assert len(infos) == 1 and infos[0].name == "X" and not infos[0].error

    def crash_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="boom")
    monkeypatch.setattr(subprocess, "run", crash_run)
    infos = C.probe_plugin_file(tmp_path / "Y.vst3")
    assert infos[0].error and "boom" in infos[0].error


def test_find_plugin_files_dedups_windows_bundles(tmp_path):
    from tonematch.catalog import find_plugin_files
    b = tmp_path / "Archetype Nolly X.vst3" / "Contents" / "x86_64-win"
    b.mkdir(parents=True)
    (b / "Archetype Nolly X.vst3").write_bytes(b"")
    (tmp_path / "Flat.vst3").write_bytes(b"")
    found = sorted(p.name for p in find_plugin_files([tmp_path]))
    assert found == ["Archetype Nolly X.vst3", "Flat.vst3"]
    assert all(p.parent == tmp_path for p in find_plugin_files([tmp_path]))


def test_vendor_rule_and_modulation_role():
    km = KnobMap.load()
    assert km.role_for("Tone King Imperial MKII", "Fx|Distortion", vendor="Neural DSP") == "amp_suite"
    assert km.role_for("Some Future Model", "Fx|Distortion", vendor="Neural DSP") == "amp_suite"
    assert km.role_for("Darkglass Ultra", vendor="Neural DSP") == "bass"
    assert km.role_for("MixWave EHX Big Muff", vendor="MixWave") == "drive"
    assert km.role_for("MixWave EHX Electric Mistress") == "modulation"
    assert km.role_for("Efektor Omnivibe") == "modulation"
    assert km.role_for("Terraform") == "modulation"
    cls = lambda n: km.classify_param("Archetype Plini X", n)[0]
    assert cls("EQ 1 Band 3 Gain") == "secondary"
    assert cls("Cab 1 Mic 1 Position") == "secondary"
    assert cls("Cab 1 Mic 1 Type") == "primary"
    assert cls("Amp 1 Low Cut") == "secondary"
    assert cls("Pedal 1 Level") == "secondary"
    assert cls("Amp 2 Gain") == "primary"
    assert cls("Amp 2 Highs") == "primary"


def test_resolve_plugin_path(tmp_path):
    from tonematch.catalog import resolve_plugin_path
    b = tmp_path / "NAM.vst3"
    (b / "Contents" / "x86_64-win").mkdir(parents=True)
    (b / "Contents" / "x86_64-win" / "NAM.vst3").write_bytes(b"")
    (b / "Contents" / "x86_64-linux").mkdir()
    (b / "Contents" / "x86_64-linux" / "NAM.so").write_bytes(b"")
    assert resolve_plugin_path(b, "Windows").endswith("x86_64-win" + __import__("os").sep + "NAM.vst3")
    assert resolve_plugin_path(b, "Linux").endswith("NAM.so")
    assert resolve_plugin_path(b, "Darwin") == str(b)
    flat = tmp_path / "Flat.vst3"
    flat.write_bytes(b"")
    assert resolve_plugin_path(flat, "Windows") == str(flat)


PLINI_PARAMS = [  # real names reported by Archetype Plini X (n_values, values)
    ("Input Gain", None, None), ("Output Gain", None, None), ("Gate Active", 2, ["Off", "On"]),
    ("Gate Threshold", None, None), ("Transpose", 25, None), ("Doubler Active", 2, ["Off", "On"]),
    ("Active Pre FX Section", 2, ["Off", "On"]), ("Compressor Active", 2, ["Off", "On"]),
    ("Compressor Threshold", None, None), ("Octaver Active", 2, ["Off", "On"]),
    ("Overdrive Active", 2, ["Off", "On"]), ("Overdrive Gain", None, None), ("Overdrive Level", None, None),
    ("Delay 1 Active", 2, ["Off", "On"]), ("Delay 1 Sync Note", 21, None),
    ("Active Amp Section", 2, ["Off", "On"]), ("Amp Type", 3, ["Clean", "Crunch", "Lead"]),
    ("Clean Gain", None, None), ("Clean Bright", 2, ["Off", "On"]), ("Clean Bass", None, None),
    ("Clean Master", None, None), ("Clean Presence", None, None), ("Clean Output", None, None),
    ("Crunch Gain", None, None), ("Crunch Treble", None, None), ("Crunch Output", None, None),
    ("Lead Gain", None, None), ("Lead Mid", None, None), ("Lead Output", None, None),
    ("Active Cab Section", 2, ["Off", "On"]), ("Cab L Active", 2, ["Off", "On"]),
    ("Cab L Mic Type", 7, ["57", "121", "421", "160", "414", "67", "87"]), ("Cab L Position", None, None),
    ("Cab L Level", None, None), ("Cab L Pan", 101, None), ("Cab L Phase", 2, ["Off", "On"]),
    ("R3", 1, ["x"]), ("Active EQ Section", 2, ["Off", "On"]), ("Clean EQ Active", 2, ["Off", "On"]),
    ("Clean EQ 65 Hz", None, None), ("Clean EQ 1 kHz", None, None), ("Lead EQ 16 kHz", None, None),
    ("Clean EQ High Pass", None, None), ("Active Post FX Section", 2, ["Off", "On"]),
    ("Chorus Active", 2, ["Off", "On"]), ("Delay 2 Note", 21, None), ("Reverb Mix", None, None),
    ("Bypass", 2, ["Off", "On"]),
]


def _plini():
    p = PluginInfo(id="plini", name="Archetype Plini X", path="/p.vst3", vendor="Neural DSP", params=[
        ParamInfo(n, n.lower().replace(" ", "_"), "float" if nv is None else "str", 0.5, n_values=nv, values=vals,
                  raw_centers=[i / max(1, nv - 1) for i in range(nv)] if nv and nv <= 64 else None)
        for n, nv, vals in PLINI_PARAMS])
    tag_plugin(p, KnobMap.load())
    return p


def test_neural_dsp_real_parameter_names():
    p = _plini()
    kinds = {q.name: q.kind for q in p.params}
    fixed = {q.name: q.fixed_value for q in p.params if q.kind == "fixed"}
    assert fixed == {"Gate Active": 1.0, "Doubler Active": 0.0, "Active Amp Section": 1.0,
                     "Active Cab Section": 1.0, "Cab L Active": 1.0}
    for n in ("Amp Type", "Clean Gain", "Clean Bright", "Clean Bass", "Clean Master", "Clean Presence", "Lead Mid",
              "Cab L Mic Type", "Overdrive Active", "Overdrive Gain", "Compressor Active", "Active Pre FX Section",
              "Active EQ Section", "Clean EQ Active"):
        assert kinds[n] == "primary", n
    for n in ("Clean EQ 65 Hz", "Clean EQ 1 kHz", "Lead EQ 16 kHz", "Clean EQ High Pass", "Clean Output",
              "Lead Output", "Cab L Position", "Cab L Level", "Cab L Phase", "Overdrive Level", "Compressor Threshold"):
        assert kinds[n] == "secondary", n
    for n in ("Octaver Active", "Delay 1 Active", "Chorus Active", "Reverb Mix", "Active Post FX Section", "Delay 2 Note"):
        assert kinds[n] == "ambience", n
    for n in ("Input Gain", "Output Gain", "Transpose", "Gate Threshold", "Cab L Pan", "Bypass", "R3", "Delay 1 Sync Note"):
        assert kinds[n] == "excluded", n
    # channel groups
    g = {q.name: (q.group, q.group_selector) for q in p.params}
    assert g["Clean Gain"] == ("Clean", "Amp Type") and g["Lead EQ 16 kHz"] == ("Lead", "Amp Type")
    assert g["Crunch Output"] == ("Crunch", "Amp Type") and g["Amp Type"] == (None, None)
    assert g["Cab L Mic Type"] == (None, None) and g["Overdrive Gain"] == (None, None)
    assert len([q for q in p.params if q.kind == "primary"]) == 17


def test_search_only_touches_selected_channel(di):
    """With 'Amp Type' = Clean, no Crunch/Lead knob is suggested."""
    from tonematch.catalog import Catalog
    from tonematch.search import SearchConfig, ToneSearch
    from tonematch.chain import ChainSpec, Renderer, Slot
    from tonematch import features as F
    import optuna
    p = _plini()
    p.format = "mock"   # never rendered in this test
    cat = Catalog({"plini": p})
    s = ToneSearch(cat, Renderer(cat), di, 44100, F.extract(di, 44100), SearchConfig())
    base = ChainSpec([Slot("plini")])
    space = s.space(base, {"primary"})
    assert space[0]["param"] == "Amp Type"          # selectors come first, before grouped knobs
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    seen = set()
    for _ in range(12):
        t = study.ask()
        spec = s._suggest(t, space, base)
        names = set(spec.slots[0].params)
        sel = s._selected_group(0, "Amp Type", {(0, "Amp Type"): spec.slots[0].params["Amp Type"]}, base)
        seen.add(sel)
        for ch in ("Clean", "Crunch", "Lead"):
            assert any(n.startswith(ch + " ") for n in names) == (ch == sel), (sel, names)
        assert "Cab L Mic Type" in names          # ungated knob: always suggested
        study.tell(t, 1.0)
    assert seen == {"Clean", "Crunch", "Lead"}


def test_gates_from_switches():
    p = _plini()
    g = {q.name: q.gates for q in p.params}
    assert g["Overdrive Gain"] == ["Overdrive Active", "Active Pre FX Section"]
    assert g["Overdrive Level"] == ["Overdrive Active", "Active Pre FX Section"]
    assert g["Compressor Threshold"] == ["Compressor Active", "Active Pre FX Section"]
    assert g["Overdrive Active"] == ["Active Pre FX Section"]
    assert "Active EQ Section" in g["Clean EQ 65 Hz"] and "Clean EQ Active" in g["Clean EQ 65 Hz"]
    assert g["Clean EQ Active"] == ["Active EQ Section"]
    assert g["Clean Gain"] == [] and g["Amp Type"] == [] and g["Cab L Mic Type"] == []
    assert g["Active Pre FX Section"] == []


def test_search_skips_knobs_behind_off_switches(di):
    from tonematch.catalog import Catalog
    from tonematch.search import SearchConfig, ToneSearch
    from tonematch.chain import ChainSpec, Renderer, Slot
    from tonematch import features as F
    import optuna
    p = _plini()
    p.format = "mock"
    cat = Catalog({"plini": p})
    s = ToneSearch(cat, Renderer(cat), di, 44100, F.extract(di, 44100), SearchConfig())
    base = ChainSpec([Slot("plini")])
    space = s.space(base, {"primary", "secondary"})
    order = [d["param"] for d in space]
    assert order.index("Overdrive Active") < order.index("Overdrive Gain")
    assert order.index("Active Pre FX Section") < order.index("Overdrive Active")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    seen_on = seen_off = False
    for _ in range(16):
        t = study.ask()
        prm = s._suggest(t, space, base).slots[0].params
        od_on = prm.get("Overdrive Active", 0.0) >= 0.5 and prm.get("Active Pre FX Section", 0.0) >= 0.5
        assert ("Overdrive Gain" in prm) == od_on and ("Overdrive Level" in prm) == od_on
        eq_on = prm.get("Active EQ Section", 0.0) >= 0.5 and prm.get("Clean EQ Active", 0.0) >= 0.5
        if s._selected_group(0, "Amp Type", {(0, "Amp Type"): prm["Amp Type"]}, base) == "Clean":
            assert ("Clean EQ 65 Hz" in prm) == eq_on
        seen_on |= od_on
        seen_off |= not od_on
        study.tell(t, 1.0)
    assert seen_on and seen_off
