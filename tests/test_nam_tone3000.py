import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tonematch.audio import SR, synth_di
from tonematch.catalog import Catalog, scan_nam_library, IR_LOADER_ID
from tonematch.chain import ChainSpec, Renderer, Slot
from tonematch.namengine import load_nam, read_metadata, is_legacy_config
from tonematch.search import SearchConfig, propose_chains

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("name,ref", [("wavenet", "ref_wavenet_out"), ("standard_random", "ref_standard_out"),
                                      ("gated_random", "ref_gated_out")])
def test_legacy_wavenet_matches_reference(name, ref):
    """Bit-for-bit against the official (old) neural-amp-modeler package's output."""
    m = load_nam(DATA / f"{name}.nam")
    x = np.load(DATA / "ref_in.npy")
    y = m._net(torch.as_tensor(x))[0].numpy()
    r = np.load(DATA / f"{ref}.npy")[0]
    assert np.abs(y - r).max() < 1e-6


def test_legacy_lstm_loads_and_runs():
    m = load_nam(DATA / "lstm.nam")
    y = m.process(np.random.default_rng(0).normal(0, 0.1, 4410).astype(np.float32), 44100)
    assert y.shape == (4410,) and np.all(np.isfinite(y))


def test_process_resamples_and_keeps_length():
    m = load_nam(DATA / "standard_random.nam")       # 48 kHz model
    x = synth_di(SR, 1.0)
    y = m.process(x, SR)
    assert y.shape == x.shape and np.all(np.isfinite(y)) and np.abs(y).max() > 0
    assert is_legacy_config(json.loads((DATA / "standard_random.nam").read_text()))
    meta = read_metadata(DATA / "wavenet.nam")
    assert meta["gear_type"] == "amp" and meta["gear_make"] == "Darkglass Electronics"


@pytest.fixture
def library(tmp_path):
    import soundfile as sf
    nam = tmp_path / "nam"
    (nam / "1-plexi").mkdir(parents=True)
    (nam / "1-plexi" / "plexi.nam").write_bytes((DATA / "standard_random.nam").read_bytes())
    (nam / "1-plexi" / "plexi.nam.json").write_text(json.dumps({"title": "Plexi crunch", "gear": "amp", "tags": ["crunch"], "downloads_count": 100}))
    (nam / "2-rig").mkdir()
    (nam / "2-rig" / "rig.nam").write_bytes((DATA / "gated_random.nam").read_bytes())
    (nam / "2-rig" / "rig.nam.json").write_text(json.dumps({"title": "5150 rig", "gear": "amp-cab", "tags": ["high gain"]}))
    ir = tmp_path / "ir"
    ir.mkdir()
    t = np.arange(4410) / 44100
    for i, f in enumerate((150, 3000)):
        sf.write(ir / f"ir{i}.wav", (np.exp(-t / 0.01) * np.sin(2 * np.pi * f * t)).astype(np.float32), 44100)
    return nam, ir


def test_library_scan_roles_and_chains(library):
    nam, ir = library
    cat = Catalog(scan_nam_library([nam], [ir]))
    roles = {p.name: p.role for p in cat.plugins.values()}
    assert roles["NAM: Plexi crunch"] == "amp"
    assert roles["NAM: 5150 rig"] == "amp_suite"
    assert cat.plugins[IR_LOADER_ID].role == "cab" and cat.plugins[IR_LOADER_ID].param("IR").n_values == 2
    keys = [c.key for c in propose_chains(cat, SearchConfig(), gain_class="high")]
    assert "nam:2-rig/rig.nam" in keys and "nam:1-plexi/plexi.nam > irloader" in keys
    assert keys[0] == "nam:2-rig/rig.nam"          # the high-gain tagged capture is ordered first


def test_nam_chain_renders_and_exports(library, tmp_path):
    nam, ir = library
    cat = Catalog(scan_nam_library([nam], [ir]))
    r = Renderer(cat)
    di = synth_di(SR, 2.0)
    spec = ChainSpec([Slot("nam:1-plexi/plexi.nam", {"Input Gain": 0.75}), Slot(IR_LOADER_ID, {"IR": 1.0})])
    y = r.render(spec, di, SR)
    assert y.shape == di.shape and np.all(np.isfinite(y)) and np.abs(y).max() > 0
    y2 = r.render(ChainSpec([Slot("nam:1-plexi/plexi.nam", {"Input Gain": 0.25}), Slot(IR_LOADER_ID, {"IR": 0.0})]), di, SR)
    assert not np.allclose(y, y2)
    desc = r.describe(spec)
    assert desc[0]["params"][0]["display"] == "+9.0 dB"
    assert desc[1]["params"][0]["display"] == "ir1.wav"
    from tonematch.export import write_reaper_script
    out = write_reaper_script(tmp_path / "run", r, spec)
    lua = out.read_text()
    assert "01_plexi.nam" in lua and "02_ir1.wav" in lua and '{"Input", 0.725000}' in lua
    assert '{"ToneStack", 0.000000}' in lua and '{"NoiseGateActive", 0.000000}' in lua and '{"IRToggle", 1.000000}' in lua
    assert (tmp_path / "run/reaper/nam/01_plexi.nam").exists()
    lupa = pytest.importorskip("lupa")
    lupa.LuaRuntime().compile(lua)


# --------------------------------------------------------------------------- TONE3000 client
class FakeT3K:
    """Minimal stand-in for the API, driven through the client's transport hook."""

    def __init__(self):
        self.calls = []
        self.expired_once = False

    def __call__(self, method, url, data, headers):
        from tonematch.tone3000 import Response
        self.calls.append((method, url, data, headers))
        if url.endswith("/oauth/token"):
            form = dict(x.split("=") for x in data.decode().split("&"))
            assert form["grant_type"] in ("authorization_code", "refresh_token")
            tok = "fresh" if form["grant_type"] == "refresh_token" else "first"
            return Response(200, json.dumps({"access_token": tok, "refresh_token": "r1", "expires_in": 3600}).encode())
        auth = headers.get("Authorization")
        if auth == "Bearer stale" and not self.expired_once:
            self.expired_once = True
            return Response(401, b"expired")
        if "/tones/search" in url:
            assert "architecture=1" in url
            if "format=ir" in url:
                return Response(200, json.dumps({"data": [], "page": 1, "total_pages": 1, "total": 0}).encode())
            return Response(200, json.dumps({"data": [
                {"id": 42, "title": "Plexi 1959", "gear": "amp", "url": "https://t3k/42", "downloads_count": 999,
                 "makes": [{"name": "Marshall"}], "tags": [{"name": "crunch"}], "user": {"username": "bob"}},
                {"id": 43, "title": "No models", "gear": "amp"}], "page": 1, "total_pages": 1, "total": 2}).encode())
        if "/models" in url:
            if "tone_id=43" in url:
                return Response(200, json.dumps({"data": []}).encode())
            return Response(200, json.dumps({"data": [
                {"id": 1, "name": "Plexi nano", "size": "nano", "model_url": "https://cdn/nano"},
                {"id": 2, "name": "Plexi standard", "size": "standard", "model_url": "https://cdn/std"}]}).encode())
        if url == "https://cdn/std":
            return Response(200, (DATA / "wavenet.nam").read_bytes())
        return Response(404, b"nope")


def test_tone3000_fetch_and_refresh(tmp_path):
    from tonematch.tone3000 import Tone3000, Tokens
    fake = FakeT3K()
    c = Tone3000(Tokens("stale", "r1", expires_at=9e12, client_id="t3k_pub_x"), transport=fake, token_file=tmp_path / "tok.json")
    files = c.fetch("plexi", limit=5, nam_dir=tmp_path / "nam", ir_dir=tmp_path / "ir", log=lambda s: None)
    assert len(files) == 1 and files[0].name == "Plexi_standard.nam"
    side = json.loads((files[0].parent / (files[0].name + ".json")).read_text())
    assert side["make"] == "Marshall" and side["tags"] == ["crunch"] and side["size"] == "standard"
    assert c.tokens.access_token == "fresh"            # 401 → refresh → retry
    assert json.loads((tmp_path / "tok.json").read_text())["access_token"] == "fresh"
    cat = Catalog(scan_nam_library([tmp_path / "nam"], [tmp_path / "ir"]))
    (p,) = cat.plugins.values()
    assert p.role == "amp" and p.extra["make"] == "Marshall" and "Plexi 1959" in p.name
    n_calls = len(fake.calls)
    c.fetch("plexi", limit=5, nam_dir=tmp_path / "nam", ir_dir=tmp_path / "ir", log=lambda s: None)
    assert not any("cdn" in u for _, u, _, _ in fake.calls[n_calls:])


def test_tone3000_pkce_and_urls():
    from tonematch.tone3000 import Tone3000, REDIRECT_URI
    v, ch = Tone3000.pkce()
    assert len(v) >= 43 and "=" not in ch
    url = Tone3000.authorize_url("t3k_pub_abc", ch, "st8")
    assert url.startswith("https://www.tone3000.com/api/v1/oauth/authorize?") and "code_challenge_method=S256" in url
    assert "client_id=t3k_pub_abc" in url and "redirect_uri=http%3A%2F%2Flocalhost%3A3927%2Fcallback" in url
    assert REDIRECT_URI == "http://localhost:3927/callback"


def test_cli_parses_tone3000_and_run_flags():
    from tonematch.cli import build_parser
    a = build_parser().parse_args(["tone3000", "fetch", "5150", "--limit", "3", "--sizes", "lite"])
    assert a.t3k_cmd == "fetch" and a.sizes == ["lite"]
    a = build_parser().parse_args(["run", "--target", "t.wav", "--out", "o", "--tone3000", "plexi", "--nam-dir", "x"])
    assert a.tone3000 == "plexi" and a.nam_dir == ["x"] and a.max_prescreen == 300


def test_full_rig_captures_filed_as_amp_become_suites():
    from tonematch.catalog import nam_role
    assert nam_role("amp") == "amp"
    assert nam_role("amp", "Full Rig Peavey 5150 + Mesa 4x12") == "amp_suite"
    assert nam_role("amp", "5150 Green NAM Profiles + SD1 + Mesa V30 Full_Rig_5150_Green_SD1_Scooped_Mesa_OS_SM57-58") == "amp_suite"
    assert nam_role("amp", "Peavey - 5150 II (0.5.2) PEAVEY_-_5150_II_-_CRUNCH_-_B1_-_G4.0 1990s boost") == "amp"
    assert nam_role("amp", "APP-EVH-5150III-Stealth-100w APP-EVH-Stealth100-Dialled high gain") == "amp"
    assert nam_role("amp-cab", "anything") == "amp_suite" and nam_role("pedal", "Full rig") == "drive"
