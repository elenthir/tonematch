# Handoff: continue tonematch on the Windows machine

Start a Claude Code session in this folder and tell it: **"Read HANDOFF.md and continue."**

## Context

tonematch matches the guitar tone of a record with the user's own plugins (Neural DSP suites hosted
through Pedalboard, NAM captures run natively, impulse responses) and exports the result to REAPER
as a ReaScript. It was built in a cloud container with **no access to the plugins**: everything
plugin-related has only been exercised with the built-in mock plugins. `README.md` documents the
workflow, the knob-map YAML, the NAM / TONE3000 library and the known limits.

Environment: Windows, REAPER, VST3s under `C:\Program Files\Common Files\VST3` (Neural DSP
Archetypes Cory Wong X / John Mayer X / Mateus Asato / Plini X, Mesa Boogie Mark IIC+ Suite, Tone
King Imperial MKII; Darkglass Ultra and Parallax X are bass; NeuralAmpModeler; UAD plugins; MixWave
pedals; drum/bass instruments). Python via `uv`, venv `.venv`, CLI `.venv\Scripts\tonematch`.
tonematch must run on Windows itself, not WSL2: Pedalboard can only load Windows plugin binaries
from a Windows Python.

## To do, in order (commit to `main` as you go, test suite green before each commit)

1. `git pull`; `uv pip install -e ".[nam]"` (torch for NAM captures); optionally
   `uv pip install demucs`. `uv run --group dev pytest -q` must pass (31 tests, mocks only).
2. `tonematch scan --rescan` then `tonematch list`. The previous scan here had 9 plugins fail with
   "unable to scan" because Windows installs some VST3s as folder bundles
   (`X.vst3\Contents\x86_64-win\X.vst3`); `catalog.resolve_plugin_path` now loads the inner DLL
   but was never run on this machine. Verify NeuralAmpModeler and the `uaudio_*` plugins load (if
   UAD still fails it is probably UA Connect licensing: note it, move on). Check roles: Neural DSP
   guitar suites `amp_suite`, Darkglass/Parallax `bass`, NeuralAmpModeler `nam_player`,
   instruments / utilities set aside.
3. `tonematch list "Plini X"` prints every parameter with its kind (primary / secondary /
   ambience / fixed / excluded). The kinds come from regexes in `tonematch/knobs/default.yaml`
   written **without seeing real Neural DSP parameter names**. Tune them against the real names:
   screening should run over the amp's main knobs (gain, tone stack, presence, amp / cab / mic
   selectors, section on/off switches); EQ bands and mic position / level secondary; delay /
   reverb / modulation ambience; input / output / transpose / doubler / tuner excluded; noise
   gate forced on. Extend `tests/test_catalog_knobmap.py` with real names.
4. TONE3000: `tonematch tone3000 login --client-id t3k_pub_...` got
   `{"error":"invalid_client","error_description":"Unknown client_id"}` from the authorize page
   (which did receive the redirect URI `http://localhost:3927/callback`). The client,
   `tonematch/tone3000.py`, was written from the public docs (github.com/tone-3000/api and the
   tone3000-rs SDK) with no network access to tone3000.com. Find out with the user what the API
   Keys page really shows (separate client id? redirect URI saved on the key? key enabled?), fix
   the client, then verify `tone3000 search "5150"` and `tone3000 fetch "5150" --limit 10` work
   and the captures show up in `tonematch list`. Fix any response-field mismatch.
5. With a real DI (dry guitar, 20–45 s) and a target (song excerpt with `--separate`, or an
   isolated guitar track) run `tonematch run ... --minutes 10 --workers 2 --out runs\first`.
   Watch the real plugin path (`chain._Vst3Handle`: parameters by name, `preset_data`, latency).
   Then in REAPER run `runs\first\reaper\apply_tone.lua`: plugins added by name, `.vstpreset`
   loaded through `TrackFX_SetPreset`, knobs set by name, and the NAM plugin's input knob name
   ("Input" in the script) matches the real parameter. Fix what does not work.
6. Report what works, what changed, and what still needs the user (licensing, manual loads).
