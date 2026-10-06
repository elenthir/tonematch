# Handoff: state of tonematch on the Windows machine

Start a Claude Code session in this folder and tell it: **"Read HANDOFF.md and continue."**

## Verified on this machine (Oct 2026)

- `tonematch scan`: all 27 plugins load (only the MIDI-only Chord Analyzer fails, as it should).
  Neural DSP guitar suites are `amp_suite`, Darkglass/Parallax `bass`, NeuralAmpModeler
  `nam_player`, instruments / utilities / modulation pedals set aside.
- Knob rules tuned on real parameter names (Plini X, Mark IIC+): channel selector ("Amp Type")
  groups per-channel knobs, section / pedal switches gate their knobs, EQ bands secondary.
- `selftest --plugin "Plini X"` (hidden tone found back: 2.19 vs floor 1.96), `"Mark IIC+"`
  (2.79 vs 1.46), real plugins render ~1 candidate/s.
- TONE3000: login, search, fetch work (`~/.tonematch/tone3000.json` holds the tokens).
  Captures run natively at ~10 renders/s; `selftest --plugin "5150 II"` recovers capture + IR
  (1.43 vs floor 1.45) and exports both into `reaper/nam/`.
- Test suite: 39 passed on Windows.

## Not yet verified (needs the user)

1. **REAPER script**: run any `runs\<run>\reaper\apply_tone.lua` as a ReaScript action on a
   track and check the console: plugin added by name, `.vstpreset` loaded via
   `TrackFX_SetPreset`, knobs set by name, NAM's `Input` / `ToneStack` / `NoiseGateActive` /
   `IRToggle` set. Fix `tonematch/export.py` / `tonematch/reaper/apply_tone_template.lua` if not.
2. **A real match**: a dry DI (20–45 s, the style of the part) + a song excerpt:
   `tonematch run --target Song.mp3 --separate --start S --duration 25 --di di.wav --minutes 30 --workers 2 --out runs\song`
   (add `--tone3000 "<amp>"` to pull captures first). Judge `best.wav` vs `target.wav`.
3. Demucs separation has not been run here yet (`uv pip install demucs` is done).

Conventions: keep changes minimal and tested (`uv run --group dev pytest -q`), commit to `main`,
no model identifiers in commits, ask before anything destructive.
