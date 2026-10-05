# tonematch — reproduce the guitar tone of a record with your own plugins, hands-off

Give it a song (or an already-isolated guitar track) and a DI recording of your guitar.
It extracts the guitar from the mix, scans your plugins (Neural DSP & friends), works out which
chains are worth trying, and then spends the time budget you give it (say 30 minutes) iterating
*plugin choice → knob settings → render → compare → adjust*. When you come back there is a
`best.md` with every knob value, a `best.wav` you can A/B against the target, a spectrum plot,
and a ReaScript that recreates the chain on a REAPER track.

```
                 song.mp3 ──demucs──► guitar stem ──► target tone descriptors
                                                                │
  your DI.wav ──► plugin chain (pedalboard-hosted VST3) ──► rendered ──► distance ──► optimizer
                      ▲                                                                 │
                      └──────────────── new knob values / new chain ◄───────────────────┘
```

Rendering is done by hosting the VST3 plugins directly in Python (Spotify's `pedalboard`), not
by driving REAPER: a 10 s render through an amp sim takes a fraction of a second, so a 30-minute
run is hundreds to thousands of evaluations instead of a handful. REAPER only comes in at the end.

## Setup (on the machine that has the plugins)

```
cd tonematch
uv venv && uv pip install -e .            # python 3.10+, mac / windows / linux
uv pip install demucs                     # optional: guitar stem separation (needs torch, ~2 GB)
uv pip install yt-dlp                     # optional: pass a YouTube URL as the target
```

Neural DSP (and any iLok-licensed) plugins work as long as the licence is active on the machine —
the plugin checks the licence itself, whichever host loads it.

**Windows + WSL2:** run tonematch on *Windows* (PowerShell), not inside WSL2. The renderer hosts
the plugins in-process, and a Linux Python can only load Linux VST3 builds — Windows plugin DLLs
under `/mnt/c/...` are invisible to it. Every dependency has Windows wheels:

```
winget install astral-sh.uv
cd tonematch
uv venv ; uv pip install -e . ; uv pip install demucs
.venv\Scripts\tonematch scan
```

If you prefer a WSL shell, keep the checkout on the Windows side and call the Windows
interpreter from WSL (`/mnt/c/.../tonematch/.venv/Scripts/tonematch.exe …`). Either way the
generated REAPER script then carries Windows paths for the `.vstpreset` files, which is what
REAPER needs.

### 1. Scan your plugins (once)

```
tonematch scan                                   # standard VST3 / AU folders
tonematch scan "C:\Program Files\Common Files\VST3"    # or explicit folders / bundles
tonematch list                                   # roles and knob kinds per plugin
tonematch list "Nolly"                           # every knob of one plugin and how it is treated
```

Each plugin is loaded in a subprocess (a crashing plugin cannot kill the scan), its parameters
are read (name, type, discrete values, default), and the knob map tags them (see below).
The catalog lives in `~/.tonematch/catalog.json`. Re-run `scan` after installing plugins;
`--rescan` re-probes everything.

### 2. Record a DI

In REAPER: record your guitar **dry** (interface Hi-Z input, no plugins) playing something in the
style of the part you want to match — ideally the actual riff, 15–45 s, with the same pickup
you'd use. Render that item to a wav (`File → Render`, or right-click item → *Render items as
new take*). The DI does **not** have to be the same performance as the record, the comparison is
performance-agnostic. If it *is* the same riff played in time, add `--aligned` for a tighter match.

### 3. Run it

```
tonematch run --target "Song.mp3" --separate --start 61 --duration 25 \
              --di my_di.wav --minutes 30 --out runs/song_riff
```

* `--target` is a song file (with `--separate` to pull the guitar out with Demucs' 6-stem model),
  a URL (downloaded then separated), or an already isolated guitar track (a stem you have, a
  re-amp track, a plugin demo…). `--start/--duration` pick the section — choose 15–40 s where
  the guitar you want is exposed and representative (one tone, not the clean intro *and* the
  solo).
* `--di` your dry recording. Without it a synthetic DI is used — fine for trying the tool out,
  useless for a real match.
* `--minutes` the wall-clock budget. The run ends on time whatever happens.
* `--workers 2..4` renders candidates in parallel threads (one plugin instance per thread).
  Amp sims are CPU heavy; try 2 on a laptop.
* `--include nolly --include gojira` restricts chains to plugins whose name contains these;
  `--exclude` the opposite; `--chain "Archetype Nolly X" "ReaEQ"` pins an exact chain and
  skips chain selection entirely.
* `--allow-ambience` also searches reverb / delay / modulation knobs (off by default: a stem
  from a mix carries the mix reverb, which you usually don't want baked into the amp preset).
* `--resume` keeps the optimizer's trial database in the run folder so a second run with the
  same `--out` continues where it stopped.

Progress is printed as it goes, and every improvement is checkpointed: `best.json`, `best.md`
and `reaper/apply_tone.lua` are always current, `best.wav` at most every 30 s. Kill it any time.

Output of a run:

| file | content |
|---|---|
| `best.md` | the chain and every knob value (plugin's own display string + normalised 0..1) |
| `best.json` | the same, machine-readable, plus the search history and chain ranking |
| `best.wav`, `target.wav`, `di.wav` | A/B material: your DI through the found chain, the target excerpt, the raw DI |
| `report.png` | long-term spectrum target vs match (with p10–p90 bands), loss over time |
| `reaper/apply_tone.lua` | ReaScript that builds the chain on the selected track |
| `reaper/*.vstpreset` | full plugin state per slot (VST3 preset format) |

### 4. Load it in REAPER

Select the guitar track, `Actions → Show action list → New action → Load ReaScript…`, pick
`runs/song_riff/reaper/apply_tone.lua`, run it. For each plugin it adds the VST3 by name, loads
the `.vstpreset` (whole state), then sets every matched knob by name as a cross-check and
reports in the ReaScript console what it could not find. Output level is deliberately not
matched — set it to taste. If REAPER names a plugin differently from what the script tried, add
the plugin to the track by hand first and re-run the script: it re-uses what's on the track.

## How the search works

**Target description.** Both signals are loudness-normalised and only the active frames are
used. The descriptors are distributional, so two different performances of similar material
compare sensibly: long-term average log-mel spectrum with level removed (voicing, cab, mic),
its 10th/90th percentile curves (how the spectrum moves with dynamics), quantiles of frame-wise
spectral centroid / flatness / rolloff / HF ratio (amount of saturation, fizz), quantiles of the
RMS envelope (compression / sustain) and MFCC statistics. The weighted sum is in dB-like units;
a few dB is a close match. From the target's flatness / HF energy / envelope the tool also
guesses a gain class (clean / crunch / high) that sets the starting range of gain-type knobs.

**Candidate chains.** From plugin roles: every all-in-one suite (an Archetype is a complete
chain by itself) on its own, each standalone amp with each cab/IR loader, then the same with a
drive pedal in front, then with a post-EQ. Capped at `--max-chains` with at most 3 chains per
amp, so one big suite doesn't crowd out the others.

**Three stages, one deadline.**

1. *Screen* (~22 % of the time) — every chain gets a short Optuna/TPE run over its *primary*
   knobs (gain, tone stack, presence, amp/cab/mic selectors, section on/off switches) on a 6 s
   excerpt of the DI, seeded with "everything at noon, gain from the gain class" and the plugin
   defaults. Chains are ranked by their best loss.
2. *Optimize* (~53 %) — the best `--keep-chains` survive. Successive halving: every survivor gets
   a time slice over primary + secondary knobs on a 10 s excerpt, the worse half is dropped,
   repeat. Each chain keeps its own study, so what was learnt in screening is reused.
3. *Refine* (~25 %) — the winner only: discrete choices are frozen, continuous knobs are
   narrowed to ±0.12 around the incumbent and polished with CMA-ES on a 20 s excerpt.

Finally the incumbent is scored on up to 40 s of DI, and everything is written out.

## Knob maps (and reusing your own)

`tonematch/knobs/default.yaml` decides, by regex on plugin and parameter names,

* the **role** of a plugin (`amp_suite`, `amp`, `cab`, `drive`, `eq`, …) → which chains get built,
* the **kind** of each parameter: `primary` (searched always), `secondary` (searched once a chain
  survives screening), `ambience` (frozen off unless `--allow-ambience`), `fixed` (set to a value
  and left alone — e.g. Neural DSP's noise gate on, doubler/transpose off), `excluded` (input/output
  levels, transpose, tuner, mix knobs, presets, …).

If you have already mapped the knobs of your plugins by hand, put that knowledge in your own YAML
and pass it with `--knobs my_knobs.yaml` (repeatable; later files win, exact names work as well
as regexes):

```yaml
plugins:
  - match: "Nolly"                 # plugin name regex
    role: amp_suite
    excluded: ["Amp 1 .*"]         # only search amp 2
    fixed:    {"Amp Select": 1.0}  # normalised 0..1
    ranges:   {"Amp 2 Gain": [0.3, 0.8]}
    primary:  ["Cab 1 Mic 1 Position"]
    ambience: ["Pedal 3 .*"]
```

`tonematch list "<plugin>"` shows how every knob ended up classified, which is the quickest way
to check a map. Discrete parameters with more than 64 values (IR lists) are searched only when
tagged primary.

## Trying it without any plugins

```
tonematch selftest --minutes 2
```

hides a random tone made with built-in mock plugins (amp with gain/tone stack/cab/mic, drive,
EQ, reverb — all built from `pedalboard`'s native effects), renders it through a *different*
performance, and tries to find it back from the DI. Prints the hidden vs found knobs and the
loss (defaults → found → "hidden chain itself"). `tonematch run --mock …` uses the same mocks
with real audio. Tests: `uv run --group dev pytest`.

## Limits worth knowing

* Only *parameters* are matched. State that is not an automatable parameter (an IR loaded from
  a file in a third-party IR loader, a Neural DSP preset's "captured" blocks) travels through the
  `.vstpreset` when REAPER can load it, not through the knob list.
* The guitar stem of a dense mix is an approximation (bleed, mix EQ, reverb). Pick a section
  where the guitar is exposed; the loss weights `pct`/`dyn` can be lowered with
  `--weight dyn=0.1` if the stem's dynamics are clearly not the guitar's.
* Two different amp settings can sound alike on the descriptors — the run returns *a* chain that
  matches the descriptors, not necessarily the one on the record. `best.wav` vs `target.wav` is
  the judge.
* This was developed and tested with mock plugins on Linux (no commercial plugins here). The
  VST3 hosting path uses `pedalboard`'s documented API but has not been run against Neural DSP
  plugins in this environment — the first `tonematch scan` on your machine is the real test.
