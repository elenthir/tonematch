import numpy as np

from tonematch.audio import SR, active_rms_db, excerpt, normalize_active_rms, synth_di, trim_silence
from tonematch import features as F


def test_synth_di_is_sane(di):
    assert di.dtype == np.float32 and len(di) == 6 * SR
    assert abs(active_rms_db(di, SR) + 18.0) < 0.5
    assert np.max(np.abs(di)) < 1.0


def test_normalize_and_trim():
    x = np.zeros(SR * 3, dtype=np.float32)
    x[SR: 2 * SR] = np.sin(np.arange(SR) * 2 * np.pi * 220 / SR) * 0.01
    y = normalize_active_rms(x, SR, -18.0)
    assert abs(active_rms_db(y, SR) + 18.0) < 0.5
    t = trim_silence(y, SR)
    assert SR * 0.9 < len(t) < SR * 1.3


def test_excerpt_picks_loud_part(di):
    e = excerpt(di, SR, 2.0)
    assert len(e) == 2 * SR
    assert active_rms_db(e, SR) >= active_rms_db(di, SR) - 1.0


def test_features_level_invariant(di):
    a = F.extract(di, SR)
    b = F.extract(di * 0.1, SR)
    total, terms = F.distance(a, b)
    assert total < 0.05, terms


def test_distance_zero_for_identical_and_positive_for_distorted(di):
    a = F.extract(di, SR)
    total, _ = F.distance(a, a)
    assert total == 0.0
    b = F.extract(np.tanh(di * 30), SR)
    total, terms = F.distance(a, b)
    assert total > 5 and terms["ltas"] > 1


def test_profile_classes(di):
    clean = F.TargetProfile.from_features(F.extract(di, SR))
    hot = F.TargetProfile.from_features(F.extract(np.tanh(di * 40), SR))
    assert clean.gain_class == "clean"
    assert hot.gain_class == "high"


def test_aligned_distance_finds_lag():
    rng = np.random.default_rng(0)
    x = np.zeros(SR * 4, dtype=np.float32)
    for s in rng.uniform(0, 3.5, 12):            # non-periodic noise bursts
        a = int(s * SR)
        x[a: a + SR // 8] += rng.normal(0, 0.1, SR // 8).astype(np.float32) * rng.uniform(0.3, 1.0)
    shifted = np.concatenate([np.zeros(int(0.3 * SR), dtype=np.float32), x])
    lag = F.align_lag(shifted, x, SR)
    assert abs(lag - int(0.3 * SR)) <= int(0.02 * SR)
    assert F.aligned_distance(shifted, x, SR) < 1.0
