"""Regression tests for BankSampler._split_mask: hash contract and speed.

Background: _split_mask once used pandas.iterrows(), costing ~1.3s per call on
a 55k-row partition. It is re-paid on every band-iterator rebuild, i.e. during
episode resets, and throttled bank-backed PPO training ~3x. The fix must keep
split membership byte-identical (fingerprints are contracts), so both the hash
behaviour and the scaling are pinned here.
"""

import hashlib
import time

import pandas as pd

from src.env.puzzle_bank import BankSampler, PuzzleBank

SPLIT_SEED = 20_260_710
EVAL_FRACTION = 0.25


def _sampler(tmp_path, split: str) -> BankSampler:
    bank = PuzzleBank(tmp_path / "bank")
    return BankSampler(
        bank,
        seed=0,
        split=split,
        eval_fraction=EVAL_FRACTION,
        split_seed=SPLIT_SEED,
        split_key="layout_hash",
    )


def test_split_mask_hash_contract(tmp_path) -> None:
    """Membership must equal sha256(f"{split_seed}:{value}") thresholding exactly."""
    values = [f"layout-{i:04d}" for i in range(256)]
    df = pd.DataFrame({"layout_hash": values})
    eval_mask = _sampler(tmp_path, "eval")._split_mask(df)
    train_mask = _sampler(tmp_path, "train")._split_mask(df)
    for value, is_eval, is_train in zip(values, eval_mask.tolist(), train_mask.tolist()):
        digest = hashlib.sha256(f"{SPLIT_SEED}:{value}".encode("utf-8")).digest()
        expected_eval = int.from_bytes(digest[:8], "big") / float(2**64) < EVAL_FRACTION
        assert is_eval == expected_eval
        assert is_train == (not expected_eval)
    # Both splits must partition the rows.
    assert (eval_mask ^ train_mask).all()
    # Sanity: the fraction lands near eval_fraction for this fixed vocabulary.
    assert 0.15 < eval_mask.mean() < 0.35


def test_split_mask_scales_to_large_partitions(tmp_path) -> None:
    """55k rows must be masked in well under half a second (iterrows took ~1.3s)."""
    df = pd.DataFrame({"layout_hash": [f"layout-{i}" for i in range(55_000)]})
    sampler = _sampler(tmp_path, "eval")
    started = time.perf_counter()
    mask = sampler._split_mask(df)
    elapsed = time.perf_counter() - started
    assert len(mask) == len(df)
    assert elapsed < 0.5, f"_split_mask took {elapsed:.2f}s on 55k rows — reset-path regression"
