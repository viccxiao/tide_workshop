"""Unit tests for the model quality checks in pipeline/checks.py.

These run anywhere (laptop or CI runner), with no Databricks connection:
    pip install pytest pandas numpy
    pytest cicd/tests
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
from checks import (check_beats_champion, check_input_data, check_minimum_quality,  # noqa: E402
                    check_sanctions_hit_raises_risk, recall_at_review_rate,
                    average_precision)


@pytest.fixture
def labels_and_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.07).astype(int)
    good = np.clip(y * 0.6 + rng.normal(0.2, 0.15, len(y)), 0, 1)   # informative scores
    random = rng.random(len(y))                                         # guessing
    return y, good, random


def test_recall_perfect_model():
    y = np.array([0] * 95 + [1] * 5)
    assert recall_at_review_rate(y, y.astype(float)) == 1.0


def test_good_model_beats_random(labels_and_scores):
    y, good, random = labels_and_scores
    assert check_beats_champion("t", y, good, random).passed
    assert not check_beats_champion("t", y, random, good).passed


def test_tolerance_allows_small_drop(labels_and_scores):
    y, good, _ = labels_and_scores
    slightly_worse = good * 0.999
    assert check_beats_champion("t", y, slightly_worse, good, tolerance=0.02).passed


def test_minimum_quality_blocks_guessing(labels_and_scores):
    y, good, random = labels_and_scores
    assert check_minimum_quality(y, good).passed
    assert not check_minimum_quality(y, random).passed


def test_sanctions_check():
    before = np.array([0.1, 0.2, 0.3])
    assert check_sanctions_hit_raises_risk(before, before + 0.2).passed
    assert not check_sanctions_hit_raises_risk(before, before - 0.05).passed


def test_data_contract():
    ok = pd.DataFrame({"customer_id": ["C1", "C2"], "avg_face_match_score": [0.9, 0.5], "pep_hits": [0, 1]})
    assert check_input_data(ok).passed
    bad = pd.DataFrame({"customer_id": ["C1", "C1"], "avg_face_match_score": [1.4, 0.5], "pep_hits": [-1, 0]})
    result = check_input_data(bad)
    assert not result.passed
    assert "duplicate" in result.detail and "outside 0-1" in result.detail and "negative" in result.detail


def test_average_precision_perfect_and_worst():
    y = np.array([1, 1, 0, 0])
    assert average_precision(y, [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert average_precision(y, [0.1, 0.2, 0.8, 0.9]) < 0.6


def test_sanctions_check_needs_average_increase():
    before = np.zeros(100)
    assert not check_sanctions_hit_raises_risk(before, before).passed   # nothing changed at all
