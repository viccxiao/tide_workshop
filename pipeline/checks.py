"""Model quality checks used by the deployment job (pipeline/evaluate) and by the unit tests (cicd/tests).

Each check returns a CheckResult. A model version is only allowed to go live if every check passes.
Plain pandas/numpy only, so the same code runs in a Databricks job and in a CI runner.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

    def as_tag(self):
        return ("✅ " if self.passed else "❌ ") + self.detail


def recall_at_review_rate(y_true, scores, review_rate=0.05):
    """Share of high-risk customers caught if analysts review the riskiest `review_rate` of sign-ups."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    if y_true.sum() == 0:
        return float("nan")
    threshold = np.quantile(scores, 1 - review_rate)
    return float((scores[y_true == 1] >= threshold).mean())


def average_precision(y_true, scores):
    """PR-AUC: how well the model ranks high-risk customers above the rest, across all review rates."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = y_true[order] == 1
    precision_at_hit = np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1)
    return float(precision_at_hit.mean())


def check_input_data(features: pd.DataFrame) -> CheckResult:
    """Data contract: keys present, values in sensible ranges."""
    problems = []
    if features["customer_id"].isna().any():
        problems.append("missing customer_id")
    if features["customer_id"].duplicated().any():
        problems.append("duplicate customer_id")
    for col in ("avg_face_match_score", "min_face_match_score", "doc_fail_rate", "max_screening_match_score"):
        if col in features and not features[col].between(0, 1).all():
            problems.append(f"{col} outside 0-1")
    for col in ("company_age_months", "doc_attempts", "pep_hits", "sanctions_hits", "adverse_media_hits"):
        if col in features and (features[col] < 0).any():
            problems.append(f"negative {col}")
    passed = not problems
    return CheckResult("data_contract", passed,
                       f"{len(features):,} rows checked, no issues" if passed else "; ".join(problems))


def check_beats_champion(name, y_true, candidate_scores, champion_scores, tolerance=0.0) -> CheckResult:
    """Candidate must rank high-risk customers at least as well as the champion (PR-AUC, minus a tolerance)."""
    cand, champ = average_precision(y_true, candidate_scores), average_precision(y_true, champion_scores)
    passed = cand >= champ - tolerance
    return CheckResult(name, passed, f"PR-AUC candidate {cand:.3f} vs champion {champ:.3f}"
                                     + (f" (tolerance {tolerance})" if tolerance else ""))


def check_sanctions_hit_raises_risk(scores_before, scores_after, min_share=0.95) -> CheckResult:
    """Behaviour test: adding a sanctions hit should raise risk for (almost) every customer, and on average."""
    before, after = np.asarray(scores_before, dtype=float), np.asarray(scores_after, dtype=float)
    share_ok = float((after >= before - 1e-9).mean())
    avg_change = float((after - before).mean())
    passed = share_ok >= min_share and avg_change > 0
    return CheckResult("sanctions_hit_raises_risk", passed,
                       f"score went up or stayed level for {share_ok:.1%} of customers "
                       f"(need {min_share:.0%}), average change {avg_change:+.3f}")


def check_minimum_quality(y_true, scores, min_recall=0.5) -> CheckResult:
    """Absolute floor: whatever the champion does, never ship a model below this."""
    r = recall_at_review_rate(y_true, scores)
    return CheckResult("minimum_quality", r >= min_recall, f"recall at 5% review {r:.1%} (floor {min_recall:.0%})")
