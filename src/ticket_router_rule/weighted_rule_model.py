"""Core weighted keyword rule model extracted from rule_based_final_optimized.py.

Adapted to work with the ticket_router_base Record/Prediction types
instead of raw pandas DataFrames.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from ticket_router_base.types import Record


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot",
    "could", "customer", "dear", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "hello", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "kindly", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "regards", "same", "she", "should", "so", "some", "such",
    "support", "team", "than", "thank", "thanks", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
    # German stopwords
    "aber", "auch", "bei", "bitte", "das", "dem", "den", "der", "des",
    "die", "doch", "ein", "eine", "einem", "einen", "einer", "eines",
    "es", "fuer", "fur", "geehrte", "geehrter", "guten", "hallo", "ich",
    "ihnen", "ihre", "ihren", "im", "ist", "kein", "keine",
    "kundendienst", "mit", "nicht", "oder", "sehr", "sie", "tag", "und",
    "vielen", "von", "wir", "zu", "zum", "zur",
}


def normalize_text(text: str) -> str:
    """Normalize text: NFKC + lower + German umlaut replacement."""
    normalized = unicodedata.normalize("NFKC", str(text).lower())
    return (
        normalized.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def tokenize(text: str) -> List[str]:
    """Tokenize text into alphabetic tokens, filtering stopwords."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", normalize_text(text))
    return [token for token in tokens if token not in STOPWORDS]


def make_features(record: Record, feature_mode: str) -> List[str]:
    """Extract feature tokens from a Record.

    feature_mode controls which sources to include:
        - "text":  tokenize title + body, plus bigrams and subject-prefixed tokens
        - "meta":  add LANG={language}
        - "tags":  add discrete feature key-value pairs (e.g. TAG_1=..., TYPE=...)

    Multiple modes can be combined with "+" (e.g. "text+meta+tags").
    """
    features: List[str] = []

    if "text" in feature_mode:
        title_tokens = tokenize(record.title or "")
        body_tokens = tokenize(record.body)
        all_tokens = title_tokens + body_tokens
        features.extend(all_tokens)
        features.extend(
            all_tokens[i] + "_" + all_tokens[i + 1]
            for i in range(len(all_tokens) - 1)
        )
        features.extend("subj_" + token for token in title_tokens)

    if "meta" in feature_mode:
        lang = record.language.value if record.language else "unknown"
        features.append(f"LANG={lang}")

    if "tags" in feature_mode:
        for key, value in record.discrete_features.items():
            if value:
                features.append(f"{key.upper()}={value}")
                features.append(f"TAG={value}")

    return sorted(set(features))


@dataclass(frozen=True)
class CandidateConfig:
    """Hyperparameter candidate for rule model training."""

    name: str
    feature_mode: str
    min_count: int
    min_log_odds: float
    max_features: int


@dataclass
class WeightedRuleModel:
    """Log-odds weighted keyword classifier.

    Learns a sparse set of discriminative features per label from training data.
    Prediction is done by summing prior + matched feature weights and picking
    the label with the highest score.
    """

    labels: Sequence[str]
    priors: Dict[str, float]
    weights: Dict[str, Dict[str, float]]

    @classmethod
    def fit(
        cls,
        features_list: List[List[str]],
        labels_list: List[str],
        all_labels: Sequence[str],
        min_count: int,
        min_log_odds: float,
        max_features: int,
    ) -> "WeightedRuleModel":
        """Train a WeightedRuleModel on feature/label pairs.

        Args:
            features_list: List of feature token lists, one per sample.
            labels_list: List of ground-truth labels, one per sample.
            all_labels: Complete ordered list of valid labels.
            min_count: Minimum feature occurrence count within a label to be considered.
            min_log_odds: Minimum log-odds ratio for a feature to be kept.
            max_features: Maximum number of features to retain per label.
        """
        total_docs = len(features_list)
        global_df: Counter[str] = Counter()
        label_df: Dict[str, Counter[str]] = defaultdict(Counter)
        label_docs: Counter[str] = Counter()

        for features, label in zip(features_list, labels_list):
            global_df.update(features)
            label_df[label].update(features)
            label_docs[label] += 1

        priors = {
            label: math.log((label_docs[label] + 1) / (total_docs + len(all_labels)))
            for label in all_labels
        }
        weights: Dict[str, Dict[str, float]] = {label: {} for label in all_labels}

        for label in all_labels:
            docs_in_label = label_docs[label]
            docs_outside = total_docs - docs_in_label
            ranked: List[tuple[float, str, float]] = []

            for feature, count_in_label in label_df[label].items():
                if count_in_label < min_count:
                    continue
                count_outside = global_df[feature] - count_in_label
                prob_label = (count_in_label + 1) / (docs_in_label + 2)
                prob_other = (count_outside + 1) / (docs_outside + 2)
                log_odds = math.log(prob_label / prob_other)

                if log_odds < min_log_odds:
                    continue
                # Score = log_odds * log1p(count) for ranking
                ranked.append((log_odds * math.log1p(count_in_label), feature, log_odds))

            ranked.sort(reverse=True)
            for _, feature, log_odds in ranked[:max_features]:
                weights[label][feature] = log_odds

        return cls(labels=list(all_labels), priors=priors, weights=weights)

    def predict(self, features: Iterable[str]) -> str:
        """Predict the label with the highest score for the given features."""
        feature_set = set(features)
        scores = {label: self.priors[label] for label in self.labels}
        for label in self.labels:
            for feature in feature_set:
                scores[label] += self.weights[label].get(feature, 0.0)
        return max(self.labels, key=lambda label: scores[label])

    def predict_with_scores(self, features: Iterable[str]) -> Tuple[str, Dict[str, float]]:
        """Predict label and return raw scores for all labels."""
        feature_set = set(features)
        scores = {label: self.priors[label] for label in self.labels}
        for label in self.labels:
            for feature in feature_set:
                scores[label] += self.weights[label].get(feature, 0.0)
        best = max(self.labels, key=lambda label: scores[label])
        return best, scores

    def top_rules(self, limit: int = 20) -> Dict[str, List[tuple[str, float]]]:
        """Return top-weighted features per label."""
        output: Dict[str, List[tuple[str, float]]] = {}
        for label in self.labels:
            output[label] = sorted(
                self.weights[label].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
        return output


def accuracy_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute accuracy."""
    return sum(expected == predicted for expected, predicted in zip(y_true, y_pred)) / len(y_true)


def macro_f1_score(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """Compute macro-averaged F1 score."""
    scores = []
    for label in labels:
        tp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == label and pred == label)
        fp = sum(1 for actual, pred in zip(y_true, y_pred) if actual != label and pred == label)
        fn = sum(1 for actual, pred in zip(y_true, y_pred) if actual == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(labels)


def run_candidate_search(
    train_records: List[Record],
    val_records: List[Record],
    task_name: str,
    labels: Sequence[str],
    candidates: Sequence[CandidateConfig],
) -> Tuple[List[dict], CandidateConfig]:
    """Search for the best hyperparameter candidate on a validation set.

    Returns:
        results: List of result dicts for each candidate.
        best: The CandidateConfig with the highest selection_score (accuracy + macro_f1).
    """
    results = []
    for candidate in candidates:
        train_features = [make_features(r, candidate.feature_mode) for r in train_records]
        val_features = [make_features(r, candidate.feature_mode) for r in val_records]
        train_labels = [r.labels.get(task_name, "") for r in train_records]
        val_labels = [r.labels.get(task_name, "") for r in val_records]

        model = WeightedRuleModel.fit(
            train_features,
            train_labels,
            labels,
            min_count=candidate.min_count,
            min_log_odds=candidate.min_log_odds,
            max_features=candidate.max_features,
        )
        predictions = [model.predict(feats) for feats in val_features]
        acc = accuracy_score(val_labels, predictions)
        mf1 = macro_f1_score(val_labels, predictions, labels)
        results.append(
            {
                "task": task_name,
                "candidate_name": candidate.name,
                "feature_mode": candidate.feature_mode,
                "min_count": candidate.min_count,
                "min_log_odds": candidate.min_log_odds,
                "max_features": candidate.max_features,
                "dev_accuracy": acc,
                "dev_macro_f1": mf1,
                "selection_score": acc + mf1,
            }
        )

    results.sort(key=lambda row: row["selection_score"], reverse=True)
    best = next(candidate for candidate in candidates if candidate.name == results[0]["candidate_name"])
    return results, best
