# reference only!!!

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

TAG_COLUMNS = [f"tag_{index}" for index in range(1, 9)]
TYPE_LABELS = ["Change", "Incident", "Problem", "Request"]
URGENCY_LABELS = ["standard", "urgent"]
URGENCY_ORDINAL = {"standard": 0, "urgent": 1}

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "customer",
    "dear",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "hello",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "kindly",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "regards",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "support",
    "team",
    "than",
    "thank",
    "thanks",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "aber",
    "auch",
    "bei",
    "bitte",
    "das",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "doch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "es",
    "fuer",
    "fur",
    "geehrte",
    "geehrter",
    "guten",
    "hallo",
    "ich",
    "ihnen",
    "ihre",
    "ihren",
    "im",
    "ist",
    "kein",
    "keine",
    "kundendienst",
    "mit",
    "nicht",
    "oder",
    "sehr",
    "sie",
    "tag",
    "und",
    "vielen",
    "von",
    "wir",
    "zu",
    "zum",
    "zur",
}


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    feature_mode: str
    min_count: int
    min_log_odds: float
    max_features: int


@dataclass
class WeightedRuleModel:
    labels: Sequence[str]
    priors: Dict[str, float]
    weights: Dict[str, Dict[str, float]]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_column: str,
        label_column: str,
        labels: Sequence[str],
        min_count: int,
        min_log_odds: float,
        max_features: int,
    ) -> "WeightedRuleModel":
        total_docs = len(frame)
        global_df: Counter[str] = Counter()
        label_df: Dict[str, Counter[str]] = defaultdict(Counter)
        label_docs: Counter[str] = Counter()

        for _, row in frame.iterrows():
            features = row[feature_column]
            label = row[label_column]
            global_df.update(features)
            label_df[label].update(features)
            label_docs[label] += 1

        priors = {
            label: math.log((label_docs[label] + 1) / (total_docs + len(labels)))
            for label in labels
        }
        weights: Dict[str, Dict[str, float]] = {label: {} for label in labels}

        for label in labels:
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
                ranked.append(
                    (log_odds * math.log1p(count_in_label), feature, log_odds)
                )

            ranked.sort(reverse=True)
            for _, feature, log_odds in ranked[:max_features]:
                weights[label][feature] = log_odds

        return cls(labels=list(labels), priors=priors, weights=weights)

    def predict(self, features: Iterable[str]) -> str:
        feature_set = set(features)
        scores = {label: self.priors[label] for label in self.labels}
        for label in self.labels:
            for feature in feature_set:
                scores[label] += self.weights[label].get(feature, 0.0)
        return max(self.labels, key=lambda label: scores[label])

    def top_rules(self, limit: int = 20) -> Dict[str, List[tuple[str, float]]]:
        output = {}
        for label in self.labels:
            output[label] = sorted(
                self.weights[label].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
        return output


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text).lower())
    return (
        normalized.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", normalize_text(text))
    return [token for token in tokens if token not in STOPWORDS]


def deterministic_split(text: str) -> str:
    bucket = int(md5(text.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket <= 5:
        return "build"
    if bucket <= 7:
        return "dev"
    return "test"


def make_features(row: pd.Series, feature_mode: str) -> List[str]:
    features: List[str] = []
    if "text" in feature_mode:
        subject_tokens = tokenize(row["subject"])
        body_tokens = tokenize(row["body"])
        all_tokens = subject_tokens + body_tokens
        features.extend(all_tokens)
        features.extend(
            all_tokens[index] + "_" + all_tokens[index + 1]
            for index in range(len(all_tokens) - 1)
        )
        features.extend("subj_" + token for token in subject_tokens)

    if "meta" in feature_mode:
        features.append("LANG=" + str(row["language"]))

    if "tags" in feature_mode:
        for column in TAG_COLUMNS:
            value = str(row[column])
            if value:
                features.append(column.upper() + "=" + value)
                features.append("TAG=" + value)

    return sorted(set(features))


def accuracy_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return sum(
        expected == predicted for expected, predicted in zip(y_true, y_pred)
    ) / len(y_true)


def macro_f1_score(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> float:
    scores = []
    for label in labels:
        tp = sum(
            1
            for actual, pred in zip(y_true, y_pred)
            if actual == label and pred == label
        )
        fp = sum(
            1
            for actual, pred in zip(y_true, y_pred)
            if actual != label and pred == label
        )
        fn = sum(
            1
            for actual, pred in zip(y_true, y_pred)
            if actual == label and pred != label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(scores) / len(labels)


def mae_ordinal(
    y_true: Sequence[str], y_pred: Sequence[str], mapping: Dict[str, int]
) -> float:
    return sum(
        abs(mapping[actual] - mapping[pred]) for actual, pred in zip(y_true, y_pred)
    ) / len(y_true)


def qwk_ordinal(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> float:
    n_labels = len(labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    observed = [[0.0] * n_labels for _ in range(n_labels)]
    for actual, pred in zip(y_true, y_pred):
        observed[label_to_index[actual]][label_to_index[pred]] += 1.0

    actual_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    expected = [[0.0] * n_labels for _ in range(n_labels)]
    sample_size = len(y_true)
    for row_index, actual_label in enumerate(labels):
        for col_index, pred_label in enumerate(labels):
            expected[row_index][col_index] = (
                actual_counts[actual_label] * pred_counts[pred_label] / sample_size
            )

    weights = [
        [
            ((row_index - col_index) ** 2) / ((n_labels - 1) ** 2)
            for col_index in range(n_labels)
        ]
        for row_index in range(n_labels)
    ]
    numerator = sum(
        weights[row_index][col_index] * observed[row_index][col_index]
        for row_index in range(n_labels)
        for col_index in range(n_labels)
    )
    denominator = sum(
        weights[row_index][col_index] * expected[row_index][col_index]
        for row_index in range(n_labels)
        for col_index in range(n_labels)
    )
    return 1.0 - numerator / denominator if denominator else 0.0


def language_metrics(
    frame: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    labels: Sequence[str],
) -> Dict[str, float]:
    grouped = {}
    for language, subset in frame.groupby("language"):
        y_true = subset[truth_column].tolist()
        y_pred = subset[prediction_column].tolist()
        grouped[language] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": macro_f1_score(y_true, y_pred, labels),
            "predictions": y_pred,
        }

    languages = sorted(grouped)
    accuracies = [grouped[language]["accuracy"] for language in languages]
    f1_scores = [grouped[language]["macro_f1"] for language in languages]

    di_values = []
    spd_values = []
    for label in labels:
        rates = []
        for language in languages:
            predictions = grouped[language]["predictions"]
            rates.append(
                sum(prediction == label for prediction in predictions)
                / len(predictions)
            )
        min_rate, max_rate = min(rates), max(rates)
        if max_rate == 0:
            di = 1.0
        elif min_rate == 0:
            di = float("inf")
        else:
            di = max_rate / min_rate
        if math.isfinite(di):
            di_values.append(di)
        spd_values.append(max_rate - min_rate)

    return {
        "language_acc_gap": max(accuracies) - min(accuracies),
        "language_acc_ratio": min(accuracies) / max(accuracies)
        if max(accuracies)
        else 0.0,
        "language_f1_gap": max(f1_scores) - min(f1_scores),
        "language_f1_ratio": min(f1_scores) / max(f1_scores) if max(f1_scores) else 0.0,
        "language_DI": sum(di_values) / len(di_values),
        "language_SPD": sum(spd_values) / len(spd_values),
    }


def evaluate(
    frame: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    labels: Sequence[str],
    ordinal_mapping: Dict[str, int] | None = None,
) -> Dict[str, float | None]:
    y_true = frame[truth_column].tolist()
    y_pred = frame[prediction_column].tolist()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1_score(y_true, y_pred, labels),
        "mae": None,
        "qwk": None,
    }
    if ordinal_mapping is not None:
        metrics["mae"] = mae_ordinal(y_true, y_pred, ordinal_mapping)
        metrics["qwk"] = qwk_ordinal(y_true, y_pred, labels)
    metrics.update(language_metrics(frame, truth_column, prediction_column, labels))
    return metrics


def run_candidate_search(
    build_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    task_name: str,
    labels: Sequence[str],
    candidates: Sequence[CandidateConfig],
) -> tuple[list[dict[str, object]], CandidateConfig]:
    results = []
    for candidate in candidates:
        build_features = build_frame.apply(
            lambda row: make_features(row, candidate.feature_mode), axis=1
        )
        dev_features = dev_frame.apply(
            lambda row: make_features(row, candidate.feature_mode), axis=1
        )
        build_copy = build_frame.copy()
        dev_copy = dev_frame.copy()
        build_copy["features"] = build_features
        dev_copy["features"] = dev_features

        model = WeightedRuleModel.fit(
            build_copy,
            feature_column="features",
            label_column=task_name,
            labels=labels,
            min_count=candidate.min_count,
            min_log_odds=candidate.min_log_odds,
            max_features=candidate.max_features,
        )
        predictions = [model.predict(features) for features in dev_copy["features"]]
        accuracy = accuracy_score(dev_copy[task_name].tolist(), predictions)
        macro_f1 = macro_f1_score(dev_copy[task_name].tolist(), predictions, labels)
        results.append(
            {
                "task": task_name,
                "candidate_name": candidate.name,
                "feature_mode": candidate.feature_mode,
                "min_count": candidate.min_count,
                "min_log_odds": candidate.min_log_odds,
                "max_features": candidate.max_features,
                "dev_accuracy": accuracy,
                "dev_macro_f1": macro_f1,
                "selection_score": accuracy + macro_f1,
            }
        )

    results.sort(key=lambda row: row["selection_score"], reverse=True)
    best = next(
        candidate
        for candidate in candidates
        if candidate.name == results[0]["candidate_name"]
    )
    return results, best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final optimized rule-based experiment."
    )
    parser.add_argument(
        "--dataset", default="aa_dataset-tickets-multi-lang-5-2-50-version.csv"
    )
    parser.add_argument("--output-dir", default="rule_based_final_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset_path)
    for column in [
        "subject",
        "body",
        "type",
        "priority",
        "queue",
        "language",
        *TAG_COLUMNS,
    ]:
        frame[column] = frame[column].fillna("")
    frame["text"] = (frame["subject"] + " " + frame["body"]).astype(str)
    frame["split"] = frame["text"].map(deterministic_split)
    frame["urgency"] = frame["priority"].map(
        lambda value: "urgent" if value == "high" else "standard"
    )

    build_frame = frame[frame["split"] == "build"].reset_index(drop=True)
    dev_frame = frame[frame["split"] == "dev"].reset_index(drop=True)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    build_dev_frame = pd.concat([build_frame, dev_frame], ignore_index=True)

    type_candidates = [
        CandidateConfig("type_text_balanced", "text", 8, 0.20, 300),
        CandidateConfig("type_text_strict", "text", 8, 0.35, 500),
    ]
    urgency_candidates = [
        CandidateConfig("urgency_text_balanced", "text", 8, 0.20, 300),
        CandidateConfig("urgency_text_strict", "text", 8, 0.35, 500),
        CandidateConfig("urgency_text_tags", "text+tags", 8, 0.25, 1200),
        CandidateConfig("urgency_tags_only", "tags", 5, 0.10, 500),
    ]

    type_results, best_type = run_candidate_search(
        build_frame, dev_frame, "type", TYPE_LABELS, type_candidates
    )
    urgency_results, best_urgency = run_candidate_search(
        build_frame, dev_frame, "urgency", URGENCY_LABELS, urgency_candidates
    )

    build_dev_frame["type_features"] = build_dev_frame.apply(
        lambda row: make_features(row, best_type.feature_mode),
        axis=1,
    )
    build_dev_frame["urgency_features"] = build_dev_frame.apply(
        lambda row: make_features(row, best_urgency.feature_mode),
        axis=1,
    )
    test_frame["type_features"] = test_frame.apply(
        lambda row: make_features(row, best_type.feature_mode),
        axis=1,
    )
    test_frame["urgency_features"] = test_frame.apply(
        lambda row: make_features(row, best_urgency.feature_mode),
        axis=1,
    )

    type_model = WeightedRuleModel.fit(
        build_dev_frame,
        "type_features",
        "type",
        TYPE_LABELS,
        best_type.min_count,
        best_type.min_log_odds,
        best_type.max_features,
    )
    urgency_model = WeightedRuleModel.fit(
        build_dev_frame,
        "urgency_features",
        "urgency",
        URGENCY_LABELS,
        best_urgency.min_count,
        best_urgency.min_log_odds,
        best_urgency.max_features,
    )

    test_frame["type_pred"] = [
        type_model.predict(features) for features in test_frame["type_features"]
    ]
    test_frame["urgency_pred"] = [
        urgency_model.predict(features) for features in test_frame["urgency_features"]
    ]

    metrics = {
        "dataset": {
            "rows": int(len(frame)),
            "build_rows": int(len(build_frame)),
            "dev_rows": int(len(dev_frame)),
            "test_rows": int(len(test_frame)),
            "languages": frame["language"].value_counts().to_dict(),
        },
        "candidate_search": {
            "type": type_results,
            "urgency": urgency_results,
        },
        "selected_configs": {
            "type": best_type.__dict__,
            "urgency": best_urgency.__dict__,
        },
        "test_metrics": {
            "type": evaluate(test_frame, "type", "type_pred", TYPE_LABELS),
            "urgency": evaluate(
                test_frame,
                "urgency",
                "urgency_pred",
                URGENCY_LABELS,
                ordinal_mapping=URGENCY_ORDINAL,
            ),
        },
    }

    (output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(type_results + urgency_results).to_csv(
        output_dir / "final_candidate_results.csv", index=False
    )
    test_frame[
        [
            "language",
            "subject",
            "body",
            "type",
            "type_pred",
            "priority",
            "urgency",
            "urgency_pred",
            "queue",
            *TAG_COLUMNS,
        ]
    ].to_csv(output_dir / "final_test_predictions.csv", index=False)

    rule_payload = {
        "type_top_rules": type_model.top_rules(limit=25),
        "urgency_top_rules": urgency_model.top_rules(limit=25),
    }
    (output_dir / "final_top_rules.json").write_text(
        json.dumps(rule_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Completed final optimized rule-based experiment.")
    print(f"Metrics: {output_dir / 'final_metrics.json'}")
    print(f"Candidate results: {output_dir / 'final_candidate_results.csv'}")
    print(f"Predictions: {output_dir / 'final_test_predictions.csv'}")


if __name__ == "__main__":
    main()
