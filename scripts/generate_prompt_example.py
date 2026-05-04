"""Generate human-readable prompt examples for the goal-based system.

Loads sample records from the test set, builds full conversations via
build_conversation(), and writes them to a plain-text file for inspection.

Supports few-shot examples sampled from the training set.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import List

from ticket_router.base.data import get_dataset
from ticket_router.base.data.datasets import DATASET_REGISTRY
from ticket_router.base.types import Record
from ticket_router.agent.prompt import build_conversation

OUTPUT_DIR = Path("outputs/goal_based")
DEFAULT_OUTPUT = OUTPUT_DIR / "prompt_example.txt"

# Number of examples to generate per language
EXAMPLES_PER_LANG = 1


def _format_conversation(record: Record, conversation: List[dict]) -> str:
    """Format a conversation list into a human-readable string."""
    lines: List[str] = []
    lines.append(
        f"=== Example | request_id={record.request_id} | language={record.language or 'N/A'} ==="
    )
    lines.append("")

    for msg in conversation:
        role = msg["role"].upper()
        content = msg["content"]
        lines.append(f"--- {role} ---")
        lines.append(content)
        lines.append("")

    lines.append("-" * 80)
    lines.append("")
    return "\n".join(lines)


def generate_prompt_examples(
    dataset_name: str,
    sample_num: int,
    few_shot: bool,
    output_path: Path,
) -> None:
    dataset_type = get_dataset(dataset_name)
    dataset = dataset_type()

    train_df, test_df, _ = dataset.load_train_test_split()
    test_records = dataset.df_to_records(test_df)[:sample_num]

    few_shot_examples: List[Record] | None = None
    if few_shot:
        few_shot_examples = dataset.sample_few_shot_examples(
            max_per_stratum=3,
            max_total=12,
        )

    # Pick one example per language to show cross-language consistency
    seen_langs: set[str] = set()
    selected: List[Record] = []
    for rec in test_records:
        lang = rec.language.value if rec.language is not None else "unknown"
        if lang not in seen_langs and len(selected) < 5:
            seen_langs.add(lang)
            selected.append(rec)

    # If we haven't filled the quota, pad with more examples
    idx = 0
    while len(selected) < 5 and idx < len(test_records):
        if test_records[idx] not in selected:
            selected.append(test_records[idx])
        idx += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Goal-Based System Prompt Examples\n")
        f.write(f"# Dataset: {dataset.name}\n")
        f.write(f"# Few-shot: {'enabled' if few_shot_examples else 'disabled'}\n")
        f.write(f"# Total examples: {len(selected)}\n")
        f.write("\n")

        for rec in selected:
            conv = build_conversation(rec, dataset, few_shot_examples)
            f.write(_format_conversation(rec, conv))

    print(f"Written {len(selected)} prompt examples to {output_path}")


def main() -> None:
    parser = ArgumentParser(
        description="Generate prompt examples for goal-based system"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to use",
    )
    parser.add_argument(
        "--sample-num",
        type=int,
        default=100,
        help="Number of test samples to consider for selection",
    )
    parser.add_argument(
        "--no-few-shot",
        dest="few_shot",
        action="store_false",
        default=True,
        help="Disable few-shot prompting (default: enabled)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    generate_prompt_examples(
        args.dataset,
        args.sample_num,
        args.few_shot,
        args.output,
    )


if __name__ == "__main__":
    from logging import basicConfig

    basicConfig(level="INFO")
    main()
