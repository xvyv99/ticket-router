default-dataset := "multilingual-customer-support"

# Data preparation

prepare-data DATASET=default-dataset:
    uv run ./scripts/01_build_test_set.py \
        --dataset {{DATASET}} 

# Supervised learning
run-ml DATASET=default-dataset OUTPUT_PREFIX="":
    uv run ./scripts/03_run_supervised_traditional.py \
        --dataset {{DATASET}} \
        {{ if OUTPUT_PREFIX != "" { "--output-prefix " + OUTPUT_PREFIX } else { "" } }}

run-mbert *ARGS:
    uv run ./scripts/04_run_mbert.py {{ARGS}}

# Evaluation
eval DATASET=default-dataset:
    uv run ./scripts/eval.py \
        --dataset {{DATASET}}

# LLM-based
quan-qwen *ARGS:
    uv run ./scripts/quantize_qwen.py {{ARGS}}

run-vllm *ARGS:
    uv run ./scripts/run_batch.py {{ARGS}}

gen-batch:
    uv run ./scripts/gen_batch.py
