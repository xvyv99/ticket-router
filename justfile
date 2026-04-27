default-dataset := "multilingual-customer-support"

# Data preparation

prepare-data DATASET=default-dataset:
    uv run ./scripts/01_build_test_set.py \
        --dataset {{DATASET}} 

infer-attr *ARGS:
    uv run ./scripts/infer_attributes.py \
        {{ARGS}}

# Supervised learning
run-ml *ARGS:
    uv run ./scripts/03_run_supervised_traditional.py {{ARGS}}

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
