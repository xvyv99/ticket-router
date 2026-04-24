default-dataset := "multilingual-customer-support"
default-test-num := "1200"
default-difficult-num := "100"

# Data preparation

prepare-data DATASET=default-dataset TEST_NUM=default-test-num DIFFICULT_NUM=default-difficult-num:
    uv run ./scripts/01_build_test_set.py \
        --dataset {{DATASET}} \
        --test-num {{TEST_NUM}} \
        --difficult-num {{DIFFICULT_NUM}}

# Supervised learning
run-ml DATASET=default-dataset OUTPUT_PREFIX="":
    uv run ./scripts/03_run_supervised_traditional.py \
        --dataset {{DATASET}} \
        {{ if OUTPUT_PREFIX != "" { "--output-prefix " + OUTPUT_PREFIX } else { "" } }}

run-mbert *ARGS:
    uv run ./scripts/04_run_mbert.py {{ARGS}}

# Evaluation
eval DATASET=default-dataset PRED_FILES="lr:xgb" PRED_DIR="outputs/supervised":
    uv run ./scripts/eval.py \
        --dataset {{DATASET}} \
        --pred-files {{PRED_FILES}} \
        --pred-dir {{PRED_DIR}}

# LLM-based
quan-qwen *ARGS:
    uv run ./scripts/quantize_qwen.py {{ARGS}}

run-vllm *ARGS:
    uv run ./scripts/run_batch.py {{ARGS}}

gen-batch:
    uv run ./scripts/gen_batch.py
