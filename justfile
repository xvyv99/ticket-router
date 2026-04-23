# BASE_PROJ := "packages/ticket_router_base"
SUPERVISED_PROJ := "packages/ticket_router_supervised"
AGENTIC_PROJ := "packages/ticket_router_agent"

default-dataset := "multilingual-customer-support"
default-test-num := "1200"
default-difficult-num := "100"

# Environment setup
sync-env:
    uv sync --no-dev --project {{SUPERVISED_PROJ}}
    uv sync --no-dev --project {{AGENTIC_PROJ}}

# Data preparation

prepare-data DATASET=default-dataset TEST_NUM=default-test-num DIFFICULT_NUM=default-difficult-num:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/01_build_test_set.py \
        --dataset {{DATASET}} \
        --test-num {{TEST_NUM}} \
        --difficult-num {{DIFFICULT_NUM}}

# Supervised learning
run-ml DATASET=default-dataset OUTPUT_PREFIX="":
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/03_run_supervised_traditional.py \
        --dataset {{DATASET}} \
        {{ if OUTPUT_PREFIX != "" { "--output-prefix " + OUTPUT_PREFIX } else { "" } }}

run-mbert *ARGS:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/04_run_mbert.py {{ARGS}}

# Evaluation
eval DATASET=default-dataset PRED_FILES="lr:xgb" PRED_DIR="outputs/supervised":
    uv run scripts/eval.py \
        --dataset {{DATASET}} \
        --pred-files {{PRED_FILES}} \
        --pred-dir {{PRED_DIR}}

# LLM-based
quan-qwen *ARGS:
    uv run --project {{AGENTIC_PROJ}} {{AGENTIC_PROJ}}/scripts/quantize_qwen.py {{ARGS}}

run-vllm *ARGS:
    uv run --project {{AGENTIC_PROJ}} {{AGENTIC_PROJ}}/scripts/run_batch.py {{ARGS}}

gen-batch:
    uv run --project {{AGENTIC_PROJ}} {{AGENTIC_PROJ}}/scripts/gen_batch.py
