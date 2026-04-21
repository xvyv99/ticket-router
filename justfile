SUPERVISED_PROJ := "packages/ticket_router_supervised"
AGENTIC_PROJ := "packages/ticket_router_agent"

# Environment setup
sync-env:
    uv sync --no-dev --project {{SUPERVISED_PROJ}}
    uv sync --no-dev --project {{AGENTIC_PROJ}}

# Data preparation
default-dataset := "multilingual-customer-support"

default-test-num := "1200"

default-difficult-num := "100"

prepare-data DATASET=default-dataset TEST_NUM=default-test-num DIFFICULT_NUM=default-difficult-num:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/01_build_test_set.py \
        --dataset {{DATASET}} \
        --test-num {{TEST_NUM}} \
        --difficult-num {{DIFFICULT_NUM}}

# Supervised learning
run-ml DATASET=default-dataset OUTPUT_PREFIX="":
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/03_run_supervised_traditional.py \
        --dataset {{DATASET}} \
        --output-prefix {{OUTPUT_PREFIX}}

run-mbert *ARGS:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/04_run_mbert.py {{ARGS}}

# LLM-based
quan-qwen *ARGS:
    uv run --project {{AGENTIC_PROJ}} {{AGENTIC_PROJ}}/scripts/quantize_qwen.py {{ARGS}}

run-vllm *ARGS:
    uv run --project {{AGENTIC_PROJ}} {{AGENTIC_PROJ}}/scripts/run_batch.py {{ARGS}}

gen-batch DATASET=default-dataset:
    uv run --project {{AGENTIC_PROJ}} {{AGENTIC_PROJ}}/scripts/gen_batch.py
