SUPERVISED_PROJ := "packages/ticket_router_supervised"
AGENTIC_PROJ := "packages/ticket_router_agent"

prepare-data:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/01_build_test_set.py

# Supervised learning

run-ml:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/03_run_supervised_traditional.py

run-mbert *ARGS:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/04_run_mbert.py {{ARGS}}

# LLM-based
quan-qwen *ARGS:
    uv run --project {{AGENTIC_PROJ}} {{AGENTIC_PROJ}}/scripts/quantize_qwen.py {{ARGS}}

run-vllm *ARGS:
    uv run --project {{AGENTIC_PROJ}} {{AGENTIC_PROJ}}/scripts/run_batch.py {{ARGS}}

gen-batch:
    uv run --project {{AGENTIC_PROJ}} {{AGENTIC_PROJ}}/scripts/gen_batch.py
