SUPERVISED_PROJ := "packages/ticket_router_supervised"
AGENTIC_PROJ := "packages/ticket_router_agent"

prepare-data:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/01_build_test_set.py

# Supervised learning

supervised-traditional:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/03_run_supervised_traditional.py

supervised-mbert *ARGS:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/04_run_mbert.py {{ARGS}}

# LLM-based

quantize-qwen *ARGS:
    uv run --project {{AGENTIC_PROJ}} {{AGENTIC_PROJ}}/scripts/quantize_qwen.py {{ARGS}}
