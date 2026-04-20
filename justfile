SUPERVISED_PROJ := "packages/ticket_router_supervised"

prepare-data:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/01_build_test_set.py

supervised-traditional:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/03_run_supervised_traditional.py

supervised-mbert *ARGS:
    uv run --project {{SUPERVISED_PROJ}} {{SUPERVISED_PROJ}}/scripts/04_run_mbert.py {{ARGS}}


