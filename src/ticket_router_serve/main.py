"""FastAPI application and routes for ticket-router serve."""

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status

from ticket_router_serve.cache import compute_fingerprint, find_by_fingerprint, get_cache_entry
from ticket_router_serve.deps import verify_api_key
from ticket_router_serve.models import get_pool, SUPPORTED_MODELS
from ticket_router_serve.schemas import (
    AttributionResponse,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ResultResponse,
    TaskStatus,
)
from ticket_router_serve.tasks import submit_attribution, submit_task


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

predict_router = APIRouter(prefix="/predict", dependencies=[Depends(verify_api_key)])
result_router = APIRouter(prefix="/result", dependencies=[Depends(verify_api_key)])
attribution_router = APIRouter(prefix="/attribution", dependencies=[Depends(verify_api_key)])


@predict_router.post("", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Validate model, check cache, and submit a new prediction task."""
    # 1. Validate model
    if request.model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(error=f"Unknown model: {request.model}").model_dump(),
        )

    # 2. Compute fingerprint
    fingerprint = compute_fingerprint(request.title, request.body, request.model)

    # 3. Check cache hit
    existing_req_id = find_by_fingerprint(fingerprint)
    if existing_req_id is not None:
        entry = get_cache_entry(existing_req_id)
        if entry is not None and entry.get("status") == TaskStatus.COMPLETED:
            return PredictResponse(
                req_id=existing_req_id,
                status=TaskStatus.COMPLETED,
                cached=True,
            )

    # 4. Submit new task
    req_id = submit_task(request.title, request.body, request.model)
    return PredictResponse(req_id=req_id, status=TaskStatus.PENDING, cached=False)


@result_router.get("/{req_id}", response_model=ResultResponse)
async def result(req_id: str) -> ResultResponse:
    """Load result from cache by req_id."""
    entry = get_cache_entry(req_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(error=f"Request {req_id} not found").model_dump(),
        )

    task_status = TaskStatus(entry.get("status", "PENDING"))
    result_data = entry.get("result") if task_status == TaskStatus.COMPLETED else None
    error_msg = entry.get("error") if task_status == TaskStatus.FAILED else None

    return ResultResponse(
        req_id=req_id,
        status=task_status,
        result=result_data,
        error=error_msg,
    )


@attribution_router.get("/{req_id}", response_model=AttributionResponse)
async def attribution(req_id: str) -> AttributionResponse:
    """Trigger or return attribution for rembert/xlm-roberta models."""
    entry = get_cache_entry(req_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(error=f"Request {req_id} not found").model_dump(),
        )

    task_status = TaskStatus(entry.get("status", "PENDING"))

    # No attribution until task is complete
    if task_status != TaskStatus.COMPLETED:
        return AttributionResponse(
            req_id=req_id,
            status=task_status,
            attribution=None,
            error=None,
        )

    model = entry.get("model", "")

    # Only rembert and xlm-roberta support attribution
    if model not in ("rembert", "xlm-roberta"):
        return AttributionResponse(
            req_id=req_id,
            status=task_status,
            attribution=None,
            error=None,
        )

    # Return cached attribution if already computed
    existing_attribution = entry.get("attribution")
    if existing_attribution is not None:
        return AttributionResponse(
            req_id=req_id,
            status=task_status,
            attribution=existing_attribution,
            error=None,
        )

    # Trigger async computation and return current status
    submit_attribution(req_id)
    return AttributionResponse(
        req_id=req_id,
        status=task_status,
        attribution=None,
        error=None,
    )


# Health check does not require auth
health_router = APIRouter()


@health_router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


# Register routers
app.include_router(predict_router)
app.include_router(result_router)
app.include_router(attribution_router)
app.include_router(health_router)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    """Initialize the model pool on application startup."""
    get_pool().initialize()
