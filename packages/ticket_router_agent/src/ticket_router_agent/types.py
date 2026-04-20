from typing import Annotated, List

from pydantic import BaseModel, Field

from ticket_router_base.types import Queue, Priority


class TicketOutput(BaseModel):
    queue: Queue
    priority: Priority
    tags: Annotated[List[str], Field(min_length=1, max_length=2)]
    preliminary_answer: str


TICKET_SCHEMA = TicketOutput.model_json_schema()
