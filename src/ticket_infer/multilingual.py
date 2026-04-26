"""Attribute inferrer for multilingual customer support tickets dataset."""

import json
from typing import Any, Dict

from ticket_router_base.types import Record

from .base import AttributeInferrer, register_inferrer
from .schema import AttributePrediction, TechProficiency, UserType


SYSTEM_PROMPT = """You are a customer support ticket analysis expert. Infer user attributes from ticket content.

## Tasks

1. **user_type**: Classify user type
   - individual: Uses singular first-person pronouns (I/my/me)
   - enterprise: Uses plural first-person pronouns (we/our/team/company) or mentions organizations
   - unknown: When content is too short or ambiguous

2. **industry**: Classify industry based on products and services
   - IT Services
   - Tech Online Store
   - IT Consulting Firm
   - Software Development Company
   - Online Store
   - Other (when cannot classify)

3. **tech_proficiency**: Assess user's technical proficiency
   - low: Non-technical user, only describes symptoms, no technical terms, no troubleshooting attempts
   - high: Technical professional, provides detailed error info, logs, configurations, troubleshooting steps
   - unknown: When content is too short or lacks information

4. **reason**: Provide brief reasoning in Chinese (2-3 sentences)

## Output Format

Return ONLY a JSON object, no markdown formatting.
JSON must contain:
- user_type: "individual" or "enterprise" or "unknown"
- industry: industry name string
- tech_proficiency: "low" or "high" or "unknown"
- reason: reasoning in Chinese

## Notes

- For user_type, check the BODY content, not the subject
- For tech_proficiency, assess the user's ability to communicate technically, not the difficulty of the issue
"""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "user_type": {
            "type": "string",
            "enum": ["individual", "enterprise", "unknown"],
        },
        "industry": {
            "type": "string",
        },
        "tech_proficiency": {
            "type": "string",
            "enum": ["low", "high", "unknown"],
        },
        "reason": {
            "type": "string",
        },
    },
    "required": [
        "user_type",
        "industry",
        "tech_proficiency",
        "reason",
    ],
}


def _clean_raw_output(raw: str) -> str:
    """Remove markdown formatting if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[-1] == "```":
            raw = "\n".join(lines[1:-1])
        else:
            raw = "\n".join(lines[1:])
    return raw.strip()


@register_inferrer("multilingual-customer-support")
class MultilingualAttributeInferrer(AttributeInferrer):
    """Attribute inferrer for multilingual customer support tickets."""

    @property
    def dataset_name(self) -> str:
        return "multilingual-customer-support"

    @property
    def output_schema(self) -> Dict[str, Any]:
        return OUTPUT_SCHEMA

    def build_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_prompt(self, record: Record) -> str:
        lang = record.language.value if record.language else "unknown"
        title = record.title or ""
        body = record.body

        body_short = body[:1500] + "..." if len(body) > 1500 else body

        return f"""Language: {lang}
Subject: {title}
Body: {body_short}

Infer the user attributes for the ticket above. Output JSON only."""

    def parse_output(self, raw: str, request_id: str) -> AttributePrediction:
        """Parse raw LLM output into AttributePrediction."""
        raw = _clean_raw_output(raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return AttributePrediction(
                request_id=request_id,
                user_type=UserType.UNKNOWN,
                industry="unknown",
                tech_proficiency=TechProficiency.UNKNOWN,
                reason=f"JSON parse failed: {str(e)[:50]}",
            )

        try:
            user_type = UserType(data["user_type"])
        except (KeyError, ValueError):
            user_type = UserType.UNKNOWN

        try:
            tech_proficiency = TechProficiency(data["tech_proficiency"])
        except (KeyError, ValueError):
            tech_proficiency = TechProficiency.UNKNOWN

        return AttributePrediction(
            request_id=request_id,
            user_type=user_type,
            industry=data.get("industry", "unknown"),
            tech_proficiency=tech_proficiency,
            reason=data.get("reason", ""),
        )
