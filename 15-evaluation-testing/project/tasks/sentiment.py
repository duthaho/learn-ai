"""
Sentiment classifier — the bundled system-under-test for Module 15.

Public callable:
    classify_sentiment(review: str, model: str = MODEL) -> dict
        Returns {"sentiment": "...", "confidence": float}

Also exposes `SentimentLabel` so the harness can use it as a SchemaEvaluator.
"""

from __future__ import annotations

import json
import os
from typing import Literal

from litellm import completion
from pydantic import BaseModel, Field

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")

SYSTEM_PROMPT = """You are a sentiment classifier. Given a short movie review, return JSON with:
- sentiment: "positive", "negative", or "neutral"
- confidence: a float between 0.0 and 1.0

Return ONLY JSON. Do not include explanations or markdown fences."""


class SentimentLabel(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0)


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = s[3:]
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    s = s[i:]
    s = s.lstrip("\r\n ")
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def classify_sentiment(review: str, model: str = MODEL) -> dict:
    """Classify a movie review's sentiment. Returns a dict matching SentimentLabel."""
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": review},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    cleaned = _strip_code_fence(raw)
    return json.loads(cleaned)
