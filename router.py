"""Classify the assignment's three intents without another model call."""

import re
from typing import Literal

from conversation import resolve_followup

Intent = Literal["learn", "pricing", "demo"]


def classify_intent(
    message: str, history: list[dict] | None = None, page_path: str = ""
) -> Intent:
    text = resolve_followup(message, history or [], page_path)
    text = re.sub(r"\b(?:meating|meting)\b", "meeting", text, flags=re.I)
    if re.search(
        r"\b(?:book|schedule|request|arrange)\b.{0,30}\b(?:demo|meeting|call)\b|"
        r"\b(?:talk|speak|connect|contact)\b.{0,15}\b(?:sales|team)\b|"
        r"\blive demonstration\b",
        text, re.I,
    ):
        return "demo"
    pricing_text = re.sub(r"\b(?:reduce|cut|save|lower)\b.{0,20}\bcosts?\b", "", text, flags=re.I)
    if re.search(r"\b(?:pric(?:e|es|ing)|costs?|fees?|plans?|quotes?|subscriptions?)\b", pricing_text, re.I):
        return "pricing"
    if re.search(r"\b(?:demo|trial)\b", text, re.I) or re.fullmatch(
        r"(?:book|schedule)[?!. ]*", text, re.I
    ):
        return "demo"
    return "learn"
