"""Resolve conversational references before retrieval starts."""

import re


PRODUCTS = (
    "NeXtory",
    "SpeedHire",
    "SpeedRec",
    "Sp3dhire",
    "Sp33dhire",
    "H3iVE",
    "Send3r",
    "Invenio",
    "A3i Video Studio",
    "Video AI",
    "3i Voice Assist",
    "AI-Note",
    "Notes AI",
    "AI-Hire",
    "AI-Lead",
    "AI-Advisor",
    "AI-Marketing",
)

WEBSITE_TOPICS = {
    "Company": r"\b(company|about us|team|leadership|founders?|co-founders?|ceo|coo)\b",
    "Contact": r"\b(contact|address|email address|phone number)\b",
    "Careers": r"\b(careers?|jobs?|vacancies|hiring opportunities)\b",
    "Industries": (
        r"\b(industries|industry|banks?|hedge funds?|financial advisors?|"
        r"restaurants?|clinics?|hospitality)\b"
    ),
    "Resources": r"\b(resources?|blogs?|articles?|guides?)\b",
    "Responsible AI": r"\bresponsible ai\b",
    "Privacy policy": r"\bprivacy\b",
    "Terms of service": r"\bterms\b",
    "Refund policy": r"\brefunds?\b",
    "Collaborations": r"\b(collaborations?|partnerships?|partners?)\b",
    "Competitions": r"\bcompetitions?\b",
    "Services": r"\bservices\b",
    "Workflows": r"\bworkflows?\b",
}

VAGUE_FOLLOW_UP = re.compile(
    r"(?:please )?(?:(?:tell me|explain)(?: more)?(?: about)?(?: this| that| it)?|"
    r"what (?:is|about) (?:it|that)|(?:what about |what is )?(?:its|their) "
    r"(?:price|pricing|features)|how much (?:does|will) (?:it|that|this) cost|"
    r"(?:what is the )?(?:price|pricing))",
    re.I,
)

CONTEXTUAL_FOLLOW_UP = re.compile(
    r"\b(it|its|their|this|that|pricing|price)\b|^what about|^and |tell me more",
    re.I,
)


def mentioned_products(text: str) -> list[str]:
    """Return A3i product names explicitly present in text."""
    return [
        product
        for product in PRODUCTS
        if re.search(r"\b" + re.escape(product) + r"\b", text, re.I)
    ]


def pricing_overview(text: str) -> bool:
    """Recognize catalog-wide pricing without expanding a single product's plans."""
    return bool(
        re.search(r"\b(?:pric(?:e|es|ing)|costs?|fees?|plans?|subscriptions?)\b", text, re.I)
        and not mentioned_products(text)
        and re.search(r"\b(?:products|all|every|each|entire)\b", text, re.I)
    )


def mentioned_topics(text: str) -> list[str]:
    """Return explicit product or website topics, in display order."""
    products = mentioned_products(text)
    if products:
        return products
    return [
        topic
        for topic, pattern in WEBSITE_TOPICS.items()
        if re.search(pattern, text, re.I)
    ]


def recent_topics(history: list[dict]) -> list[str]:
    """Find the most recent message that names an A3i topic."""
    for previous in reversed(history[-6:]):
        topics = mentioned_topics(previous["content"])
        if topics:
            return topics
    return []


def clarification(message: str, history: list[dict], page_path: str = "") -> str | None:
    """Ask for a topic when a reference cannot be resolved safely."""
    if pricing_overview(resolve_followup(message, history, page_path)):
        return None
    if mentioned_topics(message) or re.search(
        r"\bthis (page|website|site|company)\b", message, re.I
    ):
        return None

    choices = recent_topics(history)
    if re.search(r"\b(?:pric(?:e|es|ing)|costs?|fees?|quotes?|subscriptions?)\b", message, re.I):
        if not choices and page_path not in ("", "/"):
            return None
        if not choices or (len(choices) == 1 and choices[0] not in PRODUCTS):
            return "Which A3i product or service would you like pricing for?"
        if len(choices) > 1:
            return "Which product do you mean: " + ", ".join(choices[:3]) + "?"
    stripped_message = message.strip(" ?!.")
    is_vague = bool(VAGUE_FOLLOW_UP.fullmatch(stripped_message))
    is_vague = is_vague or bool(
        history and re.fullmatch(r"what(?:'s| is) this[?!. ]*", message, re.I)
    )
    is_vague = is_vague or bool(
        len(choices) > 1 and re.search(r"\b(it|its|that|this)\b", message, re.I)
    )
    if not is_vague:
        return None
    if len(choices) == 1 or (not choices and page_path not in ("", "/")):
        return None
    if not choices:
        return "Which A3i page or topic would you like to know about?"

    label = "product" if all(choice in PRODUCTS for choice in choices) else "A3i website topic"
    return f"Which {label} do you mean: " + ", ".join(choices[:3]) + "?"


def resolve_followup(message: str, history: list[dict], page_path: str = "") -> str:
    """Add the last clear topic without choosing between ambiguous options."""
    products = mentioned_products(message)
    if len(products) == 1 and re.fullmatch(r"(?:what about|and)\s+" + re.escape(products[0]) + r"[?!. ]*", message, re.I):
        previous = next((item["content"] for item in reversed(history[-6:]) if item["role"] == "user"), "")
        if re.search(r"\b(?:pric(?:e|es|ing)|costs?|plans?|subscriptions?)\b", previous, re.I):
            return f"{products[0]} pricing: {message}"
    if pricing_overview(message):
        return message
    if not products and re.fullmatch(r"(?:all(?: of them| products)?|every product|each product)[?!. ]*", message, re.I):
        previous_question = next(
            (item["content"] for item in reversed(history[-6:]) if item["role"] == "user"), ""
        )
        if re.search(r"\b(?:pric(?:e|es|ing)|costs?|fees?|plans?|subscriptions?)\b", previous_question, re.I):
            return "All A3i product pricing: " + message
    if (
        len(products) == 1
        and message.strip(" ?!.").casefold() == products[0].casefold()
        and history
        and history[-1]["role"] == "assistant"
        and re.search(r"which (?:a3i )?product", history[-1]["content"], re.I)
    ):
        previous_question = next(
            (item["content"] for item in reversed(history[-6:]) if item["role"] == "user"),
            "",
        )
        if re.search(r"\b(pric(?:e|es|ing)|cost|plans?|subscription)\b", previous_question, re.I):
            return f"{products[0]}: {previous_question}"
    if mentioned_topics(message):
        return message
    if re.search(r"\bthis (page|website|site)\b", message, re.I):
        return f"A3i page {page_path}: {message}" if page_path else message
    if not CONTEXTUAL_FOLLOW_UP.search(message):
        return message

    topics = recent_topics(history)
    if len(topics) == 1:
        return f"{topics[0]}: {message}"
    if not topics and page_path not in ("", "/"):
        return f"A3i page {page_path}: {message}"
    return message
