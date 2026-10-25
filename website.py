"""Read the public HTML fragments used by the official A3i website."""

import json
import logging
import re
from html.parser import HTMLParser
from pathlib import Path

import httpx
from conversation import mentioned_products, pricing_overview

logger = logging.getLogger(__name__)
SOURCES_PATH = Path(__file__).with_name("website_sources.json")
PRODUCT_URLS = {
    "send3r": "https://send3r.a3i-accel.com/",
    "send3r-demo": "https://send3r.a3i-accel.com/demo",
}
STOPWORDS = set(
    "a3i accel a3i-accel ai what wha is are the a an of for to in and or this that it its "
    "website does do how can i me my you your we our with about will all from "
    "which who provide information explain tell more have has".split()
)
ALIASES = {
    "cofounder": "founder", "co-founder": "founder",
    "founders": "founder", "co-founders": "founder",
    "established": "founded", "establish": "founded", "located": "location",
    "headquarters": "location", "headquartered": "location", "based": "location",
    "prices": "pricing", "price": "pricing", "cost": "pricing", "costs": "pricing",
    "quote": "pricing", "quotes": "pricing", "fees": "pricing",
    "products": "product", "agents": "agent", "apps": "app", "ants": "ant",
    "integrations": "integration", "services": "service", "businesses": "business",
    "industries": "industry", "industry-specific": "industry", "solutions": "solution",
    "customized": "custom", "customization": "custom",
}


def search_terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.casefold())
    return {ALIASES.get(word, word) for word in words if word not in STOPWORDS}


class PageText(HTMLParser):
    """Keep text blocks and headings; exclude scripts, styles and forms."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "form"}:
            self.hidden += 1
        if not self.hidden and tag == "a":
            attributes = dict(attrs)
            label = attributes.get("aria-label")
            if label:
                target = attributes.get("data-nav", "")
                if re.fullmatch(r"[a-z0-9-]+", target):
                    label += " — linked product page: https://a3i-accel.com/partials/" + target + ".html"
                self.parts.append("\n" + label + "\n")
        if not self.hidden and tag in {"p", "li", "div", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")
            if tag.startswith("h"):
                self.parts.append("#" * int(tag[1]) + " ")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "form"}:
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden and tag in {"p", "li", "div", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self):
        text = "\n".join(
            line for part in "".join(self.parts).splitlines()
            if (line := " ".join(part.split()))
        )
        return re.sub(r"(?m)^(#{1,4})[ \t]*\n[ \t]*", r"\1 ", text)


def website_results(query: str) -> list[dict]:
    """Fetch two relevant pages, or all catalog product pages for a pricing overview."""
    try:
        sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Website source catalog is unavailable")
        return []
    terms = search_terms(query)
    products = set(mentioned_products(query))
    ranked = sorted(
        sources,
        key=lambda page: (
            bool(page.get("product")) and bool(products & set(mentioned_products(page["keywords"]))),
            len(terms & search_terms(page["keywords"])),
        ),
        reverse=True,
    )
    selected = [page for page in ranked if terms & search_terms(page["keywords"])][:2]
    if pricing_overview(query):
        selected = [page for page in sources if page.get("product") or page["id"] == "home"]
    if not selected:
        selected = [sources[0]]
    results = []
    with httpx.Client(timeout=8, follow_redirects=False) as client:
        for page in selected:
            # Only fetch fixed fragment IDs, never a URL supplied by a visitor or model.
            page_id = page["id"]
            if not re.fullmatch(r"[a-z0-9-]+", page_id):
                continue
            url = PRODUCT_URLS.get(page_id) or "https://a3i-accel.com/" + (
                "" if page_id == "home" else f"partials/{page_id}.html"
            )
            try:
                response = client.get(url)
                response.raise_for_status()
                if len(response.content) > 2_000_000:
                    continue
                parser = PageText()
                parser.feed(response.text)
                content = parser.text()
                if content:
                    results.append({"url": url, "title": page["title"], "content": content})
                    logger.info("Read public A3i page: %s", page_id)
            except (httpx.HTTPError, ValueError):
                logger.warning("Could not read public A3i page: %s", page_id)
    return results
