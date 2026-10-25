"""Search approved A3i content without exposing configurable domains to the model."""

import os
import logging
import re
import time
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from urllib.parse import urlsplit

from langchain_core.tools import ToolException, tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, ConfigDict, Field

from pathlib import Path
from dotenv import load_dotenv
from website import search_terms, website_results
from conversation import mentioned_products
from external_sources import approved_results, discover, external_allowed
load_dotenv(Path(__file__).with_name(".env"))
logger = logging.getLogger(__name__)
CACHE_TTL = 15 * 60
MAX_CACHE_ENTRIES = 128
OFFICIAL_HOSTS = {"a3i-accel.com", "www.a3i-accel.com", "send3r.a3i-accel.com"}
search_cache: OrderedDict = OrderedDict()
cache_lock = Lock()


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=1000)


def relevant_excerpt(text: str, query: str) -> str:
    """Keep relevant page sections within a small context budget."""
    terms = search_terms(query)
    sections = re.split(r"(?m)(?=^(?:[*+-] )?#{1,4}\s)", text)
    def score(section):
        heading = section.split("\n", 1)[0]
        heading_terms = search_terms(heading)
        focused_heading = len(heading_terms) == 1 and bool(terms & heading_terms)
        product_heading = "product" in terms and bool(mentioned_products(heading))
        pricing_section = "pricing" in terms and bool(re.search(
            r"call for pricing|\$\s*\d[^\n]*?/\s*(?:month|year)|licensed on this page",
            section, re.I,
        ))
        return (
            3 * len(terms & heading_terms)
            + len(terms & search_terms(section))
            + 8 * focused_heading
            + 5 * product_heading
            + 6 * pricing_section
        )
    count = 6 if "product" in terms else 3
    selected = sorted(enumerate(sections), key=lambda item: score(item[1]), reverse=True)[:count]
    # Give the best section space first so introductory text cannot cut off a pricing table.
    excerpts = []
    remaining = 3600
    for index, section in selected:
        plan_table = re.search(r"licensed on this page|\$\s*\d[^\n]*?/\s*(?:month|year)", section, re.I)
        limit = 3600 if plan_table else 1200
        excerpt = section[:min(limit, remaining)]
        excerpts.append((index, excerpt))
        remaining -= len(excerpt) + 2
        if remaining <= 0:
            break
    return "\n\n".join(section for _, section in sorted(excerpts))[:3600]


def _search_query(query: str) -> str:
    """Add company context to a visitor's search terms."""
    if not re.search(r"\ba3i\b", query, re.I):
        query = "A3i Accel " + query
    return query


def _cached_result(key: str) -> dict | None:
    with cache_lock:
        cached = search_cache.get(key)
        if cached and time.monotonic() - cached[0] < CACHE_TTL:
            search_cache.move_to_end(key)
            return deepcopy(cached[1])
        search_cache.pop(key, None)
    return None


def _cache_result(key: str, result: dict) -> None:
    """Store successful searches in a bounded least-recently-used cache."""
    with cache_lock:
        search_cache[key] = (time.monotonic(), deepcopy(result))
        search_cache.move_to_end(key)
        while len(search_cache) > MAX_CACHE_ENTRIES:
            search_cache.popitem(last=False)


def _official_results(response: dict, query: str) -> list[dict]:
    """Keep content from valid HTTPS URLs on the official A3i domain."""
    results = []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    for item in response.get("results", [])[:3]:
        url = str(item.get("url", ""))
        parsed = urlsplit(url)
        valid_url = (
            parsed.scheme == "https"
            and parsed.hostname in OFFICIAL_HOSTS
            and not parsed.username
            and not parsed.password
            and parsed.port in (None, 443)
        )
        if not valid_url:
            continue
        raw_content = str(item.get("raw_content") or item.get("content", ""))
        content = relevant_excerpt(raw_content, query)
        if content.strip():
            results.append(
                {
                    "url": url,
                    "content": content,
                    "source": "tavily",
                    "priority": "primary",
                    "retrieved_at": retrieved_at,
                }
            )
    return results


def _search_tavily(query: str) -> dict:
    search = TavilySearch(
        max_results=3,
        topic="general",
        search_depth="basic",
        include_domains=sorted(OFFICIAL_HOSTS),
        include_raw_content=True,
    )
    response = search.invoke({"query": query, "include_domains": sorted(OFFICIAL_HOSTS)})
    if isinstance(response, str) and "no search results" in response.lower():
        return {"results": []}
    if not isinstance(response, dict):
        return {"error": "Tavily returned an unexpected response. Please retry."}
    if response.get("error"):
        return search_failure(response["error"])
    return {"results": _official_results(response, query)}


@tool(args_schema=SearchQuery)
def search_a3i_web(query: str) -> dict:
    """Read official A3i evidence and reviewed external company-background snapshots."""
    query = _search_query(query)
    official = _search_official(query)
    if not external_allowed(query):
        return official
    try:
        reviewed = approved_results(query)
        key = "external:" + " ".join(query.casefold().split())
        if not reviewed and os.getenv("TAVILY_API_KEY") and _cached_result(key) is None:
            ids = discover(query)
            if ids:
                _cache_result(key, {"candidate_ids": ids})
        if reviewed:
            primary = official.get("results", [])
            primary = [{**row, "content": row["content"][:5400 // max(1, len(primary))]}
                       for row in primary]
            remaining = 7200 - sum(len(row["content"]) for row in primary)
            return {"results": primary + [
                {**row, "content": relevant_excerpt(row["content"], query)[:remaining // len(reviewed)]}
                for row in reviewed
            ]}
    except Exception as error:
        # Optional evidence must not turn a successful official lookup into an outage.
        logger.warning("External source fallback unavailable (%s)", type(error).__name__)
    return official


def _matches_question(pages: list[dict], query: str) -> bool:
    """Check topic coverage, not whether the evidence proves an answer."""
    terms = search_terms(query) - {"company", "background", "information"}
    content_terms = search_terms("\n".join(page["content"] for page in pages))
    required = terms & {"ceo", "coo", "founder", "pricing"}
    # shortcut: lexical coverage misses paraphrases, upgrade if real questions show gaps.
    return required <= content_terms and (not terms or bool(terms & content_terms))


def _search_official(query: str) -> dict:
    """Cache official evidence separately from revocable external approvals."""
    query = _search_query(query)
    key = " ".join(re.sub(r"[^\w\s]", " ", query.casefold()).split())
    cached = _cached_result(key)
    if cached is not None:
        return cached
    result = {"results": []}
    try:
        pages = website_results(query)
        if pages:
            result = {"results": [
                {**page, "content": relevant_excerpt(page["content"], query),
                 "source": "official_website", "priority": "primary",
                 "retrieved_at": datetime.now(timezone.utc).isoformat()}
                for page in pages
            ]}
            if _matches_question(result["results"], query):
                _cache_result(key, result)
                return result
        if not os.getenv("TAVILY_API_KEY"):
            if result["results"]:
                _cache_result(key, result)
                return result
            return {"error": "Tavily search is not configured."}
        logger.info("Calling Tavily search on the official A3i domain")
        fallback = _search_tavily(query)
        if fallback.get("results"):
            # Prefer search excerpts when the same URL's direct excerpt missed the topic.
            found_urls = {page["url"] for page in fallback["results"]}
            result = {"results": fallback["results"] + [
                page for page in result["results"] if page["url"] not in found_urls
            ]}
        elif not result["results"]:
            return fallback
        if result.get("results"):
            _cache_result(key, result)
        return result
    except ToolException as error:
        if result["results"]:
            return result
        if "no search results" in str(error).lower():
            return {"results": []}
        return search_failure(error)
    except Exception as error:
        return result if result["results"] else search_failure(error)


def search_failure(error) -> dict:
    """Classify failures without returning provider text or credentials."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    description = str(error).lower()
    if status in (401, 403) or "unauthorized" in description or "invalid api key" in description:
        message = "Tavily rejected the API key. Check TAVILY_API_KEY in .env."
    elif status == 432 or "credit" in description or "quota" in description:
        message = "Tavily search credits are exhausted. Check your Tavily account."
    elif status == 429:
        message = "Tavily rate limit reached. Please retry shortly."
    elif "timeout" in type(error).__name__.lower() or "timed out" in description:
        message = "Tavily search timed out. Please retry."
    else:
        message = "Tavily search is temporarily unavailable. Please retry."
    return {"error": message}
