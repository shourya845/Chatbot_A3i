"""A separate confirmed demo form; it does not add steps to normal chat."""
import sqlite3
import os
from datetime import timezone
from contextlib import closing
from typing import Literal
from uuid import UUID, uuid4
from urllib.parse import urlsplit

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "leads.sqlite3"


def meeting_booking_url() -> str:
    url = os.getenv("MEETING_BOOKING_URL", "").strip()
    try:
        parsed = urlsplit(url)
        if (parsed.scheme == "https" and parsed.hostname and not parsed.username
                and not parsed.password and parsed.port in (None, 443)
                and not any(char.isspace() for char in url)):
            return url
    except ValueError:
        pass
    return ""


class DemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    company: str = Field(min_length=1, max_length=200)
    need: str = Field(min_length=1, max_length=2000)
    email: str = Field(default="", max_length=254, pattern=r"^(?:[^\s@]+@[^\s@]+\.[^\s@]+)?$")
    preferred_time: AwareDatetime | None = None
    confirmed: Literal[True]
    request_id: UUID


def save_demo(lead: DemoRequest) -> dict:
    preferred_time = lead.preferred_time.astimezone(timezone.utc).isoformat() if lead.preferred_time else None
    details = (lead.name, lead.company, lead.need, lead.email, preferred_time)
    DB_PATH.parent.mkdir(exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as db:
        with db:
            # Compatible with the lead table from the earlier prototype.
            db.execute(
                """CREATE TABLE IF NOT EXISTS leads (
                    id TEXT PRIMARY KEY,
                    request_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    company TEXT NOT NULL,
                    need TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(leads)")}
            if "email" not in columns:
                db.execute("ALTER TABLE leads ADD COLUMN email TEXT NOT NULL DEFAULT ''")
            if "preferred_time" not in columns:
                db.execute("ALTER TABLE leads ADD COLUMN preferred_time TEXT")
            existing = db.execute(
                "SELECT id, name, company, need, email, preferred_time FROM leads WHERE request_id = ?",
                (str(lead.request_id),),
            ).fetchone()
            if existing:
                if existing[1:] != details:
                    raise ValueError("Request ID was already used for different details.")
                return {"lead_id": existing[0], "status": "saved"}
            lead_id = str(uuid4())
            db.execute(
                "INSERT INTO leads (id, request_id, name, company, need, email, preferred_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (lead_id, str(lead.request_id), *details),
            )
    return {"lead_id": lead_id, "status": "saved"}
