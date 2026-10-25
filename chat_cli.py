"""Run terminal chat with: python chat_cli.py [--debug]."""
import asyncio
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from pydantic import ValidationError

from agent import MODEL, stream_chat
from memory import load_history, save_history
from leads import DemoRequest, meeting_booking_url, save_demo
from router import classify_intent


async def request_meeting() -> str:
    if booking_url := meeting_booking_url():
        return (f"Choose an available time and complete your booking here: {booking_url} "
                "Your appointment is confirmed only when the booking service confirms it.")
    print("Let's request a meeting. Type cancel at any step to stop. "
          "Details are saved locally; no calendar invitation or team notification is sent.")
    details = {}
    for field, prompt in (
        ("name", "Your name: "),
        ("company", "Company: "),
        ("email", "Contact email: "),
        ("need", "What would you like to discuss? "),
        ("preferred_time", "Preferred time (YYYY-MM-DDTHH:MM+05:30, or leave blank): "),
    ):
        value = input(prompt).strip()
        if value.casefold() in {"cancel", "bye", "/bye", "exit"}:
            return "Meeting request cancelled. Nothing was saved."
        if not value and field != "preferred_time":
            return "Meeting request not saved: name, company, contact email and topic are required. Please try again."
        details[field] = value or None
    try:
        lead = DemoRequest(**details, confirmed=True, request_id=uuid4())
    except ValidationError as error:
        fields = ", ".join(str(item["loc"][0]) for item in error.errors())
        return f"Meeting request not saved. Please check: {fields}. Use a valid email and include a time zone with the preferred time."
    if lead.preferred_time and lead.preferred_time <= datetime.now(timezone.utc):
        return "Meeting request not saved. Please choose a future time."
    print(f"Name: {lead.name}\nCompany: {lead.company}\nEmail: {lead.email}\n"
          f"Topic: {lead.need}\nPreferred time: {lead.preferred_time.isoformat() if lead.preferred_time else 'To be agreed'}")
    if input("Save this meeting request? (yes/no): ").strip().casefold() != "yes":
        return "Meeting request cancelled. Nothing was saved."
    try:
        saved = await asyncio.to_thread(save_demo, lead)
    except (sqlite3.Error, OSError, ValueError):
        return "Could not save the meeting request. Please try again."
    return (f"Meeting request saved. Reference: {saved['lead_id']}. "
            "Your preferred time is not confirmed. No invitation or team notification has been sent.")


async def main():
    session_id = uuid4()
    print("a31-assistance | type bye to exit.", flush=True)
    while True:
        try:
            message = input("You: ").strip()
            if message.lower() in {"bye", "/bye", "exit"}:
                break
            if not message:
                continue
            history = load_history(session_id)
            if classify_intent(message, history) == "demo":
                print("a31-assistance: " + await request_meeting(), flush=True)
                continue
            print("Thinking...", flush=True)
            chunks = []
            started = False
            async for delta in stream_chat(message, history):
                if not started:
                    print("a31-assistance: ", end="", flush=True)
                    started = True
                print(delta, end="", flush=True)
                chunks.append(delta)
            print(flush=True)
            answer = "".join(chunks)
            save_history(session_id, history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ])
        except (EOFError, KeyboardInterrupt):
            break
        except httpx.ConnectError:
            print(
                "Cannot connect to Ollama. Start the Ollama app or run: ollama serve",
                flush=True,
            )
        except httpx.TimeoutException:
            print("Ollama took too long. Close memory-heavy apps and try again.", flush=True)
        except ValueError:
            print(
                "\nSearch or answer generation failed. "
                "Check your search configuration and retry.",
                flush=True,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                print(f"Model not found. Run: ollama pull {MODEL}", flush=True)
            else:
                print(
                    f"Ollama returned HTTP {exc.response.status_code}. "
                    "Check the Ollama server.",
                    flush=True,
                )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--debug" in sys.argv:
        logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
        logging.getLogger("agent").setLevel(logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

