"""
The tools the agent can use.

A tool is an ordinary Python function. LangChain's @tool decorator turns its
name, docstring and type hints into the description (a JSON schema) that the
model sees. So the docstring is written for the model: it's how the model
decides when to use the tool and what to pass in.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain.tools import tool

from clinic import DAY_NAMES, DAYS, Clinic

Topic = Literal["services_and_prices", "opening_hours", "contacts"]


def make_tools(clinic: Clinic) -> list:
    """Build the tools for one clinic. Returns a list for create_agent."""

    @tool
    def get_clinic_info(topic: Topic) -> str:
        """Look up facts about the clinic. Use it whenever the patient asks about
        services, prices, opening hours, the address or the phone number.
        Never answer those from memory.

        topic: services_and_prices gives every service with its price and duration.
        opening_hours gives the hours of each weekday, the current date and time
        at the clinic, and whether it's open now. contacts gives the address and
        phone number.
        """
        return json.dumps(clinic_info(clinic, topic), ensure_ascii=False)

    return [get_clinic_info]


def clinic_info(clinic: Clinic, topic: str) -> dict:
    """The facts themselves. A plain function, so it's easy to test without LangChain."""
    if topic == "services_and_prices":
        return {
            "services": [
                {"name": s.name, "price": clinic.format_price(s.price), "duration_minutes": s.duration_min}
                for s in clinic.services
            ]
        }
    if topic == "opening_hours":
        hours = {}
        for day in DAYS:
            slot = clinic.hours[day]
            hours[DAY_NAMES[day]] = "closed" if slot is None else f"{slot[0]:%H:%M}-{slot[1]:%H:%M}"
        return {
            "now_at_clinic": f"{clinic.now():%A %d %B %Y, %H:%M}",
            "open_now": clinic.is_open(),
            "hours": hours,
        }
    if topic == "contacts":
        return {"name": clinic.name, "address": clinic.address, "phone": clinic.phone}
    return {"error": f"unknown topic {topic!r}. Use services_and_prices, opening_hours or contacts."}
