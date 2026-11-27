"""
Everything the bot knows about the clinic: loads clinic.yaml and builds the
texts the bot sends.

Nothing in this file knows about Telegram. That keeps it easy to test, and the
same code could later power a website or another messenger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

CONFIG_PATH = Path(__file__).with_name("clinic.yaml")

# The keys used in clinic.yaml, in Python's weekday order (Monday = 0)
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DAY_NAMES = {
    "mon": "Monday",
    "tue": "Tuesday",
    "wed": "Wednesday",
    "thu": "Thursday",
    "fri": "Friday",
    "sat": "Saturday",
    "sun": "Sunday",
}


@dataclass(frozen=True)
class Service:
    name: str
    price: float
    duration_min: int


@dataclass(frozen=True)
class Clinic:
    name: str
    address: str
    phone: str
    timezone: ZoneInfo
    currency: str
    hours: dict[str, tuple[time, time] | None]  # None means closed that day
    services: tuple[Service, ...]

    def now(self) -> datetime:
        """The current time where the clinic is, not where the server is."""
        return datetime.now(self.timezone)

    def is_open(self, at: datetime | None = None) -> bool:
        at = (at or self.now()).astimezone(self.timezone)
        today = self.hours[DAYS[at.weekday()]]
        if today is None:
            return False
        opens, closes = today
        return opens <= at.time() < closes

    def format_price(self, price: float) -> str:
        amount = f"{price:,.0f}" if price == int(price) else f"{price:,.2f}"
        return f"{self.currency}{amount}"

    def prices_text(self) -> str:
        lines = [f"💰 Prices at {self.name}", ""]
        for s in self.services:
            lines.append(f"• {s.name}: {self.format_price(s.price)} ({s.duration_min} min)")
        return "\n".join(lines)

    def hours_text(self, at: datetime | None = None) -> str:
        at = (at or self.now()).astimezone(self.timezone)
        today = DAYS[at.weekday()]
        lines = ["🕒 Opening hours", ""]
        for day in DAYS:
            slot = self.hours[day]
            when = "closed" if slot is None else f"{slot[0]:%H:%M}–{slot[1]:%H:%M}"
            marker = "  ← today" if day == today else ""
            lines.append(f"{DAY_NAMES[day]}: {when}{marker}")
        lines.append("")
        lines.append("🟢 Open now" if self.is_open(at) else "🔴 Closed now")
        return "\n".join(lines)

    def contacts_text(self) -> str:
        return f"📍 {self.address}\n📞 {self.phone}"


def load_clinic(path: Path = CONFIG_PATH) -> Clinic:
    """Read clinic.yaml. Stops the bot at startup with a clear message if something's wrong."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    try:
        return Clinic(
            name=str(raw["name"]),
            address=str(raw["address"]),
            phone=str(raw["phone"]),
            timezone=ZoneInfo(str(raw["timezone"])),
            currency=str(raw.get("currency", "")),
            hours={day: _parse_hours(day, raw["hours"].get(day)) for day in DAYS},
            services=tuple(
                Service(str(s["name"]), float(s["price"]), int(s["duration_min"]))
                for s in raw["services"]
            ),
        )
    # ZoneInfoNotFoundError is a kind of KeyError, so it has to be checked first
    except ZoneInfoNotFoundError:
        raise ValueError(
            f"{path.name}: unknown timezone {raw['timezone']!r}. Use a name like 'Europe/Berlin'."
        ) from None
    except KeyError as missing:
        raise ValueError(f"{path.name}: the setting {missing} is missing") from None


def _parse_hours(day: str, value: object) -> tuple[time, time] | None:
    """'09:00-18:00' becomes (09:00, 18:00). 'closed' or a missing day becomes None."""
    if value is None or str(value).strip().lower() == "closed":
        return None
    try:
        opens, closes = (time.fromisoformat(part.strip()) for part in str(value).split("-"))
    except ValueError:
        raise ValueError(
            f"clinic.yaml: hours.{day} should look like \"09:00-18:00\" or closed, got {value!r}"
        ) from None
    if opens >= closes:
        raise ValueError(f"clinic.yaml: on {day} the clinic closes before it opens ({value!r})")
    return opens, closes
