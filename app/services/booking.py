from datetime import date, time
from typing import Any

from pydantic import EmailStr, TypeAdapter, ValidationError
from sqlalchemy.orm import Session

from app.models import Booking


def create_booking(
    db: Session,
    details: dict[str, Any],
) -> tuple[Booking | None, str | None]:
    field_parsers = {
        "name": lambda value: value.strip() if isinstance(value, str) else None,
        "email": TypeAdapter(EmailStr).validate_python,
        "date": date.fromisoformat,
        "time": time.fromisoformat,
    }
    parsed: dict[str, Any] = {}
    invalid_fields: list[str] = []
    for field, parser in field_parsers.items():
        value = details.get(field)
        if value is None or value == "":
            parsed[field] = None
            continue
        try:
            parsed[field] = parser(value)
        except (TypeError, ValueError, ValidationError):
            parsed[field] = None
            invalid_fields.append(field)

    if invalid_fields:
        labels = ", ".join(invalid_fields)
        return None, f"Please provide a valid {labels} to complete the interview booking."

    missing = [field for field, value in parsed.items() if not value]
    if missing:
        labels = ", ".join(missing)
        return None, f"To book the interview, please provide your {labels}."

    booking = Booking(
        name=parsed["name"],
        email=str(parsed["email"]),
        interview_date=parsed["date"],
        interview_time=parsed["time"],
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)
    return booking, None
