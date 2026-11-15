from datetime import date, time

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database import Base
from app.models import Booking
from app.services.booking import create_booking


def test_create_booking_persists_valid_details() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        booking, prompt = create_booking(
            db,
            {
                "name": "Alex Doe",
                "email": "alex@example.com",
                "date": "2026-12-01",
                "time": "14:30",
            },
        )

        assert prompt is None
        assert booking is not None
        assert booking.interview_date == date(2026, 12, 1)
        assert booking.interview_time == time(14, 30)
        assert db.scalar(select(Booking).where(Booking.id == booking.id)) is not None

    engine.dispose()


def test_create_booking_requests_missing_details() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        booking, prompt = create_booking(db, {"name": "Alex Doe"})

        assert booking is None
        assert prompt == "To book the interview, please provide your email, date, time."
        assert db.scalar(select(Booking)) is None

    engine.dispose()


def test_create_booking_rejects_invalid_email() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        booking, prompt = create_booking(
            db,
            {
                "name": "Alex Doe",
                "email": "not-an-email",
                "date": "2026-12-01",
                "time": "14:30",
            },
        )

        assert booking is None
        assert prompt == "Please provide a valid email to complete the interview booking."
        assert db.scalar(select(Booking)) is None

    engine.dispose()
