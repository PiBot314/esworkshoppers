"""IIIT-Hyderabad academic calendar for 2025, transcribed from the two official almanacs
(Spring 2025 - Final, dated 18 Nov 2024; Monsoon 2025 - Final, dated 24 Apr 2025), supplied by the user.

Used by build_dataset.py to fill the campus-load calendar features for real-2025 training
(plan_19112026.md B1). This replaces the earlier plan to *infer* term/vacation from load drops:
the holidays below independently match the biggest 2025 grid-load drops (Holi, Gandhi Jayanti/Oct 2,
Independence Day, Diwali, Vinayaka Chavithi, Milad, Christmas), which cross-validates both.

Confidence:
  * holiday dates and the semester/vacation boundaries are taken directly from the almanacs (high).
  * exam windows and events are approximate day-ranges read off the almanac grid (marked `approx`),
    good enough for a coarse "exams in progress / fest on" flag but not exact per-course schedules.

Flags produced per day (daily_flags):
  in_session : 1 during a teaching semester (classes + exam weeks), 0 during summer/winter vacation
  holiday    : 1 on an institute holiday (campus largely closed; big load drop)
  exam       : 1 during a mid-/end-sem examination window (labs busy, fewer regular classes)
  event      : 1 on a notable campus event (fest/convocation/foundation day/showcase)
  day_type   : one of {vacation, holiday, exam, event, class} (priority holiday>exam>event>class, else vacation)
"""
import pandas as pd

# --- institute holidays 2025 (no classwork) -------------------------------------------------
HOLIDAYS = {
    "2025-01-13": "Bhogi",
    "2025-01-14": "Pongal/Sankranti",
    "2025-01-26": "Republic Day",
    "2025-02-26": "Mahashivaratri",
    "2025-03-14": "Holi",
    "2025-03-30": "Ugadi",
    "2025-03-31": "Ramzan (Eid al-Fitr)",
    "2025-04-18": "Good Friday",
    "2025-07-06": "Muharram",
    "2025-08-15": "Independence Day",
    "2025-08-21": "Bonalu",
    "2025-08-27": "Vinayaka Chavithi",
    "2025-09-05": "Milad-un-Nabi",
    "2025-10-02": "Vijaya Dasami / Gandhi Jayanti",
    "2025-10-20": "Naraka Chaturdasi",
    "2025-10-21": "Deepavali",
    "2025-11-05": "Gurunanak Jayanti",
    "2025-12-25": "Christmas",
}

# --- teaching semesters (inclusive) ---------------------------------------------------------
# Spring 2025: first day of classes 2 Jan; classes end 24 Apr; end-sem exams to ~10 May.
# Monsoon 2025: first day of classwork (non-UG1) 31 Jul; classes end 20 Nov; end-sem to 1 Dec; grades due 6 Dec.
SEMESTERS = [("2025-01-02", "2025-05-10"), ("2025-07-31", "2025-12-06")]

# --- examination windows (approx, inclusive) ------------------------------------------------
EXAMS = [
    ("2025-03-01", "2025-03-08"),  # Spring mid exams
    ("2025-04-28", "2025-05-10"),  # Spring end-sem / lab exams / project presentations
    ("2025-09-22", "2025-09-26"),  # Monsoon mid exams
    ("2025-11-24", "2025-12-01"),  # Monsoon end-sem / lab exams
]

# --- notable campus events (approx, inclusive) ----------------------------------------------
EVENTS = {
    ("2025-02-15", "2025-02-16"): "Felicity (fest)",
    ("2025-03-08", "2025-03-08"): "R&D Showcase",
    ("2025-08-12", "2025-08-12"): "24th Convocation",
    ("2025-09-02", "2025-09-02"): "IIITH Foundation Day",
    ("2025-10-11", "2025-10-11"): "Research Fest",
}


def _in_ranges(d, ranges):
    return any(pd.Timestamp(a) <= d <= pd.Timestamp(b) for a, b in ranges)


def daily_flags(days):
    """days: a DatetimeIndex (naive IST dates). Returns a DataFrame indexed by `days`."""
    days = pd.DatetimeIndex(pd.Series(days).dt.normalize().unique()).sort_values()
    rows = []
    for d in days:
        iso = d.strftime("%Y-%m-%d")
        hol = iso in HOLIDAYS
        insess = _in_ranges(d, SEMESTERS)
        exam = insess and _in_ranges(d, EXAMS)
        ev = next((name for (a, b), name in EVENTS.items() if pd.Timestamp(a) <= d <= pd.Timestamp(b)), None)
        day_type = ("holiday" if hol else "exam" if exam else "event" if ev else
                    "class" if insess else "vacation")
        rows.append({"in_session": int(insess), "holiday": int(hol), "exam": int(exam),
                     "event": int(ev is not None), "day_type": day_type,
                     "holiday_name": HOLIDAYS.get(iso, ""), "event_name": ev or ""})
    return pd.DataFrame(rows, index=days)


if __name__ == "__main__":
    f = daily_flags(pd.date_range("2025-01-01", "2025-12-31", freq="D"))
    f.to_csv("campus_calendar_2025.csv")
    print(f["day_type"].value_counts().to_dict(), "\nholidays:", int(f.holiday.sum()),
          "exam days:", int(f.exam.sum()), "event days:", int(f.event.sum()))
