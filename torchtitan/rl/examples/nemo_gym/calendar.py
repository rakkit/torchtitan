# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Calendar scheduling grader: a port of NeMo Gym's resources_servers/calendar (utils.py,
Apache-2.0). The answer must print the whole calendar as a JSON list of events
(``event_id``, ``start_time``, ``duration``, ...); it passes when it has exactly the expected
events, none overlap, and each keeps its duration, its time window and its constraint
("before T" / "after T" / "between T1 and T2" / "at T")."""

from __future__ import annotations

import json
import re

# the first JSON list holding at least one object (NeMo Gym's extract_json_list)
_JSON_LIST = re.compile(
    r"\[(?:[^\[\]]|\{[^}]*\})*\{(?:[^\[\]]|\{[^}]*\})*\}(?:[^\[\]]|\{[^}]*\})*\]",
    re.DOTALL,
)


def grade_calendar(response: str, expected: dict) -> float:
    """1.0 when the answer's calendar meets every expected event, else 0 (NeMo Gym's
    grade_assistant_response with allow_no_json_list=False; an empty answer scores 0 first,
    as its verify does)."""
    if not response or "<think>" in response:
        return 0.0
    if not expected:
        return 1.0  # nothing to schedule
    try:
        match = _JSON_LIST.search(response)
        events = json.loads(match.group(0)) if match else None
        if not events:
            return 0.0
        by_id: dict[str, dict] = {}
        for event in events:
            event_id = str(event["event_id"])
            if event_id in by_id:
                return 0.0
            by_id[event_id] = event
        if len(by_id) != len(expected):
            return 0.0
        for event in events:
            if _overlaps(events, event):
                return 0.0
        for event_id, want in expected.items():
            if not _satisfied(by_id[event_id], want):
                return 0.0
    except Exception:  # malformed events (missing keys, bad times): NeMo Gym's error_in_grading
        return 0.0
    return 1.0


def _minutes(text: str) -> int:
    """'10:30am' / '2pm' / '14:15' -> minutes after midnight."""
    text = text.strip()
    if "am" in text or "pm" in text:
        pm = "pm" in text
        clock = text.replace("am", "").replace("pm", "")
        hour, minute = map(int, clock.split(":")) if ":" in clock else (int(clock), 0)
        return (hour * 60 if hour != 12 else 0) + (12 * 60 if pm else 0) + minute
    hours, minutes = map(int, text.split(":"))
    return hours * 60 + minutes


def _overlaps(events: list[dict], event: dict) -> bool:
    start = _minutes(event["start_time"])
    end = start + event["duration"]
    for other in events:
        if other == event:
            continue
        other_start = _minutes(other["start_time"])
        if not (end <= other_start or start >= other_start + other["duration"]):
            return True
    return False


def _satisfied(event: dict, want: dict) -> bool:
    if event["duration"] != want["duration"]:
        return False
    start = _minutes(event["start_time"])
    end = start + event["duration"]
    if start < _minutes(want["min_time"]) or end > _minutes(want["max_time"]):
        return False
    constraint = want["constraint"]
    if constraint is None:
        return True
    if constraint.startswith("before "):
        return end <= _minutes(constraint[len("before ") :])
    if constraint.startswith("after "):
        return start >= _minutes(constraint[len("after ") :])
    if constraint.startswith("between "):
        low, high = constraint[len("between ") :].split(" and ")
        return start >= _minutes(low) and end <= _minutes(high)
    if constraint.startswith("at "):
        return start == _minutes(constraint[len("at ") :])
    return True
