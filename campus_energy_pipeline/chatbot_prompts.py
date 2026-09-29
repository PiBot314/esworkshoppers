from __future__ import annotations

from typing import Dict, List


def build_anomaly_prompt(row: Dict[str, float | str]) -> str:
    issues: List[str] = []
    actions: List[str] = []

    timestamp = str(row.get("timestamp", "unknown time"))
    if float(row.get("Voltage_max", 0.0)) > 260:
        issues.append("voltage spike above the normal campus service envelope")
        actions.append("check the feeder breaker, inspect busbar insulation, and verify the local transformer load balance")
    if float(row.get("B Power_max", 0.0)) > 800:
        issues.append("phase B demand surge is concentrated in the highest-load segment")
        actions.append("verify HVAC and lighting load scheduling on phase B and inspect for a stuck compressor or unexpected pump start")
    if float(row.get("R Power_max", 0.0)) > 400 or float(row.get("Y Power_max", 0.0)) > 250:
        issues.append("unbalanced phase loading suggests a localized energy waste event")
        actions.append("check for an unequal load distribution across the meter phases and review event schedules in the building automation system")
    if float(row.get("Power Factor_max", 0.0)) < 0.7:
        issues.append("poor power factor indicates reactive losses and underutilized electrical efficiency")
        actions.append("inspect capacitor banks, run power-factor correction checks, and confirm there is no motor idling or undervoltage issue")
    if float(row.get("energy_delta_kwh", 0.0)) > 25:
        issues.append("a large energy delta indicates non-trivial wasted draw over the 15-minute window")
        actions.append("compare this interval with the building energy baseline and review overnight equipment schedules for unnecessary operation")

    if not issues:
        issues.append("a moderate but non-routine deviation from the campus energy profile")
        actions.append("confirm the interval against baseline telemetry and inspect the connected equipment schedule before escalating maintenance")

    issue_text = "; ".join(issues)
    action_text = "; ".join(actions)
    return (
        f"At {timestamp}, the campus meter shows {issue_text}. "
        f"Recommended actions: {action_text}. "
        "This anomaly should be triaged first by facility staff to avoid excess demand and prevent equipment wear."
    )
