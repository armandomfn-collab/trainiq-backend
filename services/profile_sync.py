"""Auto-build athlete profile from TrainingPeaks data + stored measurements."""

from collections import defaultdict
from datetime import timedelta, timezone

_BRT = timezone(__import__("datetime").timedelta(hours=-3))
def _now_brt():
    from datetime import datetime
    return datetime.now(_BRT)

from tp_mcp.tools.settings import tp_get_athlete_settings
from tp_mcp.tools.workouts import tp_get_workouts

from services.database import get_athlete_profile, save_athlete_profile, get_body_measurements

DAY_MAP = {0: "segunda", 1: "terca", 2: "quarta", 3: "quinta", 4: "sexta", 5: "sabado", 6: "domingo"}

# workoutTypeId mapping from TP
WT_RUN  = 3
WT_BIKE = 2
WT_SWIM = 1

SPORT_NAME_MAP: dict[str, str] = {
    "run":      "Run",
    "bike":     "Bike",
    "swim":     "Swim",
    "strength": "Strength",
    "walk":     "Walk",
    "brick":    "Brick",
}


def _mps_to_run_pace(mps: float) -> str:
    """Convert m/s to MM:SS/km string."""
    if not mps or mps <= 0:
        return ""
    secs = round(1000 / mps)
    return f"{secs // 60}:{secs % 60:02d}/km"


def _mps_to_swim_pace(mps: float) -> str:
    """Convert m/s to M:SS/100m string."""
    if not mps or mps <= 0:
        return ""
    secs = round(100 / mps)
    return f"{secs // 60}:{secs % 60:02d}/100m"


def _extract_from_settings(settings: dict) -> dict:
    """Extract FTP, HR zones, pace thresholds, personal info from TP settings."""
    result: dict = {}

    # Personal
    result["name"]   = (settings.get("firstName") or "").title() or None
    result["age"]    = settings.get("age")
    result["gender"] = "Masculino" if settings.get("gender") == "m" else (
                       "Feminino"  if settings.get("gender") == "f" else None)

    # FTP — power zones workoutTypeId=2 (bike)
    for pz in settings.get("powerZones", []):
        if pz.get("workoutTypeId") == WT_BIKE and pz.get("threshold"):
            result["ftp_watts"] = int(pz["threshold"])
            break

    # HR zones — use run zones (workoutTypeId=3) as primary
    for hz in settings.get("heartRateZones", []):
        if hz.get("workoutTypeId") == WT_RUN:
            zones = hz.get("zones", [])
            if zones:
                result["resting_hr"] = hz.get("restingHeartRate") or None
                result["hr_zones"] = {
                    "z1": zones[0]["maximum"] if len(zones) > 0 else None,
                    "z2": zones[1]["maximum"] if len(zones) > 1 else None,
                    "z3": zones[2]["maximum"] if len(zones) > 2 else None,
                    "z4": zones[3]["maximum"] if len(zones) > 3 else None,
                    "z5": zones[4]["maximum"] if len(zones) > 4 else None,
                }
            break

    # Run threshold pace — speed zone workoutTypeId=0
    for sz in settings.get("speedZones", []):
        if sz.get("workoutTypeId") == 0 and sz.get("threshold"):
            result["threshold_pace_run"] = _mps_to_run_pace(sz["threshold"])
            break

    # CSS swim — speed zone workoutTypeId=1
    for sz in settings.get("speedZones", []):
        if sz.get("workoutTypeId") == 1 and sz.get("threshold"):
            result["css_swim"] = _mps_to_swim_pace(sz["threshold"])
            break

    return result


def _infer_weekly_schedule(workouts: list[dict]) -> dict:
    """Analyse last N weeks of workouts to infer preferred training days & sports."""
    day_sports: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for w in workouts:
        raw_date = w.get("workout_date") or w.get("date") or ""
        if not raw_date:
            continue
        try:
            d = date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            continue

        day_key = DAY_MAP[d.weekday()]
        sport_raw = (w.get("sport") or w.get("workout_type") or "").lower().strip()
        sport = SPORT_NAME_MAP.get(sport_raw)
        if sport:
            day_sports[day_key][sport] += 1

    schedule: dict = {}
    for day_key in DAY_MAP.values():
        sports = [sp for sp, cnt in day_sports[day_key].items() if cnt >= 2]
        if sports:
            schedule[day_key] = {"hora": "", "esportes": sports}

    return schedule


def _merge(existing: dict, fresh: dict) -> dict:
    """Merge fresh TP data into existing profile.

    Rules:
    - Always update TP-sourced performance fields (ftp, hr_zones, pace thresholds)
    - Preserve user-entered fields (height, HRV, sleep, target_race, notes, hours in schedule)
    - For weekly_schedule: update sports lists from TP but keep user-set hours
    """
    merged = dict(existing)

    # Always overwrite these from TP
    for key in ("ftp_watts", "hr_zones", "threshold_pace_run", "css_swim", "resting_hr"):
        if fresh.get(key) is not None:
            merged[key] = fresh[key]

    # Only fill if not set by user
    for key in ("name", "age", "gender"):
        if not merged.get(key) and fresh.get(key) is not None:
            merged[key] = fresh[key]

    # Weekly schedule: update sports, preserve hours
    fresh_sched  = fresh.get("weekly_schedule") or {}
    merged_sched = merged.get("weekly_schedule") or {}
    new_sched = dict(merged_sched)
    for day_key, fresh_day in fresh_sched.items():
        existing_day = merged_sched.get(day_key, {})
        new_sched[day_key] = {
            "hora":     existing_day.get("hora", ""),
            "esportes": fresh_day["esportes"],
        }
    merged["weekly_schedule"] = new_sched

    # Weight from latest body measurement
    if fresh.get("weight_kg") and not merged.get("weight_kg"):
        merged["weight_kg"] = fresh["weight_kg"]

    return merged


async def sync_profile_from_tp() -> dict:
    """Fetch TP data, analyse patterns, upsert profile. Returns saved profile."""

    today      = _now_brt().date().isoformat()
    eight_ago  = (_now_brt().date() - timedelta(days=56)).isoformat()

    # Fetch in parallel via asyncio.gather
    import asyncio
    settings_raw, workouts_raw = await asyncio.gather(
        tp_get_athlete_settings(),
        tp_get_workouts(eight_ago, today),
    )

    settings = settings_raw.get("settings", {})
    workouts = workouts_raw.get("workouts", [])

    fresh = _extract_from_settings(settings)
    fresh["weekly_schedule"] = _infer_weekly_schedule(workouts)

    # Latest body measurement → weight
    measurements = get_body_measurements(days=90)
    if measurements:
        fresh["weight_kg"] = measurements[0].get("weight_kg")

    existing = get_athlete_profile() or {}
    merged   = _merge(existing, fresh)

    save_athlete_profile(merged)
    return get_athlete_profile() or {}
