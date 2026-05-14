"""SQLite database for device tokens and workout tracking."""

import os
import sqlite3
from pathlib import Path

# Em produção (Railway) usa /data (volume persistente). Localmente usa a pasta do backend.
_data_dir = os.environ.get("DATA_DIR", str(Path(__file__).parent.parent))
DB_PATH = Path(_data_dir) / "trainiq.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    # Garante que o diretório de dados existe
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS device_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS processed_workouts (
                workout_id TEXT PRIMARY KEY,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                review_sent INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS daily_analysis (
                date TEXT PRIMARY KEY,
                analysis_json TEXT,
                notification_sent INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS body_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                weight_kg REAL NOT NULL,
                body_fat_pct REAL,
                muscle_mass_kg REAL,
                visceral_fat INTEGER,
                water_pct REAL,
                bone_mass_kg REAL,
                bmr_kcal INTEGER,
                bmi REAL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS athlete_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                body_part TEXT,
                description TEXT NOT NULL,
                severity TEXT DEFAULT 'moderate',
                status TEXT DEFAULT 'active',
                first_reported TEXT DEFAULT (datetime('now')),
                last_updated TEXT DEFAULT (datetime('now')),
                last_asked TEXT
            );

            CREATE TABLE IF NOT EXISTS athlete_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                name TEXT,
                age INTEGER,
                gender TEXT,
                height_cm REAL,
                resting_hr INTEGER,
                hrv_baseline REAL,
                sleep_hours_target REAL,
                ftp_watts INTEGER,
                threshold_pace_run TEXT,
                css_swim TEXT,
                hr_zones TEXT,
                weekly_schedule TEXT,
                target_race TEXT,
                race_date TEXT,
                race_distance TEXT,
                notes TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            );
        """)


def save_device_token(token: str):
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO device_tokens (token) VALUES (?)", (token,)
        )


def get_all_tokens() -> list[str]:
    with get_db() as conn:
        rows = conn.execute("SELECT token FROM device_tokens").fetchall()
        return [row["token"] for row in rows]


def is_workout_processed(workout_id: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM processed_workouts WHERE workout_id = ? AND review_sent = 1",
            (workout_id,)
        ).fetchone()
        return row is not None


def mark_workout_processed(workout_id: str):
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO processed_workouts (workout_id, review_sent) VALUES (?, 1)",
            (workout_id,)
        )


# ─── Body measurements ────────────────────────────────────────────────────────

def save_body_measurement(data: dict) -> int:
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO body_measurements
              (date, weight_kg, body_fat_pct, muscle_mass_kg,
               visceral_fat, water_pct, bone_mass_kg, bmr_kcal, bmi, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["date"],
            data["weight_kg"],
            data.get("body_fat_pct"),
            data.get("muscle_mass_kg"),
            data.get("visceral_fat"),
            data.get("water_pct"),
            data.get("bone_mass_kg"),
            data.get("bmr_kcal"),
            data.get("bmi"),
            data.get("notes"),
        ))
        return cur.lastrowid


def get_body_measurements(days: int = 180) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM body_measurements
            WHERE date >= date('now', ?)
            ORDER BY date DESC
        """, (f"-{days} days",)).fetchall()
        return [dict(r) for r in rows]


def delete_body_measurement(measure_id: int) -> bool:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM body_measurements WHERE id = ?", (measure_id,))
        return cur.rowcount > 0


# ─── Chat history ─────────────────────────────────────────────────────────────

def save_chat_message(role: str, content: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chat_messages (role, content) VALUES (?, ?)",
            (role, content)
        )


def get_chat_history(limit: int = 40) -> list[dict]:
    """Retorna as últimas N mensagens no formato {role, content}."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_messages ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    # Inverte para ordem cronológica
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def clear_chat_history():
    with get_db() as conn:
        conn.execute("DELETE FROM chat_messages")


# ─── Athlete alerts (lesões, dores, eventos significativos) ───────────────────

def get_active_alerts() -> list[dict]:
    """Retorna todos os alertas ativos do atleta."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM athlete_alerts WHERE status != 'resolved' ORDER BY last_updated DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_alert(category: str, body_part: str | None, description: str,
                 severity: str = "moderate") -> int:
    """Cria ou atualiza um alerta. Usa body_part+category como chave de deduplicação."""
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM athlete_alerts WHERE category=? AND body_part=? AND status!='resolved'",
            (category, body_part or "")
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE athlete_alerts SET description=?, severity=?, last_updated=datetime('now') WHERE id=?",
                (description, severity, existing["id"])
            )
            return existing["id"]
        else:
            cur = conn.execute(
                "INSERT INTO athlete_alerts (category, body_part, description, severity) VALUES (?,?,?,?)",
                (category, body_part or "", description, severity)
            )
            return cur.lastrowid


def resolve_alert(alert_id: int):
    """Marca um alerta como resolvido."""
    with get_db() as conn:
        conn.execute(
            "UPDATE athlete_alerts SET status='resolved', last_updated=datetime('now') WHERE id=?",
            (alert_id,)
        )


def mark_alert_asked(alert_id: int):
    """Registra que o coach perguntou sobre esse alerta agora."""
    with get_db() as conn:
        conn.execute(
            "UPDATE athlete_alerts SET last_asked=datetime('now') WHERE id=?",
            (alert_id,)
        )


def resolve_alert_by_key(category: str, body_part: str):
    """Resolve alerta por category+body_part."""
    with get_db() as conn:
        conn.execute(
            "UPDATE athlete_alerts SET status='resolved', last_updated=datetime('now') WHERE category=? AND body_part=? AND status!='resolved'",
            (category, body_part or "")
        )


# ─── Athlete profile ──────────────────────────────────────────────────────────

import json as _json

def save_athlete_profile(data: dict):
    """Upsert: sempre sobrescreve o único registro (id=1)."""
    hr_zones       = _json.dumps(data.get("hr_zones") or {}, ensure_ascii=False)
    weekly_schedule = _json.dumps(data.get("weekly_schedule") or {}, ensure_ascii=False)
    with get_db() as conn:
        conn.execute("""
            INSERT INTO athlete_profile
              (id, name, age, gender, height_cm,
               resting_hr, hrv_baseline, sleep_hours_target,
               ftp_watts, threshold_pace_run, css_swim,
               hr_zones, weekly_schedule,
               target_race, race_date, race_distance, notes,
               updated_at)
            VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
            ON CONFLICT(id) DO UPDATE SET
              name=excluded.name, age=excluded.age, gender=excluded.gender,
              height_cm=excluded.height_cm,
              resting_hr=excluded.resting_hr, hrv_baseline=excluded.hrv_baseline,
              sleep_hours_target=excluded.sleep_hours_target,
              ftp_watts=excluded.ftp_watts,
              threshold_pace_run=excluded.threshold_pace_run,
              css_swim=excluded.css_swim,
              hr_zones=excluded.hr_zones,
              weekly_schedule=excluded.weekly_schedule,
              target_race=excluded.target_race,
              race_date=excluded.race_date,
              race_distance=excluded.race_distance,
              notes=excluded.notes,
              updated_at=datetime('now')
        """, (
            data.get("name"), data.get("age"), data.get("gender"), data.get("height_cm"),
            data.get("resting_hr"), data.get("hrv_baseline"), data.get("sleep_hours_target"),
            data.get("ftp_watts"), data.get("threshold_pace_run"), data.get("css_swim"),
            hr_zones, weekly_schedule,
            data.get("target_race"), data.get("race_date"), data.get("race_distance"),
            data.get("notes"),
        ))


def get_athlete_profile() -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM athlete_profile WHERE id = 1").fetchone()
    if not row:
        return None
    d = dict(row)
    d["hr_zones"]        = _json.loads(d["hr_zones"] or "{}")
    d["weekly_schedule"] = _json.loads(d["weekly_schedule"] or "{}")
    return d
