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
