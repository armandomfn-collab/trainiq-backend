"""Garmin LiveTrack real-time monitor + AI coaching alerts."""

import asyncio
import os
import re
from datetime import datetime
from typing import Optional

import anthropic
import httpx

GARMIN_TP_URL = "https://livetrack.garmin.com/services/session/{session_id}/trackpoints"

COACH_SYSTEM = """Voce e um coach de triathlon falando diretamente no ouvido do atleta durante a atividade.
Regras:
- MAX 2 frases curtissimas. Direto. Sem enrolacao.
- Usa os numeros reais que te passarem (FC, pace, distancia).
- Voz imperativa: "Hidrata agora.", "Reduz o ritmo.", "Mantem esse pace."
- Nao repete o que ja foi dito antes.
"""

LIVETRACK_URL_RE = re.compile(
    r"https://livetrack\.garmin\.com/session/([a-f0-9\-]+)/token/([A-Za-z0-9]+)"
)


def parse_livetrack_url(url: str) -> tuple[str, str]:
    """Extract (session_id, token) from a LiveTrack URL."""
    m = LIVETRACK_URL_RE.search(url)
    if not m:
        raise ValueError("URL do LiveTrack invalida. Formato esperado: https://livetrack.garmin.com/session/.../token/...")
    return m.group(1), m.group(2)


class LiveTrackSession:
    def __init__(self, session_id: str, token: str, workout_coach=None):
        self.session_id    = session_id
        self.token         = token
        self.active        = False
        self._task: Optional[asyncio.Task] = None

        # Dados acumulados
        self.all_points:   list  = []
        self.last_tp_idx:  int   = 0
        self.start_time:   Optional[datetime] = None
        self.last_hydration: Optional[datetime] = None
        self.alert_counter: int  = 0  # polls desde ultimo alerta

        # Coach estruturado (opcional)
        self.coach = workout_coach  # WorkoutCoach | None

        # Estado exposto ao app
        self.current_metrics: dict         = {}
        self.latest_alert:    Optional[str] = None
        self.alert_time:      Optional[datetime] = None
        self.status:          str           = "waiting"  # waiting | active | ended
        self.current_block:   Optional[dict] = None

    # ── Garmin API ────────────────────────────────────────────────────────
    async def _fetch(self) -> list:
        url = GARMIN_TP_URL.format(session_id=self.session_id)
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                r = await client.get(url, params={
                    "token": self.token,
                    "from":  self.last_tp_idx,
                }, headers={
                    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
                    "Accept":     "application/json, */*",
                    "Referer":    "https://livetrack.garmin.com/",
                })
                if r.status_code == 200:
                    data = r.json()
                    return data.get("trackPoints") or data.get("trackpoints") or []
                if r.status_code == 404:
                    return []   # sessao ainda nao iniciou ou ja encerrou
        except Exception as e:
            print(f"[LiveTrack] fetch error: {e}")
        return []

    # ── Métricas ──────────────────────────────────────────────────────────
    def _update_metrics(self, points: list) -> dict:
        if not points:
            return self.current_metrics
        p = points[-1]

        m: dict = {}
        if hr := (p.get("heartRate") or p.get("heart_rate")):
            m["heart_rate"] = int(hr)
        if spd := p.get("speed"):          # m/s
            m["speed_ms"] = round(spd, 2)
            if spd > 0.3:
                m["pace_min_km"] = round(1000 / spd / 60, 2)
        if dist := (p.get("totalDistance") or p.get("distance")):
            m["distance_km"] = round(dist / 1000, 2)
        if pwr := p.get("power"):
            m["power_w"] = int(pwr)
        if cad := p.get("cadence"):
            m["cadence"] = int(cad)
        if self.start_time:
            m["elapsed_min"] = round((datetime.now() - self.start_time).seconds / 60, 1)

        self.current_metrics = m
        return m

    # ── Hidratação ────────────────────────────────────────────────────────
    def _hydration_due(self) -> bool:
        if not self.start_time:
            return False
        elapsed = (datetime.now() - self.start_time).seconds
        if self.last_hydration is None:
            return elapsed >= 1200   # 20 min de atividade
        return (datetime.now() - self.last_hydration).seconds >= 1200

    # ── Alerta Claude ─────────────────────────────────────────────────────
    async def _generate_alert(self, metrics: dict) -> str:
        client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        hydration = self._hydration_due()

        prompt = (
            f"FC: {metrics.get('heart_rate','—')} bpm | "
            f"Pace: {metrics.get('pace_min_km','—')} min/km | "
            f"Distancia: {metrics.get('distance_km','—')} km | "
            f"Tempo: {metrics.get('elapsed_min','—')} min | "
            f"Potencia: {metrics.get('power_w','—')} W | "
            f"Cadencia: {metrics.get('cadence','—')}\n"
        )
        if hydration:
            prompt += "PRIORIDADE: hidratacao pendente — 20 min sem hidratacao.\n"

        prompt += "Gere o alerta de coaching agora. Responda SOMENTE o texto do alerta."

        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=80,
            system=COACH_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        if hydration:
            self.last_hydration = datetime.now()
        return response.content[0].text.strip()

    def _set_alert(self, text: str):
        self.latest_alert = text
        self.alert_time   = datetime.now()
        print(f"[LiveTrack] Coach: {text}")

    # ── Loop principal ────────────────────────────────────────────────────
    async def _poll_loop(self):
        self.start_time   = datetime.now()
        alert_interval    = 0
        consecutive_empty = 0

        # Anuncia abertura do treino se tiver coach estruturado
        if self.coach:
            self._set_alert(self.coach.abertura)
            await asyncio.sleep(5)

        while self.active:
            try:
                new_pts = await self._fetch()

                if new_pts:
                    consecutive_empty = 0
                    self.status = "active"
                    self.all_points.extend(new_pts)
                    self.last_tp_idx += len(new_pts)
                    metrics = self._update_metrics(new_pts)
                    elapsed_min = metrics.get("elapsed_min", 0)

                    # ── Modo coach estruturado (treino do TP) ────────────
                    if self.coach:
                        coach_alert = self.coach.process_tick(elapsed_min, metrics)
                        if coach_alert:
                            self._set_alert(coach_alert)
                        block = self.coach.current_block
                        if block:
                            self.current_block = {
                                "index":           block.index,
                                "name":            block.name,
                                "duration_min":    block.duration_min,
                                "zone":            block.zone,
                                "target_pace_min": block.target_pace_min,
                                "target_pace_max": block.target_pace_max,
                            }
                        if self._hydration_due():
                            self._set_alert("Hora de hidratar! Toma agua agora.")
                            self.last_hydration = datetime.now()

                    # ── Modo livre (sem treino do TP) ────────────────────
                    else:
                        alert_interval += 1
                        if alert_interval >= 4 or self._hydration_due():
                            try:
                                alert = await self._generate_alert(metrics)
                                self._set_alert(alert)
                                alert_interval = 0
                            except Exception as e:
                                print(f"[LiveTrack] alert error: {e}")

                else:
                    consecutive_empty += 1
                    if consecutive_empty >= 20:
                        self.status = "ended"
                        print("[LiveTrack] Sessao encerrada (sem dados por 10 min)")
                        break

            except Exception as e:
                print(f"[LiveTrack] poll error: {e}")

            await asyncio.sleep(30)

    def start(self):
        self.active = True
        self.status = "waiting"
        self._task  = asyncio.create_task(self._poll_loop())
        print(f"[LiveTrack] Monitorando sessao {self.session_id}")

    def stop(self):
        self.active = False
        self.status = "ended"
        if self._task:
            self._task.cancel()
        print("[LiveTrack] Monitoramento encerrado")

    def to_dict(self) -> dict:
        blocks_data = []
        if self.coach:
            for b in self.coach.blocks:
                blocks_data.append({
                    "index": b.index, "name": b.name,
                    "duration_min": b.duration_min, "zone": b.zone,
                    "target_pace_min": b.target_pace_min,
                    "target_pace_max": b.target_pace_max,
                })
        return {
            "status":          self.status,
            "session_id":      self.session_id,
            "current_metrics": self.current_metrics,
            "latest_alert":    self.latest_alert,
            "alert_time":      self.alert_time.isoformat() if self.alert_time else None,
            "total_points":    len(self.all_points),
            "current_block":   self.current_block,
            "workout_blocks":  blocks_data,
            "start_time":      self.start_time.isoformat() if self.start_time else None,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_current: Optional[LiveTrackSession] = None


def get_current() -> Optional[LiveTrackSession]:
    return _current


def set_current(session: Optional[LiveTrackSession]):
    global _current
    _current = session
