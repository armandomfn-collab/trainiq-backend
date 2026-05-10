"""
Workout-aware coach for LiveTrack sessions.
Parses TP workout into timed blocks and monitors pace/HR compliance.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import anthropic


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class WorkoutBlock:
    index:        int
    name:         str          # "Aquecimento", "Bloco 1", "Recuperacao", etc.
    duration_min: float
    target_pace_min: Optional[float] = None   # min/km (lower bound — faster)
    target_pace_max: Optional[float] = None   # min/km (upper bound — slower)
    target_hr_min:   Optional[int]   = None
    target_hr_max:   Optional[int]   = None
    zone:            Optional[str]   = None   # "Z1", "Z2", "Z3", "Z4", "Z5"
    description:     str             = ""
    tip:             str             = ""     # dica de execucao

@dataclass
class BlockState:
    block:            WorkoutBlock
    started_at_min:   float          # minuto da atividade em que o bloco começou
    pace_violations:  int  = 0       # contagem de polls com pace fora da zona
    last_alert_min:   float = -99    # minuto do ultimo alerta dentro do bloco
    announced_start:  bool  = False
    announced_end:    bool  = False
    paces:            list  = field(default_factory=list)   # paces coletados
    hrs:              list  = field(default_factory=list)   # HRs coletados


# ── Parser ────────────────────────────────────────────────────────────────────

PARSE_SYSTEM = """Voce e um coach de triathlon que analisa treinos e extrai blocos estruturados.
REGRAS:
- Se nao houver pace explicito, infira pela zona (Z1=6:30-7:30, Z2=5:45-6:30, Z3=5:10-5:45, Z4=4:40-5:10, Z5=<4:40)
- Se nao houver zona, infira pelo contexto (aquecimento=Z1-Z2, recuperacao=Z1, tiro=Z4-Z5)
- HR: Z1<115, Z2=115-135, Z3=135-155, Z4=155-170, Z5>170
- tip: dica pratica de execucao para cada bloco (nao generica)
- Abertura: entusiasta mas direto, max 2 frases
"""

# Tool definition — garante JSON válido via protocol
PARSE_TOOL = {
    "name": "estruturar_treino",
    "description": "Estrutura o treino em blocos com metadados de coaching",
    "input_schema": {
        "type": "object",
        "properties": {
            "nome_treino": {"type": "string"},
            "abertura":    {"type": "string"},
            "blocos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index":           {"type": "integer"},
                        "name":            {"type": "string"},
                        "duration_min":    {"type": "number"},
                        "target_pace_min": {"type": "number"},
                        "target_pace_max": {"type": "number"},
                        "target_hr_min":   {"type": "integer"},
                        "target_hr_max":   {"type": "integer"},
                        "zone":            {"type": "string"},
                        "description":     {"type": "string"},
                        "tip":             {"type": "string"},
                    },
                    "required": ["index", "name", "duration_min"],
                },
            },
        },
        "required": ["nome_treino", "abertura", "blocos"],
    },
}


def parse_workout_into_blocks(
    title: str,
    description: str,
    duration_planned_h: float,
    tss_planned: Optional[float],
) -> dict:
    """Use Claude tool_use to parse a TP workout description into structured blocks."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = (
        f"TREINO: {title}\n"
        f"DURACAO TOTAL: {round(duration_planned_h * 60)} minutos\n"
        f"TSS PLANEJADO: {tss_planned or 'N/A'}\n"
        f"DESCRICAO:\n{description or '(sem descricao — infira blocos tipicos pelo titulo e duracao)'}\n\n"
        "Estruture esse treino em blocos usando a ferramenta estruturar_treino."
    )

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=PARSE_SYSTEM,
        tools=[PARSE_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": prompt}],
    )

    # Extrai o input da tool_use — sempre JSON válido garantido pela API
    for block in response.content:
        if block.type == "tool_use" and block.name == "estruturar_treino":
            data = block.input
            break
    else:
        raise ValueError("Claude nao chamou a ferramenta estruturar_treino")

    # Converte para dataclasses
    blocks = []
    for b in data.get("blocos", []):
        blocks.append(WorkoutBlock(
            index=b["index"],
            name=b["name"],
            duration_min=b["duration_min"],
            target_pace_min=b.get("target_pace_min"),
            target_pace_max=b.get("target_pace_max"),
            target_hr_min=b.get("target_hr_min"),
            target_hr_max=b.get("target_hr_max"),
            zone=b.get("zone"),
            description=b.get("description", ""),
            tip=b.get("tip", ""),
        ))

    return {
        "nome": data.get("nome_treino", title),
        "blocks": blocks,
        "abertura": data.get("abertura", f"Bom dia! Treino de hoje: {title}. Vamos la!"),
        "total_min": sum(b.duration_min for b in blocks),
    }


# ── Coach ─────────────────────────────────────────────────────────────────────

def fmt_pace(pace_min_km: Optional[float]) -> str:
    if not pace_min_km:
        return "—"
    m = int(pace_min_km)
    s = round((pace_min_km - m) * 60)
    return f"{m}:{s:02d}/km"


class WorkoutCoach:
    def __init__(self, workout_data: dict):
        self.nome:   str            = workout_data["nome"]
        self.blocks: list[WorkoutBlock] = workout_data["blocks"]
        self.abertura: str          = workout_data["abertura"]
        self.total_blocks           = len(self.blocks)

        self._state: Optional[BlockState] = None
        self._current_idx: int = 0
        self._workout_done: bool = False

    @property
    def current_block(self) -> Optional[WorkoutBlock]:
        if 0 <= self._current_idx < len(self.blocks):
            return self.blocks[self._current_idx]
        return None

    def get_block_at_minute(self, elapsed_min: float) -> int:
        """Return which block index should be active at elapsed_min."""
        cumulative = 0.0
        for i, b in enumerate(self.blocks):
            cumulative += b.duration_min
            if elapsed_min < cumulative:
                return i
        return len(self.blocks) - 1  # ultimo bloco

    def process_tick(self, elapsed_min: float, metrics: dict) -> Optional[str]:
        """
        Called every poll. Returns a voice alert string or None.
        Manages block transitions and compliance monitoring.
        """
        if self._workout_done or not self.blocks:
            return None

        pace = metrics.get("pace_min_km")
        hr   = metrics.get("heart_rate")
        expected_idx = self.get_block_at_minute(elapsed_min)
        block = self.blocks[expected_idx]

        # ── Transição de bloco ───────────────────────────────────────────
        if expected_idx != self._current_idx or self._state is None:
            # Fecha bloco anterior
            end_msg = self._close_block(expected_idx)

            # Abre novo bloco
            self._current_idx = expected_idx
            start_offset = sum(b.duration_min for b in self.blocks[:expected_idx])
            self._state = BlockState(block=block, started_at_min=start_offset)
            self._state.announced_start = True

            start_msg = self._announce_block_start(block, expected_idx)
            return (end_msg + " " + start_msg).strip() if end_msg else start_msg

        # ── Coleta métricas do bloco atual ───────────────────────────────
        if pace:
            self._state.paces.append(pace)
        if hr:
            self._state.hrs.append(hr)

        # ── Verifica desvio de pace ──────────────────────────────────────
        alert = self._check_compliance(elapsed_min, pace, hr, block)
        return alert

    def _announce_block_start(self, block: WorkoutBlock, idx: int) -> str:
        is_last = idx == len(self.blocks) - 1
        suffix  = "Ultimo bloco!" if is_last else f"Bloco {idx + 1} de {self.total_blocks}."

        pace_info = ""
        if block.target_pace_min and block.target_pace_max:
            pace_info = f" Meta: {fmt_pace(block.target_pace_min)} a {fmt_pace(block.target_pace_max)}."
        elif block.zone:
            pace_info = f" Zona {block.zone}."

        dur = f"{int(block.duration_min)} minutos."
        tip = f" {block.tip}" if block.tip else ""

        return f"{suffix} {block.name} — {dur}{pace_info}{tip}"

    def _close_block(self, next_idx: int) -> Optional[str]:
        if self._state is None:
            return None
        prev = self._state.block

        # Calcula média de pace do bloco
        avg_pace = (sum(self._state.paces) / len(self._state.paces)) if self._state.paces else None
        in_zone  = True
        if avg_pace and prev.target_pace_min and prev.target_pace_max:
            in_zone = prev.target_pace_min <= avg_pace <= prev.target_pace_max

        if avg_pace:
            result = "bom ritmo" if in_zone else "fora da zona"
            return f"{prev.name} concluido. Media {fmt_pace(avg_pace)}, {result}."
        return f"{prev.name} concluido."

    def _check_compliance(
        self, elapsed_min: float, pace: Optional[float],
        hr: Optional[int], block: WorkoutBlock,
    ) -> Optional[str]:
        if not self._state:
            return None

        dur = block.duration_min  # duração do bloco em minutos

        # ── Cooldown proporcional: 5% da duração, mínimo 30s (0.5 min) ──
        cooldown_min = max(0.5, dur * 0.05)
        mins_since_alert = elapsed_min - self._state.last_alert_min
        if mins_since_alert < cooldown_min:
            return None

        # ── Violations needed: 1 por cada 10 min de bloco, mínimo 1 ─────
        # Tiro de 1 min → 1 poll fora já alerta
        # Bloco de 20 min → 2 polls consecutivos fora para alertar
        violations_needed = max(1, round(dur / 10))

        # ── Tom do alerta: curto = incentivo, longo = correção ───────────
        is_short = dur <= 3   # tiros e intervalos curtos

        # Verifica pace
        if pace and block.target_pace_min and block.target_pace_max:
            if pace < block.target_pace_min - 0.15:  # muito rapido
                self._state.pace_violations += 1
                if self._state.pace_violations >= violations_needed:
                    self._state.last_alert_min = elapsed_min
                    self._state.pace_violations = 0
                    if is_short:
                        return f"Segura! Pace {fmt_pace(pace)}, ta rapido demais. Controla."
                    else:
                        return f"Segura o ritmo! Pace {fmt_pace(pace)}, meta maxima {fmt_pace(block.target_pace_min)}."

            elif pace > block.target_pace_max + 0.2:  # muito lento
                self._state.pace_violations += 1
                if self._state.pace_violations >= violations_needed:
                    self._state.last_alert_min = elapsed_min
                    self._state.pace_violations = 0
                    if is_short:
                        # Em tiros, incentiva forte: pouco tempo restante
                        mins_in_block = elapsed_min - self._state.started_at_min
                        remaining = max(0, dur - mins_in_block)
                        if remaining <= 1:
                            return f"Vai! Ultimo minuto, tudo agora! Pace {fmt_pace(pace)}."
                        else:
                            secs = int(remaining * 60)
                            return f"Forca! Faltam {secs}s, sobe o ritmo! Meta {fmt_pace(block.target_pace_min)}–{fmt_pace(block.target_pace_max)}."
                    else:
                        return f"Acelera! Pace {fmt_pace(pace)}, meta {fmt_pace(block.target_pace_min)} a {fmt_pace(block.target_pace_max)}."
            else:
                self._state.pace_violations = 0  # reset se voltou na zona
                # Para tiros curtos, reforça positivamente quando na zona
                if is_short and mins_since_alert >= cooldown_min * 3:
                    self._state.last_alert_min = elapsed_min
                    return f"Isso! Pace {fmt_pace(pace)}, ta na zona. Sustenta!"

        # Verifica FC
        if hr and block.target_hr_max and hr > block.target_hr_max + 5:
            self._state.last_alert_min = elapsed_min
            if is_short:
                return f"FC em {hr}! Respira fundo, controla o esforco."
            else:
                return f"FC alta — {hr} bpm, acima do limite de {block.target_hr_max}. Reduz o esforco."

        return None

    def get_summary(self) -> list[WorkoutBlock]:
        return self.blocks
