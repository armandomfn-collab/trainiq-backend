"""
Workout Builder — converte descrição em linguagem natural para estrutura
de treino no formato TrainingPeaks, usando Claude como parser inteligente.
"""

import json
import os
import anthropic

# Zonas do atleta (Armando) mapeadas para % do limiar
# Run → percentOfThresholdHr  (limiar FC run = 160bpm)
# Bike → percentOfFtp         (FTP = 234W)
# Swim → percentOfThresholdPace (CSS = 1:40/100m)

ZONE_MAP = {
    "run": {
        "metric": "percentOfThresholdHr",
        "Z1": (75, 84),
        "Z2": (85, 89),
        "Z3": (90, 94),
        "Z4": (95, 99),
        "Z5": (100, 110),
    },
    "bike": {
        "metric": "percentOfFtp",
        "Z1": (40, 55),
        "Z2": (56, 75),
        "Z3": (76, 90),
        "Z4": (91, 105),
        "Z5": (106, 120),
    },
    "swim": {
        "metric": "percentOfThresholdPace",
        "Z1": (75, 84),
        "Z2": (85, 89),
        "Z3": (90, 94),
        "Z4": (95, 100),
        "Z5": (101, 110),
    },
}

_PARSE_PROMPT = """Você é um parser de treinos estruturados para TrainingPeaks.
Converta a descrição do treino para JSON no formato exato abaixo.

ZONAS DO ATLETA:
{zone_context}

FORMATO DE SAÍDA (JSON puro, sem markdown):
{{
  "titulo": "string — título curto e descritivo",
  "primaryIntensityMetric": "{metric}",
  "steps": [
    {{
      "name": "string",
      "type": "step",
      "duration_seconds": number,
      "intensity_min": number,
      "intensity_max": number,
      "intensityClass": "warmUp" | "active" | "rest" | "coolDown"
    }},
    {{
      "type": "repetition",
      "name": "string",
      "reps": number,
      "steps": [
        {{
          "name": "string",
          "type": "step",
          "duration_seconds": number,
          "intensity_min": number,
          "intensity_max": number,
          "intensityClass": "active" | "rest"
        }}
      ]
    }}
  ]
}}

REGRAS:
- Aquecimento → intensityClass "warmUp"
- Blocos de esforço → "active"
- Recuperação dentro de série → "rest"
- Volta à calma / solto → "coolDown"
- Blocos repetidos → use type "repetition" com reps e steps aninhados
- Converta minutos para segundos (ex: 12' = 720s)
- Use as zonas do atleta para definir intensity_min e intensity_max
- Se a zona não for especificada, infira pelo contexto (ex: "solto" = Z1/Z2)
- Responda SOMENTE com JSON válido, sem explicação"""


def _get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _zone_context(sport: str) -> str:
    sport_key = sport.lower()
    if sport_key not in ZONE_MAP:
        sport_key = "run"
    zones = ZONE_MAP[sport_key]
    metric = zones["metric"]
    lines = [f"Métrica: {metric}"]
    for z in ["Z1", "Z2", "Z3", "Z4", "Z5"]:
        mn, mx = zones[z]
        lines.append(f"  {z}: {mn}–{mx}%")
    return "\n".join(lines)


def parse_workout_text(
    description: str,
    sport: str = "Run",
) -> dict:
    """
    Usa Claude para converter descrição em linguagem natural para
    estrutura de treino no formato TrainingPeaks.

    Returns:
        dict com 'titulo', 'primaryIntensityMetric', 'steps' e
        metadados 'duration_min', 'tss_estimado'.
    """
    sport_key = sport.lower()
    if sport_key not in ZONE_MAP:
        sport_key = "run"

    zones = ZONE_MAP[sport_key]
    metric = zones["metric"]
    zone_ctx = _zone_context(sport)

    prompt = _PARSE_PROMPT.format(
        zone_context=zone_ctx,
        metric=metric,
    )

    response = _get_client().messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        system=prompt,
        messages=[{"role": "user", "content": f"Esporte: {sport}\n\nDescrição do treino:\n{description}"}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1]).strip()

    parsed = json.loads(text)

    # Calcula duração total em minutos
    total_seconds = _calc_duration(parsed.get("steps", []))
    parsed["duration_min"] = round(total_seconds / 60)

    return parsed


def _calc_duration(steps: list) -> int:
    total = 0
    for step in steps:
        if step.get("type") == "repetition":
            reps = step.get("reps", 1)
            inner = _calc_duration(step.get("steps", []))
            total += reps * inner
        else:
            total += step.get("duration_seconds", 0)
    return total


def build_structure_payload(parsed: dict) -> dict:
    """
    Extrai apenas os campos que o tp_create_workout espera no
    parâmetro 'structure' (sem 'titulo' e metadados).
    """
    return {
        "primaryIntensityMetric": parsed["primaryIntensityMetric"],
        "steps": parsed["steps"],
    }
