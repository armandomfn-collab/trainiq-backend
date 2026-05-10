"""AI analysis combining body composition with training load."""

import os
import json
import anthropic


def _get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


SYSTEM_PROMPT = """Voce e um coach de triathlon especializado em composicao corporal para atletas de endurance.

REGRAS:
- Frases curtas e diretas. Max 2 frases por campo.
- Foco sempre em reducao de gordura corporal SEM perder massa muscular e SEM prejudicar performance.
- Use numeros reais da medicao para embasar.
- Ajustes devem ser praticos e especificos (nao genericos).
- Correlacione carga de treino (TSS) com mudancas na composicao.

Responda SOMENTE em JSON valido:
{
  "tendencia": {
    "texto": "string — o que mudou de fato (peso, gordura, musculo)",
    "direcao": "melhorando" | "estavel" | "piorando"
  },
  "gordura_corporal": {
    "atual_pct": number,
    "meta_pct": number,
    "deficit_kg": number,
    "prazo_semanas": number
  },
  "correlacao_treino": "string — como a carga de treino esta impactando a composicao",
  "ajustes": [
    "string — ajuste 1 especifico",
    "string — ajuste 2 especifico",
    "string — ajuste 3 especifico"
  ],
  "nutricao": "string — 1 ajuste nutricional concreto para o proximo mes",
  "alerta": "string | null — so se houver algo preocupante",
  "proxima_medicao": "string — quando medir de novo e o que esperar"
}"""


def analyze_body_composition(
    measurements: list[dict],
    recent_workouts: list[dict],
    fitness: dict,
) -> dict:
    """Cross-reference body measurements with training data."""

    if not measurements:
        return {"error": "Sem medicoes registradas ainda."}

    latest = measurements[0]
    previous = measurements[1] if len(measurements) > 1 else None

    # Calcula deltas se tiver medição anterior
    deltas = {}
    if previous:
        for field in ["weight_kg", "body_fat_pct", "muscle_mass_kg", "water_pct"]:
            if latest.get(field) is not None and previous.get(field) is not None:
                deltas[field] = round(latest[field] - previous[field], 2)

    # Calcula TSS total das últimas semanas
    total_tss = sum(w.get("tss", 0) or 0 for w in recent_workouts)
    sports_count = {}
    for w in recent_workouts:
        sport = w.get("sport", "Outro")
        sports_count[sport] = sports_count.get(sport, 0) + 1

    # Prepara listas fora do f-string para evitar problema com {{}} dentro de expressões
    hist_summary = [
        {"date": m["date"], "weight_kg": m["weight_kg"], "body_fat_pct": m.get("body_fat_pct")}
        for m in measurements[:10]
    ]
    workout_details = [
        {
            "date": w.get("date"),
            "sport": w.get("sport"),
            "tss": w.get("tss"),
            "duration_h": round(w.get("duration_actual") or w.get("duration_planned") or 0, 1),
        }
        for w in recent_workouts[:15]
    ]

    msg = f"""MEDICOES REGISTRADAS ({len(measurements)} no total):

MAIS RECENTE ({latest['date']}):
- Peso: {latest.get('weight_kg')} kg
- Gordura corporal: {latest.get('body_fat_pct')} %
- Massa muscular: {latest.get('muscle_mass_kg')} kg
- Gordura visceral: {latest.get('visceral_fat')}
- Agua: {latest.get('water_pct')} %
- BMR: {latest.get('bmr_kcal')} kcal
- Observacao: {latest.get('notes') or 'nenhuma'}

{f"MEDICAO ANTERIOR ({previous['date']}): peso={previous.get('weight_kg')}kg, gordura={previous.get('body_fat_pct')}%" if previous else "PRIMEIRA MEDICAO — sem historico anterior"}

VARIACAO ENTRE MEDICOES: {json.dumps(deltas, ensure_ascii=False) if deltas else "N/A (primeira medicao)"}

HISTORICO COMPLETO (resumido):
{json.dumps(hist_summary, ensure_ascii=False, indent=2)}

TREINOS RECENTES (ultimas 4 semanas):
- Total de treinos: {len(recent_workouts)}
- TSS total: {round(total_tss)}
- Modalidades: {json.dumps(sports_count, ensure_ascii=False)}
- Detalhes: {json.dumps(workout_details, ensure_ascii=False)}

FORMA FISICA ATUAL (CTL/ATL/TSB):
{json.dumps(fitness, ensure_ascii=False)}

Analise a composicao corporal considerando o historico de treino. Foco em reducao de gordura corporal.
"""

    response = _get_client().messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": msg}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    return json.loads(text)
