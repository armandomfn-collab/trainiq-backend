"""AI-powered post-workout review service."""

import json
import os
import anthropic

def _get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

from services.coaching_brain import build_system_prompt

_REVIEW_ROLE = """Você está gerando o REVIEW PÓS-TREINO. Analise o treino concluído com rigor técnico.

Responda em JSON com esta estrutura:
{
  "titulo_notificacao": "string curta (max 50 chars) para título do push",
  "resumo_notificacao": "string curta (max 100 chars) para o corpo do push",
  "eficiencia_pct": number (0-100),
  "cumprimento_plano": "excelente" | "bom" | "parcial" | "abaixo",
  "tss_planejado": number,
  "tss_realizado": number,
  "duracao_planejada_min": number,
  "duracao_realizada_min": number,
  "pontos_positivos": ["string", "string"],
  "pontos_atencao": ["string", "string"],
  "recomendacao_recuperacao": "string — o que fazer nas próximas horas",
  "impacto_forma": "string — como isso afeta CTL/ATL/TSB"
}"""

REVIEW_PROMPT = build_system_prompt(_REVIEW_ROLE)


def generate_workout_review(workout: dict, metrics: dict, fitness: dict) -> dict:
    """Generate AI review for a completed workout."""

    user_message = f"""
Analise este treino concluído:

## TREINO
{json.dumps(workout, ensure_ascii=False, indent=2)}

## MÉTRICAS DO ATLETA HOJE
{json.dumps(metrics, ensure_ascii=False, indent=2)}

## FORMA FÍSICA ATUAL
{json.dumps(fitness, ensure_ascii=False, indent=2)}

Avalie:
1. Cumprimento do plano (TSS real vs planejado, duração real vs planejada)
2. Eficiência do treino dado o estado de recuperação
3. Impacto na forma física
4. Recomendações de recuperação
"""

    response = _get_client().messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        system=REVIEW_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    return json.loads(text)
