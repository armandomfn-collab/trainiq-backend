"""Claude AI analysis service for TrainIQ."""

import os
import json
from datetime import date
import anthropic

def _get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

from services.coaching_brain import build_system_prompt

_ANALYSIS_ROLE = """Você está gerando a ANÁLISE DIÁRIA — o painel de inteligência do atleta.

Sempre responda em JSON válido com exatamente esta estrutura:
{
  "status_geral": "string — resumo em 1 frase do estado do atleta hoje",
  "nivel_prontidao": "alto" | "moderado" | "baixo",
  "insights": ["string", "string", "string"],
  "ajustes_treino": [
    {
      "workout_id": "string",
      "titulo": "string",
      "ajuste": "manter" | "reduzir" | "aumentar" | "cancelar",
      "intensidade_sugerida_pct": number,
      "motivo": "string",
      "descricao_ajuste": "string"
    }
  ],
  "nutricao": {
    "pre_treino": "string",
    "durante_treino": "string",
    "pos_treino": "string",
    "carboidratos_g": number,
    "proteina_g": number,
    "hidratacao": "string"
  },
  "horario_ideal": {
    "janela": "string (ex: 15h-17h)",
    "motivo": "string"
  },
  "sono": {
    "qualidade_hoje": "boa" | "regular" | "ruim",
    "meta_hoje": number,
    "dicas": ["string", "string"]
  }
}"""

SYSTEM_PROMPT = build_system_prompt(_ANALYSIS_ROLE)


def analyze_athlete_data(
    today: str,
    metrics: dict,
    workouts: list,
    fitness: dict,
) -> dict:
    """Run Claude analysis on athlete data and return structured recommendations."""

    user_message = f"""
Analise os dados do atleta para hoje ({today}) e gere recomendações:

## MÉTRICAS DE HOJE
{json.dumps(metrics, ensure_ascii=False, indent=2)}

## TREINOS PLANEJADOS HOJE
{json.dumps(workouts, ensure_ascii=False, indent=2)}

## FORMA FÍSICA (CTL/ATL/TSB)
{json.dumps(fitness, ensure_ascii=False, indent=2)}

Gere recomendações específicas e práticas considerando:
1. O estado de recuperação (HRV, sono, Body Battery)
2. A fadiga acumulada (TSB/ATL/CTL)
3. Os treinos planejados para hoje
4. Otimização de performance e recuperação
5. Estratégia nutricional específica para o dia
"""

    response = _get_client().messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    return json.loads(text)
