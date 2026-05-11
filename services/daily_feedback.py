"""AI feedback for daily workout completion — professional coach level."""

import os
import json
import anthropic


def _get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


from services.coaching_brain import build_system_prompt

_FEEDBACK_ROLE = """Você está gerando o FEEDBACK DIÁRIO — análise técnica do que aconteceu no dia.

FILOSOFIA:
- Usa os numeros reais: TSS realizado vs planejado, duracao real vs planejada, zonas de FC, pace
- Se um treino foi cortado, identifica o motivo pelos dados (FC alta? TSS baixo? duracao curta?)
- Pensa no sistema completo do dia: como cada treino afeta o proximo
- Compensacao inteligente: se bike foi curto, vale natacao? Depende do TSB, do horario, do treino de amanha
- Amanha nao e isolado — e consequencia direta do que foi feito hoje

REGRAS DE ESCRITA:
- MAX 2 frases por campo. Direto. Sem elogios genericos.
- Use numeros quando tiver (ex: "TSS 47/80 — 59% do planejado")
- `o_que_faltou` e `compensacao`: null se nao se aplica

Responda SOMENTE em JSON valido:
{
  "resumo_dia": "string — 1 frase tecnica, cita numeros",
  "avaliacoes": [
    {
      "workout_id": "string",
      "titulo": "string",
      "status": "concluido" | "pendente" | "parcial",
      "nota": <number 0-10 ou null se pendente>,
      "o_que_foi_bem": "string | null — 1 frase especifica com dado real, null se nao concluido",
      "o_que_faltou": "string | null — o gap especifico (volume, intensidade, zona, etc.), null se perfeito ou pendente",
      "causa_provavel": "string | null — motivo inferido pelos dados (fadiga, dor, tempo, FC elevada...)",
      "compensacao": "string | null — sugere ajuste/treino extra SOMENTE se faz sentido energeticamente e ha tempo",
      "impacto_amanha": "string — como ESSE treino especificamente afeta o treino de amanha"
    }
  ],
  "carga_total": {
    "tss_planejado": <number>,
    "tss_realizado": <number>,
    "percentual": <number>
  },
  "estrategia_restante_dia": "string — o que fazer com o resto do dia considerando tudo (null se dia encerrado)",
  "oportunidade": "string | null — se ha treino pendente viavel, sugere modalidade + horario especifico + duracao",
  "recuperacao": "string — 1 acao concreta agora (nutricao, sono, gelo...)",
  "amanha": {
    "como_hoje_impacta": "string — consequencia direta de hoje no treino de amanha",
    "foco_principal": "string — o que priorizar amanha dado o que aconteceu hoje",
    "intensidade": "leve" | "moderada" | "alta" | "descanso",
    "melhor_horario": "string (ex: 06h-08h)",
    "preparacao_hoje": "string — 1 acao concreta hoje a noite para amanha"
  }
}"""

FEEDBACK_PROMPT = build_system_prompt(_FEEDBACK_ROLE)


def generate_daily_feedback(
    workout_status: list,
    all_done: bool,
    tomorrow_workouts: list,
    metrics: dict,
    fitness: dict,
) -> dict:
    """Generate professional coach-level feedback for the day's workouts."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    hora_atual = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%H:%M")

    completed = [w for w in workout_status if w.get("completed")]
    pending   = [w for w in workout_status if not w.get("completed")]

    # Monta dados detalhados por treino
    workout_details = []
    for w in workout_status:
        detail = {
            "id":               w.get("id"),
            "titulo":           w.get("title"),
            "sport":            w.get("sport"),
            "concluido":        w.get("completed"),
            "duracao_plan_h":   round(w.get("duration_planned") or 0, 2),
            "duracao_real_h":   round(w.get("duration_actual") or 0, 2),
            "tss_planejado":    w.get("tss_planned"),
            "tss_realizado":    w.get("tss_actual"),
            "distancia_plan":   w.get("distance_planned"),
            "distancia_real":   w.get("distance_actual"),
            "calorias":         w.get("calories_actual"),
            "fc_media":         w.get("heart_rate_avg"),
            "descricao":        w.get("description", ""),
            "notas_atleta":     w.get("athlete_comment", ""),
        }
        workout_details.append(detail)

    tomorrow_details = [
        {
            "titulo":       w.get("title"),
            "sport":        w.get("sport"),
            "tss_plan":     w.get("tss_planned"),
            "duracao_plan": round(w.get("duration_planned") or 0, 2),
        }
        for w in tomorrow_workouts
    ]

    msg = f"""Hora atual: {hora_atual}

RESUMO DO DIA:
- Concluidos: {len(completed)} treinos
- Pendentes: {len(pending)} treinos
- Hora atual: {hora_atual} (avalia viabilidade dos pendentes)

DETALHES DOS TREINOS DE HOJE:
{json.dumps(workout_details, ensure_ascii=False, indent=2)}

TREINOS DE AMANHA:
{json.dumps(tomorrow_details, ensure_ascii=False, indent=2)}

FORMA ATUAL (CTL/ATL/TSB):
{json.dumps(fitness, ensure_ascii=False)}

METRICAS RECENTES:
{json.dumps(metrics, ensure_ascii=False)}

Analise cada treino com olhar tecnico. Compara real vs planejado. Identifica o que faltou e por que. Decide se compensacao faz sentido dado o TSB atual e o treino de amanha."""

    response = _get_client().messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=FEEDBACK_PROMPT,
        messages=[{"role": "user", "content": msg}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1]).strip()
    return json.loads(text)
