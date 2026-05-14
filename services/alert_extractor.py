"""
Alert Extractor — analisa cada conversa do chat e extrai eventos significativos.
Roda em background (não bloqueia a resposta do coach).

Detecta:
- Dores e lesões (romboide, joelho, tornozelo, etc.)
- Fadiga incomum, doença
- Eventos pessoais que afetam treino (viagem, stress, trabalho intenso)
- Resolução de alertas anteriores ("a dor passou", "estou bem")
"""

import json
import os
import anthropic
from services.database import upsert_alert, resolve_alert_by_key, get_active_alerts

_EXTRACT_PROMPT = """Analise a última mensagem do atleta abaixo.

Extraia SOMENTE eventos SIGNIFICATIVOS para o treinamento:
- Dor física, lesão, desconforto (qualquer parte do corpo)
- Doença, febre, indisposição
- Fadiga incomum, insônia, problema de recuperação
- Evento pessoal que afeta treino (viagem, stress extremo, deadline)
- RESOLUÇÃO de problema anterior ("a dor passou", "tô bem", "melhorou")

Retorne JSON no formato abaixo. Se não há nada significativo, retorne {"events": []}.

{
  "events": [
    {
      "action": "create" | "resolve",
      "category": "pain" | "injury" | "illness" | "fatigue" | "personal",
      "body_part": "romboide" | "joelho" | null,
      "description": "descrição curta e objetiva em português",
      "severity": "mild" | "moderate" | "severe"
    }
  ]
}

REGRAS:
- Só extraia se o atleta MENCIONOU explicitamente. Não infira.
- Dores vagas ("cansado") → category: fatigue, body_part: null
- Dores localizadas → body_part: nome do músculo/articulação
- "Resolução": quando atleta diz que algo melhorou ou passou
- Máximo 3 eventos por mensagem
- Se nada relevante: {"events": []}

MENSAGEM DO ATLETA:
"""


async def extract_and_update_alerts(user_message: str) -> list[dict]:
    """
    Analisa a mensagem do usuário com Haiku (rápido e barato).
    Atualiza o banco de alertas silenciosamente.
    Retorna lista de eventos processados (para log).
    """
    if len(user_message.strip()) < 5:
        return []

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": _EXTRACT_PROMPT + user_message
            }]
        )

        raw = response.content[0].text.strip()

        # Extrai JSON mesmo se vier com markdown
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        events = data.get("events", [])
        processed = []

        for ev in events[:3]:
            action      = ev.get("action", "create")
            category    = ev.get("category", "pain")
            body_part   = ev.get("body_part") or ""
            description = ev.get("description", "")
            severity    = ev.get("severity", "moderate")

            if not description:
                continue

            if action == "resolve":
                resolve_alert_by_key(category, body_part)
                processed.append({"action": "resolved", "category": category, "body_part": body_part})
            else:
                alert_id = upsert_alert(category, body_part, description, severity)
                processed.append({"action": "created", "id": alert_id, "category": category, "body_part": body_part})

        return processed

    except Exception as e:
        print(f"[alert_extractor] Erro ao extrair alertas: {e}")
        return []


def format_alerts_for_context(alerts: list[dict]) -> str:
    """Formata os alertas ativos para injeção no system prompt do coach."""
    if not alerts:
        return ""

    lines = ["ALERTAS ATIVOS DO ATLETA (SEMPRE pergunte sobre estes antes de recomendar):"]
    for a in alerts:
        category  = a.get("category", "")
        body_part = a.get("body_part") or ""
        desc      = a.get("description", "")
        severity  = a.get("severity", "moderate")
        updated   = (a.get("last_updated") or "")[:10]
        last_asked = a.get("last_asked")

        emoji = {"pain": "🔴", "injury": "🚨", "illness": "🤒", "fatigue": "😴", "personal": "⚠️"}.get(category, "⚠️")
        sev_str = {"mild": "leve", "moderate": "moderado", "severe": "severo"}.get(severity, severity)
        asked_str = f" | última pergunta: {last_asked[:10]}" if last_asked else " | ainda não perguntou"
        part_str = f" ({body_part})" if body_part else ""

        lines.append(f"  {emoji} [{category.upper()}{part_str}] {desc} — {sev_str} | desde {updated}{asked_str}")

    lines.append("")
    lines.append("INSTRUÇÃO: Ao responder, PRIMEIRO reconheça o alerta mais recente e peça um status.")
    lines.append("Ex: 'Antes de responder — e a dor no romboide? Melhorou, piorou ou igual?'")
    lines.append("Só depois dê a recomendação principal.")

    return "\n".join(lines)
