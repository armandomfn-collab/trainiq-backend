"""Chat service — TrainIQ coach com tool use real no TrainingPeaks."""

import os
import json
import anthropic

from tp_mcp.tools.workouts import (
    tp_get_workouts,
    tp_delete_workout,
    tp_update_workout,
    tp_create_workout,
)
from tp_mcp.tools.fitness import tp_get_fitness


def _get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


from services.coaching_brain import build_system_prompt

_CHAT_ROLE = """Você está no modo CHAT — conversa direta com o atleta.

REGRAS DO CHAT:
- Respostas curtas e diretas. Máximo 3 frases por resposta.
- Quando o atleta pedir uma ação (excluir, criar, ajustar treino): EXECUTE com a ferramenta, depois confirme em 1 frase.
- Nunca diga "vou fazer" sem usar a ferramenta correspondente.
- Se precisar de workout_id: chame tp_get_workouts primeiro.
- Você tem acesso direto ao TrainingPeaks do atleta (athleteId: 5300597) e PODE fazer alterações reais.

FLUXO PARA ALTERAÇÕES NO TP:
1. Se não souber o workout_id → tp_get_workouts para encontrar
2. Execute a ação (delete/update/create)
3. Confirme em 1 frase com o resultado

RACIOCÍNIO TEMPORAL — REGRAS OBRIGATÓRIAS:
- O contexto sempre inclui "hora_atual" (ex: "21:30"). Use isso.
- Para avaliar intervalos entre treinos, calcule as horas EXPLICITAMENTE.
  Ex: corrida às 21h + pedal às 05h = apenas 8h de intervalo → insuficiente para recuperação.
  Ex: corrida às 21h + pedal às 21h do dia seguinte = 24h → ok.
- "Amanhã de manhã" significa menor intervalo do que "amanhã à noite".
  Nunca confunda os dois. Calcule sempre.
- Treinos próximos (< 12h): sempre mencione o intervalo real em horas.
- Se o atleta pergunta se consegue fazer treino X e depois Y, responda com:
  "X às HH:MM + Y às HH:MM = N horas de intervalo — [adequado/insuficiente] porque..."

REGRA CRÍTICA — TREINOS:
- Os únicos treinos que existem são os listados em "TREINOS DE HOJE" e "TREINOS DE AMANHÃ" no contexto.
- Se o contexto diz "TREINOS DE HOJE: [Run]", hoje SÓ TEM corrida. Não existe pedal hoje.
- NUNCA mencione um treino que não esteja explicitamente listado no contexto.
- Se não tiver certeza do que está agendado, use a ferramenta tp_get_workouts para verificar.

QUANDO O ATLETA PEDIR ANÁLISE OU CONSELHO:
- Usa os dados do contexto (TSB, HRV, treinos, hora_atual) + metodologia embutida
- Responde como coach, não como assistente: prescreve, não pergunta
- Nunca dê duas recomendações contraditórias na mesma resposta"""

def _get_system_prompt() -> str:
    """Gera o system prompt completo em runtime (inclui perfil atualizado do DB)."""
    return build_system_prompt(_CHAT_ROLE)


# ─── Ferramentas disponíveis ──────────────────────────────────────────────────
TOOLS = [
    {
        "name": "tp_get_workouts",
        "description": "Busca treinos do TrainingPeaks para um intervalo de datas. Use para encontrar o workout_id antes de alterar ou excluir.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Data inicial YYYY-MM-DD"},
                "end_date":   {"type": "string", "description": "Data final YYYY-MM-DD"},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "tp_delete_workout",
        "description": "Exclui um treino do TrainingPeaks pelo ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "workout_id": {"type": "string", "description": "ID do treino a excluir"},
            },
            "required": ["workout_id"],
        },
    },
    {
        "name": "tp_update_workout",
        "description": "Atualiza campos de um treino existente (título, duração, TSS, data, descrição).",
        "input_schema": {
            "type": "object",
            "properties": {
                "workout_id":       {"type": "string", "description": "ID do treino"},
                "title":            {"type": "string", "description": "Novo título"},
                "duration_minutes": {"type": "number", "description": "Nova duração em minutos"},
                "tss_planned":      {"type": "number", "description": "Novo TSS planejado"},
                "description":      {"type": "string", "description": "Nova descrição"},
                "date":             {"type": "string", "description": "Nova data YYYY-MM-DD"},
            },
            "required": ["workout_id"],
        },
    },
    {
        "name": "tp_create_workout",
        "description": "Cria um novo treino no TrainingPeaks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_str":         {"type": "string",  "description": "Data YYYY-MM-DD"},
                "sport":            {"type": "string",  "description": "Swim | Bike | Run | Strength | Walk | Brick"},
                "title":            {"type": "string",  "description": "Título do treino"},
                "duration_minutes": {"type": "integer", "description": "Duração em minutos"},
                "tss_planned":      {"type": "number",  "description": "TSS planejado"},
                "description":      {"type": "string",  "description": "Descrição"},
            },
            "required": ["date_str", "sport", "title"],
        },
    },
    {
        "name": "tp_get_fitness",
        "description": "Retorna CTL, ATL, TSB e forma fisica atual do atleta.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ─── Executor de ferramentas ───────────────────────────────────────────────────
async def _run_tool(name: str, inputs: dict) -> dict:
    try:
        if name == "tp_get_workouts":
            return await tp_get_workouts(inputs["start_date"], inputs["end_date"])
        elif name == "tp_delete_workout":
            return await tp_delete_workout(inputs["workout_id"])
        elif name == "tp_update_workout":
            return await tp_update_workout(**inputs)
        elif name == "tp_create_workout":
            return await tp_create_workout(**inputs)
        elif name == "tp_get_fitness":
            return await tp_get_fitness()
        else:
            return {"error": f"Ferramenta desconhecida: {name}"}
    except Exception as e:
        return {"error": str(e)}


# ─── Formatador de contexto ────────────────────────────────────────────────────
def _format_context(ctx: dict) -> str:
    """Formata o contexto do atleta de forma clara e inequívoca para o coach."""
    lines = []

    data  = ctx.get("data_hoje", "?")
    hora  = ctx.get("hora_atual", "?")
    forma = ctx.get("forma", {})
    metr  = ctx.get("metricas", {})

    lines.append(f"DATA/HORA ATUAL: {data} às {hora}")
    lines.append("")

    # Treinos de hoje — lista explícita
    hoje = ctx.get("treinos_hoje", [])
    if hoje:
        lines.append("TREINOS DE HOJE (apenas estes — não invente outros):")
        for w in hoje:
            sport   = w.get("sport") or "?"
            title   = w.get("title") or "Treino"
            dur_min = int((w.get("duration_planned") or 0) * 60)
            tss     = w.get("tss_planned")
            done    = bool(w.get("duration_actual") or w.get("tss_actual"))
            status  = "✓ concluído" if done else "pendente"
            tss_str = f" | TSS {tss}" if tss else ""
            lines.append(f"  • [{sport}] {title} — {dur_min}min{tss_str} ({status})")
    else:
        lines.append("TREINOS DE HOJE: nenhum treino registrado no TP.")
    lines.append("")

    # Treinos de amanhã — lista explícita
    amanha = ctx.get("treinos_amanha", [])
    if amanha:
        lines.append("TREINOS DE AMANHÃ (apenas estes — não invente outros):")
        for w in amanha:
            sport   = w.get("sport") or "?"
            title   = w.get("title") or "Treino"
            dur_min = int((w.get("duration_planned") or 0) * 60)
            tss     = w.get("tss_planned")
            tss_str = f" | TSS {tss}" if tss else ""
            lines.append(f"  • [{sport}] {title} — {dur_min}min{tss_str}")
    else:
        lines.append("TREINOS DE AMANHÃ: nenhum treino registrado no TP.")
    lines.append("")

    # Forma física
    if forma:
        ctl = forma.get("ctl") or forma.get("CTL", "?")
        atl = forma.get("atl") or forma.get("ATL", "?")
        tsb = forma.get("tsb") or forma.get("TSB", "?")
        lines.append(f"FORMA: CTL {ctl} | ATL {atl} | TSB {tsb}")

    # Métricas de saúde
    if metr:
        hrv    = metr.get("HRV") or metr.get("hrv") or metr.get("HRV Status")
        bb     = metr.get("Body Battery") or metr.get("body_battery")
        hr_rep = metr.get("Resting Heart Rate") or metr.get("resting_hr")
        sono   = metr.get("Sleep") or metr.get("sleep_hours")
        partes = []
        if hrv:    partes.append(f"HRV {hrv}")
        if bb:     partes.append(f"Body Battery {bb}")
        if hr_rep: partes.append(f"FC repouso {hr_rep}")
        if sono:   partes.append(f"Sono {sono}h")
        if partes:
            lines.append(f"SAÚDE: {' | '.join(partes)}")

    return "\n".join(lines)


# ─── Loop agêntico ─────────────────────────────────────────────────────────────
async def chat_with_coach(messages: list[dict], context: dict | None = None) -> str:
    """Conversa com Claude. Claude pode chamar ferramentas reais do TP."""

    system = _get_system_prompt()
    if context:
        ctx_str = _format_context(context)
        system += f"\n\n════════════════════════════════════════\nCONTEXTO ATUAL DO ATLETA:\n════════════════════════════════════════\n{ctx_str}"

    client = _get_client()
    current_messages = list(messages)

    for _ in range(8):  # max 8 rodadas de tool calls
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=800,
            system=system,
            tools=TOOLS,
            messages=current_messages,
        )

        # Resposta final — sem tool calls
        if response.stop_reason == "end_turn":
            texts = [b.text for b in response.content if hasattr(b, "text")]
            return texts[0].strip() if texts else "Feito."

        # Tem tool calls — executa e continua
        if response.stop_reason == "tool_use":
            # Monta conteúdo do assistant (text + tool_use blocks)
            assistant_content = []
            tool_blocks = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id":    block.id,
                        "name":  block.name,
                        "input": block.input,
                    })
                    tool_blocks.append(block)

            current_messages.append({"role": "assistant", "content": assistant_content})

            # Executa todas as ferramentas em paralelo
            import asyncio
            results = await asyncio.gather(*[_run_tool(b.name, b.input) for b in tool_blocks])

            tool_results = [
                {
                    "type":        "tool_result",
                    "tool_use_id": tool_blocks[i].id,
                    "content":     json.dumps(results[i], ensure_ascii=False),
                }
                for i in range(len(tool_blocks))
            ]

            current_messages.append({"role": "user", "content": tool_results})

        else:
            break

    return "Operacao concluida."
