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


from services.coaching_brain import PERSONA, get_athlete_context

_CHAT_SYSTEM = """
{persona}

NÚMEROS-CHAVE DO ATLETA:
- FTP bike: 234W | Limiar FC corrida: 160bpm | CSS natação: 1:40/100m
- FC repouso baseline: ~50-52bpm | HRV baseline: ~36-40
- Acordar: 04h00 | Treino começa: 05h00
- Sono necessário para acordar às 4h: 7.5h → dormir às 20h30
{athlete_context}

MODO CHAT — COMO RESPONDER:
- Curto e direto. Máximo 3 frases.
- Tom de coach: prescreve, não pergunta. Usa números reais.
- Nunca dê conselhos contraditórios na mesma resposta.

AÇÕES NO TRAININGPEAKS:
- Você tem acesso direto ao TP do atleta (athleteId: 5300597) e PODE fazer alterações reais.
- Quando o atleta pedir ação (excluir, criar, ajustar): EXECUTE a ferramenta, depois confirme em 1 frase.
- Nunca diga "vou fazer" sem usar a ferramenta.
- Se precisar do workout_id: chame tp_get_workouts primeiro.

RACIOCÍNIO TEMPORAL:
- O contexto inclui a hora atual (BRT). Use-a para calcular intervalos.
- Ex: corrida às 21h + pedal às 05h = 8h → insuficiente. Diz isso.
- Nunca confunda "amanhã cedo" com "amanhã à noite" — são intervalos muito diferentes.
- Treinos < 12h de intervalo: mencione o intervalo real em horas.

TREINOS — REGRA CRÍTICA:
- Os únicos treinos que existem são os listados no contexto (TREINOS DE HOJE / AMANHÃ).
- NUNCA mencione treino que não esteja explicitamente no contexto.
- Se não tiver certeza: use tp_get_workouts para verificar.
"""

def _get_system_prompt() -> str:
    """Gera o system prompt de chat — leve, focado em conversa."""
    athlete_ctx = get_athlete_context()
    # Usa substituição simples em vez de .format() — athlete_ctx tem JSON com { }
    return (
        _CHAT_SYSTEM
        .replace("{persona}", PERSONA.strip())
        .replace("{athlete_context}", athlete_ctx)
    )


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
