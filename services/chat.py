"""Chat service — TrainIQ coach com tool use real no TrainingPeaks."""

import os
import json
from datetime import datetime, timezone, timedelta
import anthropic

def _now_brt() -> str:
    """Hora atual em BRT (UTC-3) formatada."""
    brt = timezone(timedelta(hours=-3))
    return datetime.now(brt).strftime("%H:%M")

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
from services.database import get_active_alerts
from services.alert_extractor import format_alerts_for_context

_CHAT_SYSTEM = """{persona}

NÚMEROS-CHAVE DO ATLETA:
- FTP bike: 234W | Limiar FC corrida: 160bpm | CSS natação: 1:40/100m
- FC repouso baseline: ~50-52bpm | HRV baseline: ~36-40
- Acordar: 04h00 | Treino começa: 05h00
- Sono necessário para acordar às 4h: 7.5h → dormir às 20h30
{athlete_context}

MODO CHAT — COMO RESPONDER:
- Curto e direto. Máximo 3 frases por resposta.
- Tom de coach: prescreve, não pergunta. Usa números reais do contexto.
- Nunca dê conselhos contraditórios na mesma resposta.
- NUNCA peça ao atleta informações que já estão no contexto abaixo.

CONTEXTO DO ATLETA — FONTE ÚNICA DA VERDADE:
- O bloco CONTEXTO ATUAL abaixo contém TODOS os dados do dia: treinos, status, métricas, forma.
- Use esses dados diretamente. Não pergunte "quais treinos você tem" — você já sabe.
- Se um treino está marcado como ✓ concluído, trate como concluído. Se pendente, como pendente.

AÇÕES NO TRAININGPEAKS:
- Você tem acesso direto ao TP do atleta (athleteId: 5300597) e PODE fazer alterações reais.
- Quando o atleta pedir ação (excluir, criar, ajustar): EXECUTE a ferramenta, depois confirme em 1 frase.
- Nunca diga "vou fazer" sem usar a ferramenta.
- Para encontrar workout_id: use tp_get_workouts — mas só se o ID não estiver no contexto.

RACIOCÍNIO TEMPORAL:
- O contexto inclui a data e hora atuais (BRT). Use para calcular intervalos.
- Ex: corrida às 21h + pedal às 05h = 8h → insuficiente. Diz isso.
- Treinos < 12h de intervalo: mencione o intervalo real em horas.
"""

def _get_system_prompt(context_str: str = "") -> str:
    """Monta system prompt completo com persona + alertas + contexto atual."""
    athlete_ctx = get_athlete_context()
    hora = _now_brt()
    base = (
        _CHAT_SYSTEM
        .replace("{persona}", PERSONA.strip())
        .replace("{athlete_context}", athlete_ctx)
    )
    base += f"\n\nHORA ATUAL (BRT): {hora}"

    # Alertas ativos — sempre incluídos, têm prioridade máxima
    alerts = get_active_alerts()
    if alerts:
        alerts_str = format_alerts_for_context(alerts)
        base += f"\n\n════════════════════════════════════════\n{alerts_str}\n════════════════════════════════════════"

    if context_str:
        base += f"\n\n════════════════════════════════════════\nCONTEXTO ATUAL DO ATLETA (use estes dados — não pergunte ao atleta):\n════════════════════════════════════════\n{context_str}"
    return base


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

    # Treinos de hoje — lista explícita com status correto
    hoje = ctx.get("treinos_hoje", [])
    if hoje:
        lines.append("TREINOS DE HOJE (apenas estes — não invente outros):")
        for w in hoje:
            sport   = w.get("sport") or "?"
            title   = w.get("title") or "Treino"
            dur_min = int((w.get("duration_planned") or 0) * 60)
            tss     = w.get("tss_planned")
            tss_str = f" | TSS planejado {round(tss)}" if tss else ""
            # Usa campo computed 'completed' se existir, senão heurística
            done = (
                w.get("completed") is True
                or w.get("type") == "completed"
                or w.get("duration_actual") is not None
                or bool(w.get("tss_actual"))
                or bool(w.get("distance_actual"))
            )
            wid     = w.get("id", "")
            status  = "✓ CONCLUÍDO" if done else "⏳ pendente"
            id_str  = f" [id:{wid}]" if wid else ""
            lines.append(f"  • [{sport}] {title} — {dur_min}min{tss_str} — {status}{id_str}")
    else:
        lines.append("TREINOS DE HOJE: nenhum treino registrado no TP.")
    lines.append("")

    # Histórico da semana (dias anteriores)
    historico = ctx.get("historico_semana", [])
    if historico:
        lines.append("ATIVIDADES DOS ÚLTIMOS 7 DIAS:")
        for w in historico:
            sport   = w.get("sport") or "?"
            title   = w.get("title") or "Treino"
            data    = (w.get("date") or w.get("workout_day") or "")[:10]
            done    = w.get("completed", False)
            dur_min = int((w.get("duration_planned") or 0) * 60)
            dur_real = w.get("duration_actual")
            tss_real = w.get("tss_actual")
            status  = "✓" if done else "✗"
            detalhes = []
            if dur_real: detalhes.append(f"{int(dur_real/60)}min realizados")
            if tss_real: detalhes.append(f"TSS {round(tss_real)}")
            detalhe_str = f" ({', '.join(detalhes)})" if detalhes else f" ({dur_min}min planejado)"
            lines.append(f"  {status} [{data}] [{sport}] {title}{detalhe_str}")
        lines.append("")

    # Treinos de amanhã — lista explícita
    amanha = ctx.get("treinos_amanha", [])
    if amanha:
        lines.append("TREINOS DE AMANHÃ:")
        for w in amanha:
            sport   = w.get("sport") or "?"
            title   = w.get("title") or "Treino"
            dur_min = int((w.get("duration_planned") or 0) * 60)
            tss     = w.get("tss_planned")
            wid     = w.get("id", "")
            tss_str = f" | TSS {round(tss)}" if tss else ""
            id_str  = f" [id:{wid}]" if wid else ""
            lines.append(f"  • [{sport}] {title} — {dur_min}min{tss_str}{id_str}")
    else:
        lines.append("TREINOS DE AMANHÃ: nenhum registrado no TP.")
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
    """
    Coach sempre com contexto completo (Sonnet + ferramentas TP).
    O contexto do atleta é injetado em toda mensagem — o coach nunca precisa
    perguntar ao atleta informações que já estão disponíveis.
    """
    ctx_str = _format_context(context) if context else ""
    system  = _get_system_prompt(ctx_str)
    model   = "claude-sonnet-4-5"
    tools   = TOOLS

    client = _get_client()
    current_messages = list(messages[-20:])

    for _ in range(8):  # max 8 rodadas de tool calls
        kwargs: dict = dict(
            model=model,
            max_tokens=600,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=current_messages,
        )
        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)

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
