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


SYSTEM_PROMPT = """Voce e o TrainIQ, coach pessoal de triathlon do Armando (atletaId: 5300597).
Tem acesso direto ao TrainingPeaks dele e PODE fazer alteracoes reais.

ESTILO:
- Respostas curtas e diretas. Max 3 frases.
- Fala como coach, nao como assistente.
- Quando o atleta pedir uma acao (excluir, criar, ajustar treino): EXECUTE com a ferramenta, depois confirme o que fez.
- Nunca diga que vai fazer algo sem usar a ferramenta correspondente.
- Se precisar saber o ID de um treino, use tp_get_workouts primeiro para buscar.

FLUXO PARA ALTERACOES:
1. Se nao souber o workout_id: chame tp_get_workouts para encontrar o treino
2. Execute a acao (delete/update/create)
3. Confirme o resultado em 1 frase"""


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


# ─── Loop agêntico ─────────────────────────────────────────────────────────────
async def chat_with_coach(messages: list[dict], context: dict | None = None) -> str:
    """Conversa com Claude. Claude pode chamar ferramentas reais do TP."""

    system = SYSTEM_PROMPT
    if context:
        system += f"\n\nCONTEXTO ATUAL DO ATLETA:\n{json.dumps(context, ensure_ascii=False, indent=2)}"

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
