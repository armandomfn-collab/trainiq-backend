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
    tp_get_workout,
    tp_get_workout_comments,
    tp_delete_workout,
    tp_update_workout,
    tp_create_workout,
)
from tp_mcp.tools.fitness import tp_get_fitness
from tp_mcp.tools.analyze import tp_analyze_workout
from tp_mcp.tools.peaks import tp_get_peaks, tp_get_workout_prs
from tp_mcp.tools.workout_types import tp_get_workout_types
from tp_mcp.tools.atp import tp_get_atp
from tp_mcp.tools.weekly_summary import tp_get_weekly_summary
from tp_mcp.tools.events import tp_get_focus_event, tp_get_next_event
from tp_mcp.tools.settings import tp_get_athlete_settings


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

═══════════════════════════════════════════════════
DOIS MODOS DE RESPOSTA — escolha automaticamente
═══════════════════════════════════════════════════

▶ MODO RÁPIDO (default — conversas, ajustes, perguntas pontuais):
  - Curto e direto. 2-4 frases.
  - Coach prescreve, não pergunta. Usa números do contexto.
  - Nunca peça dados que já estão no contexto.

▶ MODO AVALIAÇÃO (quando o atleta pede "avalia execução", "como foi o treino",
  "review da sessão", "analisa esse treino", ou pergunta sobre desempenho recente):

  ESTRUTURA OBRIGATÓRIA — siga em ordem, cada bloco com cabeçalho em negrito:

  **Plano vs Real** (números puros, sem floreio)
  Compare o que estava prescrito com o que foi feito. TSS planejado vs realizado,
  duração, distância. Se houver blocos (intervalos), compare bloco a bloco:
  potência/pace/FC alvo vs média real. Indique % de cumprimento ou diferença.
  Ex: "Plano: 5x4min Z4 a 220W. Real: completou os 5, potência média 208W (-5%).
  FC média 162bpm — 2bpm acima do limiar de 160, esperado em Z4."

  **Análise técnica** (o que os números contam)
  - Pacing: caiu nos últimos tiros? Negative split ou degradação?
  - FC drift: subiu durante a sessão? Quanto?
  - Eficiência: relação potência/FC ou pace/FC dentro do esperado?
  - Aderência ao alvo: ficou na zona certa ou escorregou pra Z3 cinza?
  - Sinais de fadiga ou frescor: comparação com sessões semelhantes recentes.

  **Contexto da semana** (zoom out — o coach vê o quadro maior)
  Conecte essa sessão com CTL/ATL/TSB, com os outros treinos da semana,
  com o ciclo (base/build/peak). 1 linha objetiva.
  Ex: "Terceiro Z4 da semana, ATL em 65 — você está absorvendo a carga."

  **Insight de coach** (o algo além — voz humana, não robô)
  Aqui você sai do número e fala como coach que conhece o atleta:
  - O que essa execução te diz sobre o estado atual dele
  - O que ajustar na próxima sessão semelhante (prescrição concreta)
  - Onde aplaudir e onde puxar a orelha — sem rodeios
  - Se for o caso, conecta com a meta de prova (70.3)
  Ex: "Você suporta 220W estável agora — próximo bike Z4 testa 225W nos
  primeiros 3 tiros. Mas o FC drift sugere que poderia ter feito 4 reps e
  mantido qualidade ao invés de cair no quinto."

  Tamanho do modo avaliação: 12-20 linhas no total. Não corte.

═══════════════════════════════════════════════════
TOM DE COACH (vale para os dois modos)
═══════════════════════════════════════════════════
- NUNCA: "bom treino!", "parabéns!", "continue assim!", "manda ver!"
- SEMPRE: observação concreta + diagnóstico + prescrição
- Coach mistura técnica com humano. Não é planilha falando, é alguém que
  treina pessoas e usa os números pra contar uma história.
- Quando elogia, ancora num número: "potência sólida — 5W acima da última
  sessão semelhante há 12 dias". Quando puxa orelha, idem.
- Conecta o presente com o futuro: "isso fecha seu Build 1. Quarta tem
  teste de FTP — descanse a noite e mantém leve amanhã."

CONTEXTO DO ATLETA — FONTE ÚNICA DA VERDADE:
- O bloco CONTEXTO ATUAL abaixo contém TODOS os dados do dia.
- O contexto traz APENAS resumo: TSS, duração, distância. NÃO traz potência média,
  FC média, NP, IF, dados por intervalo.

WORKFLOW DE AVALIAÇÃO TÉCNICA — OBRIGATÓRIO seguir:
1. Pega o id do treino concluído no contexto (já está lá em [id:...]).
2. Chama tp_get_workout(id) → traz avg_power, normalized_power, avg_hr,
   if_actual, elevation_gain. Esses são os dados REAIS para o "Plano vs Real".
3. Se a pergunta envolve bloco/intervalo específico (ex: "potência do bloco
   principal", "como foi cada tiro"), chama tp_analyze_workout(id) → traz
   laps com média por bloco.
4. NUNCA diga "não consegui carregar a potência" sem antes ter chamado
   tp_get_workout(id). O contexto resumido não tem essa info — você precisa
   buscar via tool. Se a tool retornar avg_power null aí sim diga que falta dado.
5. Se o id não estiver no contexto (ex: treino de outra data), busca primeiro
   via tp_get_workouts(start,end) e depois tp_get_workout(id).

- Se um treino está marcado como ✓ concluído, trate como concluído.

AÇÕES NO TRAININGPEAKS:
- Você tem acesso direto ao TP do atleta (athleteId: 5300597) e PODE fazer alterações reais.
- Quando o atleta pedir ação (excluir, criar, ajustar): EXECUTE a ferramenta, depois confirme em 1 frase.
- Nunca diga "vou fazer" sem usar a ferramenta.
- Para encontrar workout_id: use tp_get_workouts — mas só se o ID não estiver no contexto.

CRIANDO TREINOS ESTRUTURADOS (blocos/intervalos):
- Sempre que o atleta pedir treino com blocos, intervalos, aquecimento+tiros+volta à calma: use o campo 'structure' no tp_create_workout.
- NÃO coloque os blocos apenas na description — isso não cria estrutura real no TP.
- Bike/Zwift → primaryIntensityMetric: "percentOfFtp" | Corrida → "percentOfThresholdPace"
- FTP do atleta: 234W. Limiar corrida: 160bpm / pace ~4:40/km.
- Zonas de bike (% FTP): Z1<55, Z2=56-75, Z3=76-90, Z4=91-105, Z5>105
- Zonas de corrida (% pace limiar): Z1<75, Z2=76-85, Z3=86-95, Z4=96-105, Z5>105
- intensityClass: "warmUp" (aquecimento), "active" (esforço), "rest" (recuperação), "coolDown" (volta à calma)
- Para repetições: use type="repetition" com reps e steps internos.
- CRÍTICO: crie APENAS os blocos que o atleta pediu explicitamente. Não adicione aquecimento, volta à calma ou recuperação que não foram solicitados.
- Se pediu "6x3min Z4": crie só o bloco de repetição. Sem aquecimento, sem cooldown.
- Se pediu "aquecimento + 6x3min + volta à calma": crie os três.
- Exemplo de apenas 6x3min Z4 bike (sem nada a mais):
  steps: [
    {type:"repetition", name:"6x3min Z4", reps:6, steps:[
      {name:"Tiro Z4", duration_seconds:180, intensity_min:91, intensity_max:105, intensityClass:"active"},
      {name:"Recuperação", duration_seconds:120, intensity_min:50, intensity_max:60, intensityClass:"rest"}
    ]}
  ]

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
        "description": "Atualiza campos de um treino existente (título, duração, TSS, data, descrição, estrutura de blocos).",
        "input_schema": {
            "type": "object",
            "properties": {
                "workout_id":       {"type": "string", "description": "ID do treino"},
                "title":            {"type": "string", "description": "Novo título"},
                "duration_minutes": {"type": "number", "description": "Nova duração em minutos"},
                "tss_planned":      {"type": "number", "description": "Novo TSS planejado"},
                "description":      {"type": "string", "description": "Nova descrição"},
                "date":             {"type": "string", "description": "Nova data YYYY-MM-DD"},
                "structure": {
                    "type": "object",
                    "description": "Nova estrutura de blocos. Mesmo formato do tp_create_workout.",
                    "properties": {
                        "primaryIntensityMetric": {"type": "string"},
                        "steps": {"type": "array", "items": {"type": "object"}}
                    }
                },
            },
            "required": ["workout_id"],
        },
    },
    {
        "name": "tp_create_workout",
        "description": (
            "Cria um novo treino no TrainingPeaks. "
            "Para treinos estruturados com blocos (intervalos, aquecimento, tiros), use o campo 'structure'. "
            "structure.primaryIntensityMetric: 'percentOfFtp' (bike/zwift) ou 'percentOfThresholdPace' (corrida). "
            "structure.steps: lista de steps ou repetition blocks. "
            "Step: {name, duration_seconds, intensity_min, intensity_max, intensityClass}. "
            "intensityClass: 'warmUp' | 'active' | 'rest' | 'coolDown'. "
            "Repetition block: {type:'repetition', name, reps, steps:[...]}. "
            "intensity_min/max = % do FTP (bike) ou % do pace limiar (corrida). "
            "Exemplo bike Z4: intensity_min=91, intensity_max=105. "
            "Exemplo corrida Z3: intensity_min=88, intensity_max=95. "
            "SEMPRE use structure para treinos com blocos — não coloque os blocos apenas na description."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date_str":         {"type": "string",  "description": "Data YYYY-MM-DD"},
                "sport":            {"type": "string",  "description": "Swim | Bike | Run | Strength | Walk | Brick"},
                "title":            {"type": "string",  "description": "Título do treino"},
                "duration_minutes": {"type": "integer", "description": "Duração em minutos (calculado automaticamente se structure fornecido)"},
                "tss_planned":      {"type": "number",  "description": "TSS planejado (calculado automaticamente se structure fornecido)"},
                "description":      {"type": "string",  "description": "Descrição textual do treino"},
                "structure": {
                    "type": "object",
                    "description": "Estrutura de blocos do treino. Use para criar treinos com intervalos reais no TP.",
                    "properties": {
                        "primaryIntensityMetric": {
                            "type": "string",
                            "description": "'percentOfFtp' para bike/zwift, 'percentOfThresholdPace' para corrida"
                        },
                        "steps": {
                            "type": "array",
                            "description": "Lista de steps e repetition blocks",
                            "items": {"type": "object"}
                        }
                    }
                },
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
    {
        "name": "tp_get_workout",
        "description": (
            "Busca DETALHES COMPLETOS de um treino específico — incluindo dados realizados de execução: "
            "avg_power, normalized_power, avg_hr, avg_cadence, if_actual, calories, elevation_gain. "
            "USE SEMPRE para avaliar execução de treino concluído — tp_get_workouts (lista) não retorna esses campos. "
            "Workflow para avaliação: 1) pega id do contexto ou via tp_get_workouts, 2) chama tp_get_workout(id) pra ver potência/FC reais."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workout_id": {"type": "string", "description": "ID do treino"},
            },
            "required": ["workout_id"],
        },
    },
    {
        "name": "tp_analyze_workout",
        "description": (
            "Análise PROFUNDA de um treino: totais, canais de dados (power/HR/pace por tempo), "
            "LAPS (cada bloco/intervalo com duração e média de potência/FC), e tempo em zonas. "
            "Use quando precisar avaliar bloco a bloco — ex: 'qual a potência média do bloco principal', "
            "'como foi a degradação dos tiros', 'FC drift entre primeiro e último intervalo'. "
            "Mais caro que tp_get_workout — use só quando precisar do detalhe por lap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workout_id": {"type": "string", "description": "ID do treino"},
            },
            "required": ["workout_id"],
        },
    },
    {
        "name": "tp_get_workout_prs",
        "description": (
            "Retorna PRs (personal records) batidos DURANTE um treino específico. "
            "Use após avaliar execução de treino — se houve PR (ex: melhor potência de 5min, melhor 1km), "
            "o coach parabeniza com base concreta. Não chame especulativamente — só quando o treino foi forte "
            "(potência alta, pace forte) e vale checar se rendeu PR."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workout_id": {"type": "string", "description": "ID do treino"},
            },
            "required": ["workout_id"],
        },
    },
    {
        "name": "tp_get_peaks",
        "description": (
            "Best efforts HISTÓRICOS do atleta para comparar com execução atual. "
            "sport: 'Bike' ou 'Run'. "
            "pr_type bike: power5sec, power1min, power5min, power10min, power20min, power60min, power90min, "
            "hR5sec/1min/5min/10min/20min/60min/90min. "
            "pr_type run: speed400Meter, speed800Meter, speed1K, speed1Mi, speed5K, speed5Mi, speed10K, "
            "speed10Mi, speedHalfMarathon, speedMarathon, hR5sec/1min/5min/10min/20min/60min/90min. "
            "Use para contextualizar: 'seu best 20min é 245W, hoje você fez 220W (10% abaixo)'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sport":   {"type": "string", "enum": ["Bike", "Run"], "description": "Esporte"},
                "pr_type": {"type": "string", "description": "Tipo de PR — ver descrição"},
                "days":    {"type": "integer", "description": "Janela de histórico em dias (default 3650 = all-time)"},
            },
            "required": ["sport", "pr_type"],
        },
    },
    {
        "name": "tp_get_atp",
        "description": (
            "ANNUAL TRAINING PLAN do atleta: TSS semanal alvo, fase atual (base 1/2, build 1/2, peak, taper, race, recovery), "
            "provas. ESSENCIAL para periodização — diz em qual fase do macrociclo o atleta está e quanto TSS a semana deve ter. "
            "Use no início de uma avaliação semanal ou quando o atleta perguntar 'estou no caminho certo pra prova?'."
        ),
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
        "name": "tp_get_weekly_summary",
        "description": (
            "VISÃO CONSOLIDADA de uma semana: workouts + métricas saúde + fitness (CTL/ATL/TSB) num único retorno. "
            "Use quando o atleta pedir 'resumo da semana' ou 'fechamento semanal' — 1 chamada substitui 3. "
            "week_of opcional (qualquer dia da semana desejada — default: semana corrente)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "week_of": {"type": "string", "description": "Qualquer dia YYYY-MM-DD da semana desejada (opcional)"},
            },
            "required": [],
        },
    },
    {
        "name": "tp_get_workout_types",
        "description": (
            "Lista oficial de sport types e subtypes do TP com seus IDs. "
            "Use APENAS quando precisar criar um treino com subtype específico (ex: 'Mountain Bike' em vez de 'Bike' genérico) "
            "e não souber o ID exato. Em 95% dos casos não é necessária — Swim/Bike/Run cobre."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "tp_get_workout_comments",
        "description": (
            "Comentários trocados num treino — incluindo feedback do COACH HUMANO (assessoria Spadotto) e do próprio atleta. "
            "Use ao avaliar um treino para incorporar o que o coach real falou — alinha sua análise com a orientação da assessoria "
            "em vez de contradizê-la. Ex: se o Spadotto comentou 'segura a intensidade essa semana', respeite isso."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workout_id": {"type": "string", "description": "ID do treino"},
            },
            "required": ["workout_id"],
        },
    },
    {
        "name": "tp_get_focus_event",
        "description": (
            "A prova-ALVO principal (prioridade A) do atleta: nome, data, distância, metas e resultados. "
            "É o norte de toda periodização — use para calcular semanas restantes e calibrar se a carga atual faz sentido "
            "rumo à prova. Chame quando o atleta perguntar sobre a prova, ou no início de planejamento de bloco/semana."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "tp_get_next_event",
        "description": (
            "A próxima prova/evento futuro mais próximo no calendário (qualquer prioridade). "
            "Use para countdown ('faltam X dias') e para saber se há prova chegando que exige taper ou ajuste de carga."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "tp_get_athlete_settings",
        "description": (
            "Configurações OFICIAIS do atleta no TP: FTP, limiar de FC, pace limiar, CSS, e todas as zonas "
            "(potência, FC, pace) calculadas. FONTE DA VERDADE para zonas — use quando precisar dos valores exatos "
            "ou quando suspeitar que o FTP/limiar mudou. Mais confiável que os números fixos do prompt."
        ),
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
        elif name == "tp_get_workout":
            return await tp_get_workout(inputs["workout_id"])
        elif name == "tp_analyze_workout":
            return await tp_analyze_workout(inputs["workout_id"])
        elif name == "tp_get_workout_prs":
            return await tp_get_workout_prs(inputs["workout_id"])
        elif name == "tp_get_peaks":
            return await tp_get_peaks(
                sport=inputs["sport"],
                pr_type=inputs["pr_type"],
                days=inputs.get("days", 3650),
            )
        elif name == "tp_get_atp":
            return await tp_get_atp(inputs["start_date"], inputs["end_date"])
        elif name == "tp_get_weekly_summary":
            return await tp_get_weekly_summary(inputs.get("week_of"))
        elif name == "tp_get_workout_types":
            return await tp_get_workout_types()
        elif name == "tp_get_workout_comments":
            return await tp_get_workout_comments(inputs["workout_id"])
        elif name == "tp_get_focus_event":
            return await tp_get_focus_event()
        elif name == "tp_get_next_event":
            return await tp_get_next_event()
        elif name == "tp_get_athlete_settings":
            return await tp_get_athlete_settings()
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
            tss_str = f" | TSS plan {round(tss)}" if tss else ""
            # Usa campo computed 'completed' se existir, senão heurística
            done = (
                w.get("completed") is True
                or w.get("type") == "completed"
                or w.get("duration_actual") is not None
                or bool(w.get("tss_actual"))
                or bool(w.get("distance_actual"))
            )
            wid       = w.get("id", "")
            status    = "✓ CONCLUÍDO" if done else "⏳ pendente"
            id_str    = f" [id:{wid}]" if wid else ""
            # Dados realizados (quando concluído) — essenciais para avaliação
            real_parts = []
            dur_real = w.get("duration_actual")
            tss_real = w.get("tss_actual")
            dist_real = w.get("distance_actual")
            if dur_real:  real_parts.append(f"{int(dur_real * 60) if dur_real < 24 else int(dur_real/60)}min real")
            if tss_real:  real_parts.append(f"TSS real {round(tss_real)}")
            if dist_real: real_parts.append(f"{round(dist_real/1000, 2)}km real" if dist_real > 100 else f"{dist_real}km real")
            real_str  = f" → {', '.join(real_parts)}" if real_parts else ""
            # Descrição do plano (blocos/intervalos) — necessária para comparar plano vs real
            desc = (w.get("description") or "").strip()
            desc_str = f"\n      Plano: {desc[:280]}{'…' if len(desc) > 280 else ''}" if desc else ""
            lines.append(f"  • [{sport}] {title} — {dur_min}min{tss_str}{real_str} — {status}{id_str}{desc_str}")
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
        hr_rep = metr.get("Resting Heart Rate") or metr.get("Pulse") or metr.get("resting_hr")
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
            max_tokens=1500,
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
