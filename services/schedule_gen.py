"""AI schedule generator — adapts recommendations based on workout order."""

import os
import json
import anthropic


def _get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


from services.coaching_brain import build_system_prompt

_SCHEDULE_ROLE = """Você está gerando o PLANO DIÁRIO — o atleta definiu a ordem dos treinos e você analisa e recomenda.
Foco especial em nutrição periodizada e gestão de carga entre blocos.

REGRAS DE NUTRICAO:
- nutricao_pre: especifique ALIMENTOS concretos, QUANTIDADE e TIMING (ex: "45min antes: 1 banana + 200ml suco de beterraba + 30g whey")
- nutricao_durante: especifique hidratacao e carboidratos em gramas por hora se duracao > 60min (ex: "a cada 20min: 1 gel de carboidrato (25g) + 150ml agua. Meta: 60g carb/h")
- nutricao_pos: janela de recuperacao — proteina + carbo em 30min (ex: "em ate 30min: 300ml leite chocolate OU 30g whey + 1 banana + agua com sal")
- Para blocos curtos (<45min) nutricao_durante pode ser so hidratacao
- Considere o timing entre blocos — se ha menos de 2h de intervalo, nutricao_pos do bloco anterior = nutricao_pre do proximo

Responda SOMENTE em JSON valido com esta estrutura:
{
  "aprovacao_sequencia": "otima" | "boa" | "aceitavel" | "atencao",
  "comentario_sequencia": "string — avaliacao da ordem escolhida em 1-2 frases",
  "alertas": ["string"],
  "blocos": [
    {
      "workout_id": "string",
      "horario": "string (HH:MM)",
      "duracao_min": number,
      "aquecimento": "string — duracao e tipo de aquecimento especifico para este esporte",
      "execucao": "string — instrucao principal para este bloco considerando posicao na sequencia",
      "nutricao_pre": "string — alimentos, quantidades e timing especificos antes deste bloco",
      "nutricao_durante": "string — hidratacao e energia durante o treino com quantidades",
      "nutricao_pos": "string — recuperacao imediata com alimentos e quantidades",
      "sinais_de_alerta": "string — o que observar dado que vem antes/depois"
    }
  ],
  "nutricao_geral": {
    "cafe_manha": "string — refeicao completa com horario, alimentos e quantidades para o dia de treino",
    "janela_treino": "string — estrategia nutricional geral para toda a janela de treinos",
    "recuperacao": "string — jantar e suplementacao noturna para otimizar recuperacao"
  }
}"""

ADAPT_PROMPT = build_system_prompt(_SCHEDULE_ROLE)


def adapt_schedule(
    workout_order: list,
    start_time: str,
    metrics: dict,
    fitness: dict,
    date: str,
) -> dict:
    """Generate AI recommendations adapted to the chosen workout order."""

    msg = f"""O atleta montou o seguinte cronograma para {date}:

ORDEM E HORARIOS DEFINIDOS PELO ATLETA:
{json.dumps(workout_order, ensure_ascii=False, indent=2)}

HORARIO DE INICIO: {start_time}

METRICAS DE SAUDE:
{json.dumps(metrics, ensure_ascii=False, indent=2)}

FORMA FISICA (CTL/ATL/TSB):
{json.dumps(fitness, ensure_ascii=False, indent=2)}

Analise a sequencia e gere recomendacoes especificas para cada bloco nesta ordem.
Considere: recuperacao entre blocos, nutricao no timing certo, sinais de alerta para fadiga.
Se a sequencia nao for ideal, explique e sugira como compensar.
"""

    response = _get_client().messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=ADAPT_PROMPT,
        messages=[{"role": "user", "content": msg}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    return json.loads(text)
