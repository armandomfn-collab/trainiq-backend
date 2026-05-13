"""Biomechanics — análise de movimento em tempo real via Claude Vision.

Suporta dois modos:
  - run:  corrida (frames a cada ~3s, lote de 10 frames)
  - gym:  exercícios de academia (frames a cada ~2s, lote de 6 frames)
"""

import os
import anthropic


def _get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


_SEVERITY_RULES = """
CRITÉRIO DE SEVERIDADE — seja criterioso, não aponte tudo:

critical   → risco real de lesão (lombar arredondada no terra, joelho colapsa sob carga, cervical hiperextendida)
significant → compromete o resultado de forma consistente (over-striding em 6+/10 frames, valgus em 4+/6 reps, oscilação vertical excessiva)
minor      → imperfeição técnica que não afeta resultado nem segurança (leve desvio de cotovelo, posição de pulso, cabeça ligeiramente baixa)
ok         → execução dentro do aceitável — não há nada que mereça correção agora

REGRA FUNDAMENTAL: na dúvida entre minor e ok, escolha ok.
Só classifique como significant ou critical se você veria isso num atleta e pensaria "preciso corrigir agora"."""

_RUN_PROMPT = """Você é um especialista em biomecânica de corrida com olho clínico.
Analise esta sequência de frames e identifique PADRÕES RECORRENTES — não eventos isolados.

CHECKPOINTS:
1. Over-striding — pé pousa muito à frente do quadril (freada, impacto calcanhar)
2. Queda lateral de quadril — Trendelenburg, glúteo médio fraco
3. Oscilação vertical excessiva — corpo sobe demais a cada passada
4. Inclinação do tronco pela cintura, não pelos tornozelos
5. Braços cruzando linha média ou cotovelos muito abertos
6. Cabeça baixa, pescoço tenso de forma consistente
""" + _SEVERITY_RULES + """

RESPOSTA — três partes separadas por "|":
1. severity: ok | minor | significant | critical
2. Feedback TTS (máximo 8 palavras, imperativo, português) — vazio se ok
3. Observação técnica (1 frase) — vazio se ok

Exemplos:
significant|encurta a passada, pé sob o quadril|Over-striding em 7/10 frames — pé ~20cm à frente do CM
critical|joelho colapsando, para e ajusta a carga|Valgus bilateral severo — risco de lesão imediato
minor||Leve oscilação de cotovelo esquerdo — não afeta performance
ok|||Mecânica consistente nos 30s — nenhuma correção necessária"""


_GYM_PROMPT = """Você é um personal trainer especialista em biomecânica de exercícios.
Analise esta sequência de frames de academia.

PASSO 1 — Identifique o exercício pelos frames
PASSO 2 — Aplique os checkpoints específicos:

AGACHAMENTO: valgus de joelho (crítico), lombar arredondada (crítico), profundidade insuficiente, calcanhar levantando
TERRA: lombar arredondada (crítico), barra afastada do corpo, hips subindo antes do tronco
ROSCA DIRETA: swing de lombar para gerar impulso, cotovelo muito à frente, amplitude incompleta consistente
SUPINO: cotovelo flaring severo (crítico), barra longe do peito de forma consistente
DESENVOLVIMENTO: hiperextensão cervical (crítico), carga muito desigual
QUALQUER: analise alinhamento geral, controle excêntrico, amplitude de movimento
""" + _SEVERITY_RULES + """

RESPOSTA — três partes separadas por "|":
1. severity: ok | minor | significant | critical
2. Feedback TTS (máximo 8 palavras, imperativo, português) — vazio se ok
3. Observação técnica (1 frase com exercício + achado) — vazio se ok

Exemplos:
critical|para, reduz a carga, lombar neutra|Levantamento terra — arredondamento lombar severo em todas as reps
significant|joelhos para fora, empurra pelos calcanhares|Agachamento — valgus em 5/6 reps, especialmente na descida
minor||Rosca direta — leve anteriorização de cotovelo, não compromete
ok||Agachamento — profundidade e alinhamento consistentes em todas as reps"""


def _parse_response(raw: str) -> dict:
    raw = raw.strip()
    parts = [p.strip() for p in raw.split("|")]

    severity   = parts[0].lower() if len(parts) > 0 else "ok"
    feedback   = parts[1].lower().rstrip(".") if len(parts) > 1 else ""
    observacao = parts[2] if len(parts) > 2 else ""

    # Normaliza severity
    if severity not in ("ok", "minor", "significant", "critical"):
        severity = "ok"

    return {
        "severity":  severity,
        "feedback":  feedback,
        "observacao": observacao,
    }


def _build_content(frames: list[str], media_type: str, text: str) -> list:
    content = []
    for frame_b64 in frames:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": frame_b64},
        })
    content.append({"type": "text", "text": text})
    return content


def analyze_running_sequence(frames: list[str], media_type: str = "image/jpeg") -> dict:
    """Analisa sequência de corrida (~30s, frames a cada 3s)."""
    content = _build_content(
        frames, media_type,
        f"Sequência de {len(frames)} frames a cada ~3s (~{len(frames)*3}s de corrida). "
        "Identifique padrões recorrentes e dê o feedback mais importante."
    )
    response = _get_client().messages.create(
        model="claude-haiku-4-5", max_tokens=120,
        system=_RUN_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return _parse_response(response.content[0].text)


def analyze_gym_sequence(frames: list[str], exercise: str = "", media_type: str = "image/jpeg") -> dict:
    """Analisa sequência de exercício de academia (~12s, frames a cada 2s)."""
    exercise_hint = f"Exercício informado pelo atleta: {exercise}. " if exercise.strip() else ""
    content = _build_content(
        frames, media_type,
        f"{exercise_hint}Sequência de {len(frames)} frames a cada ~2s (~{len(frames)*2}s de execução). "
        "Identifique o exercício, analise os padrões recorrentes e dê o feedback mais importante."
    )
    response = _get_client().messages.create(
        model="claude-haiku-4-5", max_tokens=150,
        system=_GYM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return _parse_response(response.content[0].text)
