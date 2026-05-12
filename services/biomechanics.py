"""Biomechanics — análise de movimento em tempo real via Claude Vision.

Suporta dois modos:
  - run:  corrida (frames a cada ~3s, lote de 10 frames)
  - gym:  exercícios de academia (frames a cada ~2s, lote de 6 frames)
"""

import os
import anthropic


def _get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


_RUN_PROMPT = """Você é um especialista em biomecânica de corrida com olho clínico.
Analise esta sequência de frames e identifique PADRÕES RECORRENTES (não eventos isolados).

CHECKPOINTS (prioridade):
1. Over-striding — pé pousa à frente do quadril (freada, impacto calcanhar)
2. Queda lateral de quadril — Trendelenburg, glúteo médio fraco
3. Oscilação vertical excessiva — energia desperdiçada
4. Inclinação do tronco pela cintura, não pelos tornozelos
5. Braços cruzando linha média ou cotovelos abertos
6. Cabeça baixa, pescoço tenso

RESPOSTA — duas partes separadas por "|":
1. Feedback TTS (máximo 8 palavras, imperativo, português)
2. Observação técnica (1 frase com dado: "em X/Y frames", "consistente", etc.)

Exemplos:
encurta a passada, pé sob o quadril|Over-striding em 7/10 frames — pé ~20cm à frente do CM
boa mecânica, mantém o ritmo|Postura consistente — cadência regular, quadril estável"""


_GYM_PROMPT = """Você é um personal trainer especialista em biomecânica de exercícios.
Analise esta sequência de frames de um exercício de academia.

PASSO 1 — Identifique o exercício pelos frames (agachamento, rosca direta, terra, supino, etc.)
PASSO 2 — Aplique os checkpoints específicos desse exercício:

AGACHAMENTO: joelhos colapsando (valgus), profundidade, coluna neutra, calcanhar levantando, peso no ante-pé
TERRA: arredondamento lombar, barra afastada do corpo, queda de quadril antes de puxar, hiperextensão no topo
ROSCA DIRETA: cotovelo saindo do corpo, swing de lombar, amplitude incompleta, pulso dobrado
SUPINO: cotovelo flaring (>90°), barra descendo longe do peito, arco excessivo, pés sem contato
DESENVOLVIMENTO: hiperextensão cervical, falta de rotação escapular, carga desigual
EXERCÍCIO DESCONHECIDO: avalie postura geral, alinhamento, controle da carga, range of motion

ANALISE OS PADRÕES RECORRENTES — se o erro aparece em várias reps, é hábito.

RESPOSTA — duas partes separadas por "|":
1. Feedback TTS (máximo 8 palavras, imperativo, português)
2. Observação técnica (1 frase: exercício identificado + erro principal + frequência)

Exemplos:
joelhos para fora, empurra pelos calcanhares|Agachamento — valgus de joelho em 5/6 reps, especialmente no descendo
boa execução, controla a descida|Rosca direta — amplitude completa, cotovelo estável em todas as reps"""


def _parse_response(raw: str) -> dict:
    raw = raw.strip()
    if "|" in raw:
        parts = raw.split("|", 1)
        feedback = parts[0].strip().lower().rstrip(".")
        observacao = parts[1].strip()
    else:
        feedback = raw.lower().rstrip(".")
        observacao = ""
    return {"feedback": feedback, "observacao": observacao}


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
        model="claude-opus-4-5", max_tokens=120,
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
        model="claude-opus-4-5", max_tokens=150,
        system=_GYM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return _parse_response(response.content[0].text)
