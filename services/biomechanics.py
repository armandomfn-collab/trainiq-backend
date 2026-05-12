"""Biomechanics — análise de corrida em tempo real via Claude Vision."""

import os
import anthropic


def _get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


_BIOMECH_PROMPT = """Você é um especialista em biomecânica de corrida com olho clínico.
Você recebeu uma sequência de frames capturados a cada ~3 segundos durante uma corrida.

ANALISE OS PADRÕES RECORRENTES — não eventos isolados. Se algo aparece em 3+ frames, é um hábito.

CHECKPOINTS (em ordem de prioridade):
1. Over-striding — pé pousa muito à frente do quadril (freada, impacto no calcanhar)
2. Queda lateral de quadril — pelve desce de um lado em várias passadas (Trendelenburg, glúteo médio)
3. Oscilação vertical excessiva — corpo sobe muito a cada passada (energia desperdiçada)
4. Inclinação do tronco — tombamento pela cintura, não pelos tornozelos
5. Braços — cruzando linha média, cotovelos abertos, sem balanço consistente
6. Cabeça/pescoço — olhando para baixo de forma consistente, pescoço tenso

FORMATO DE RESPOSTA — duas partes separadas por "|":
1. Feedback TTS (máximo 8 palavras, imperativo, português)
2. Observação técnica (1 frase, para exibir na tela)

Exemplo:
encurta a passada, pé sob o quadril|Over-striding em 7/10 frames — pé pousa 20cm à frente do centro de massa

Outro exemplo:
boa mecânica, mantém o ritmo|Postura consistente nos 30s — cadência regular, quadril estável

Se houver múltiplos problemas, cite o mais recorrente e crítico."""


def analyze_running_sequence(frames: list[str], media_type: str = "image/jpeg") -> dict:
    """
    Analisa uma sequência de frames de corrida (30s de filmagem).
    Identifica padrões recorrentes e retorna feedback consolidado.

    Args:
        frames: lista de imagens em base64 (JPEG), capturadas a cada ~3s
        media_type: tipo MIME das imagens

    Returns:
        dict com 'feedback' (TTS, curto) e 'observacao' (técnica, para tela)
    """
    client = _get_client()

    # Monta o conteúdo com todos os frames
    content = []
    for i, frame_b64 in enumerate(frames):
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": frame_b64,
            },
        })

    content.append({
        "type": "text",
        "text": (
            f"Sequência de {len(frames)} frames capturados a cada ~3 segundos "
            f"(~{len(frames) * 3}s de corrida). "
            "Analise os padrões recorrentes e dê o feedback mais importante."
        ),
    })

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=120,
        system=_BIOMECH_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()

    # Separa feedback TTS da observação técnica
    if "|" in raw:
        parts = raw.split("|", 1)
        feedback = parts[0].strip().lower().rstrip(".")
        observacao = parts[1].strip()
    else:
        feedback = raw.lower().rstrip(".")
        observacao = ""

    return {"feedback": feedback, "observacao": observacao}
