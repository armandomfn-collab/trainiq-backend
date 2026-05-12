"""Biomechanics — análise de corrida em tempo real via Claude Vision."""

import os
import anthropic


def _get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


_BIOMECH_PROMPT = """Você é um especialista em biomecânica de corrida com olho clínico.
Analise este frame de vídeo lateral de um atleta correndo.

CHECKPOINTS (em ordem de prioridade):
1. Over-striding — pé pousa muito à frente do quadril (causa: impacto no calcanhar, freada)
2. Queda lateral de quadril — pelve desce de um lado (sinal de Trendelenburg, glúteo médio fraco)
3. Oscilação vertical excessiva — corpo sobe muito a cada passada (energia desperdiçada)
4. Inclinação do tronco — tombamento pela cintura, não pelos tornozelos
5. Braços — cruzando linha média do corpo, cotovelos muito abertos, sem balanço
6. Cabeça/pescoço — olhando para baixo, pescoço tenso/projetado

REGRA:
- Identifique o problema MAIS CRÍTICO visível neste frame
- Se não houver problema claro: confirme o que está bom ("cadência boa, mantém")
- Resposta: máximo 8 palavras, imperativo, português, sem pontuação final

Exemplos de resposta:
encurta a passada, pé sob o quadril
solta os ombros, braços para frente
menos salto, desliza para frente
levanta o olhar, pescoço relaxado
quadril nivelado, ativa o glúteo direito
inclinação do tronco, parte dos tornozelos
boa postura, mantém o ritmo"""


def analyze_running_frame(frame_base64: str, media_type: str = "image/jpeg") -> dict:
    """
    Analisa um frame de corrida e retorna feedback corretivo curto para TTS.

    Args:
        frame_base64: imagem em base64 (JPEG)
        media_type: tipo MIME da imagem

    Returns:
        dict com 'feedback' (texto curto, max ~8 palavras)
    """
    client = _get_client()

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=50,
        system=_BIOMECH_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": frame_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Qual o feedback mais importante para este atleta agora?",
                    },
                ],
            }
        ],
    )

    feedback = response.content[0].text.strip()
    # Normaliza: minúsculo, sem ponto final
    feedback = feedback.lower().rstrip(".")

    return {"feedback": feedback}
