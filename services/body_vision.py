"""Extract bioimpedance values from an image using Claude Vision."""

import json
import os
from datetime import date

import anthropic


EXTRACTION_PROMPT = """Voce recebeu uma foto de resultado de balanca de bioimpedancia ou avaliacao corporal.
Extraia todos os valores visiveis na imagem, incluindo a data se aparecer.

Data de referencia (use SOMENTE se nao encontrar data na imagem): {today}

Retorne SOMENTE JSON valido, sem markdown, sem texto adicional:
{{
  "date": "<data no formato YYYY-MM-DD extraida da imagem, ou {today} se nao houver data visivel>",
  "weight_kg": <numero decimal ou null>,
  "body_fat_pct": <numero decimal ou null>,
  "muscle_mass_kg": <numero decimal ou null>,
  "visceral_fat": <inteiro ou null>,
  "water_pct": <numero decimal ou null>,
  "bone_mass_kg": <numero decimal ou null>,
  "bmr_kcal": <inteiro ou null>,
  "bmi": <numero decimal ou null>,
  "notes": "<marca e modelo da balanca se visivel, senao null>"
}}

REGRAS IMPORTANTES:
- DATA: procure ativamente por datas na imagem — pode aparecer como "2024/03/15", "15/03/2024", "Mar 15 2024", "15-03-24" etc. Converta para YYYY-MM-DD. Se nao houver data visivel use {today}.
- Nao invente valores numericos. Se um campo nao estiver visivel na imagem, coloque null.
- Peso: pode aparecer como "Weight", "Peso", "Body Weight" — em kg
- Gordura corporal: "Body Fat %", "Fat Mass", "Gordura", "% Gordura" — em %
- Massa muscular: "Muscle Mass", "Skeletal Muscle", "Musculo", "SMM" — em kg
- Gordura visceral: "Visceral Fat", "Gordura Visceral", numero geralmente entre 1-30
- Agua corporal: "Body Water", "TBW", "Agua", "% Agua" — em %
- Massa ossea: "Bone Mass", "Bone", "Osseo" — em kg
- TMB/BMR: "BMR", "Basal Metabolic Rate", "Metabolismo Basal" — em kcal/dia
- IMC: "BMI", "IMC", "Body Mass Index" — numero decimal
- Extraia os numeros exatamente como aparecem (nao arredonde)
"""


async def extract_bioimpedance_from_image(image_base64: str, media_type: str = "image/jpeg") -> dict:
    """Send bioimpedance exam image to Claude Vision and extract structured values."""
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    today = date.today().isoformat()

    response = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_base64,
                    },
                },
                {
                    "type": "text",
                    "text": EXTRACTION_PROMPT.format(today=today),
                },
            ],
        }],
    )

    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()

    return json.loads(text)
