"""Expo Push Notification service."""

import httpx
from services.database import get_all_tokens

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


async def send_push_notification(title: str, body: str, data: dict = {}):
    """Send push notification to all registered devices."""
    tokens = get_all_tokens()
    if not tokens:
        print("Nenhum dispositivo registrado para push.")
        return

    messages = [
        {
            "to": token,
            "title": title,
            "body": body,
            "data": data,
            "sound": "default",
        }
        for token in tokens
    ]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            EXPO_PUSH_URL,
            json=messages,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        print(f"Push enviado: {response.status_code}")
        return response.json()
