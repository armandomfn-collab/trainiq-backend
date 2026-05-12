"""Watch Gmail inbox for Garmin LiveTrack emails and extract the URL."""

import email
import imaplib
import os
import re
from datetime import datetime, timedelta

GARMIN_SENDER = "noreply@garmin.com"
LIVETRACK_URL_RE = re.compile(
    r"https://livetrack\.garmin\.com/session/[a-f0-9\-]+/token/[A-Za-z0-9]+"
)


def find_latest_livetrack_url() -> str | None:
    """
    Connects to Gmail via IMAP and looks for a recent Garmin LiveTrack email.
    Returns the LiveTrack URL if found, None otherwise.

    Requires env vars:
        GMAIL_USER         — ex: trainiq.coach@gmail.com
        GMAIL_APP_PASSWORD — app password (not the account password)
    """
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD")

    if not gmail_user or not gmail_pass:
        print("[EmailWatcher] GMAIL_USER / GMAIL_APP_PASSWORD nao configurados.")
        return None

    mail = None
    try:
        # Gmail: imap.gmail.com | Outlook: outlook.office365.com
        imap_server = "imap.gmail.com" if "gmail" in gmail_user else "outlook.office365.com"
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(gmail_user, gmail_pass)
        # Busca em All Mail (pega inbox, promoções, atualizações, etc.)
        mail.select('[Gmail]/All Mail')

        # Busca emails do Garmin nas ultimas 24h (lidos ou nao)
        since = (datetime.now() - timedelta(hours=24)).strftime("%d-%b-%Y")
        _, ids = mail.search(None, f'(FROM "{GARMIN_SENDER}" SINCE "{since}")')

        if not ids[0]:
            print(f"[EmailWatcher] Nenhum email do Garmin encontrado desde {since}")
            return None

        # Percorre do mais recente ao mais antigo procurando URL válida
        all_ids = ids[0].split()
        print(f"[EmailWatcher] {len(all_ids)} email(s) do Garmin encontrado(s)")

        for email_id in reversed(all_ids):
            _, data   = mail.fetch(email_id, "(RFC822)")
            raw_email = data[0][1]
            msg       = email.message_from_bytes(raw_email)
            body      = _extract_body(msg)

            # Limpa HTML encoding e quebras de linha que podem quebrar a URL
            body = re.sub(r'=\r?\n', '', body)   # quoted-printable line breaks
            body = re.sub(r'=3D', '=', body)      # = encoded
            body = re.sub(r'\s+', ' ', body)       # normaliza espaços

            match = LIVETRACK_URL_RE.search(body)
            if match:
                url = match.group(0).rstrip('>"\'')  # remove chars HTML ao final
                mail.store(email_id, "+FLAGS", "\\Seen")
                print(f"[EmailWatcher] LiveTrack URL encontrada: {url}")
                return url
            else:
                print(f"[EmailWatcher] Email ID {email_id}: sem URL de LiveTrack no corpo")

    except Exception as e:
        print(f"[EmailWatcher] Erro: {e}")
    finally:
        if mail:
            try:
                mail.logout()
            except Exception:
                pass

    return None


def _extract_body(msg: email.message.Message) -> str:
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct in ("text/plain", "text/html"):
                try:
                    body += part.get_payload(decode=True).decode("utf-8", errors="ignore")
                except Exception:
                    pass
    else:
        try:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
        except Exception:
            pass
    return body
