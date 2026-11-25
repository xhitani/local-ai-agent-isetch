import os
import re
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# Adresse de la scolarité
# Change this to the real ISET Charguia scolarity email.
MAIL_TO = os.getenv("MAIL_TO", "")


def valid_email(email: str) -> bool:
    return bool(
        re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            (email or "").strip()
        )
    )


def mode() -> str:
    if SMTP_USER and SMTP_PASSWORD:
        return "smtp"
    return "simulation"


def recipient() -> str:
    return MAIL_TO or "(destinataire non configuré)"


def send_request(subject: str, body: str, copy_to: str | None = None) -> str:
    if mode() != "smtp":
        return "Mode simulation : aucun e-mail réel n'a été envoyé."

    if not MAIL_TO:
        return (
            "Envoi impossible : MAIL_TO n'est pas configuré. "
            "Ajoutez l'adresse e-mail de la scolarité dans .env."
        )

    if not valid_email(MAIL_TO):
        return "Envoi impossible : MAIL_TO est invalide."

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = MAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    if copy_to:
        if not valid_email(copy_to):
            return "Envoi impossible : votre adresse e-mail est invalide."
        msg["Cc"] = copy_to

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.send_message(msg)

        if copy_to:
            return f"Email envoyé avec succès à {MAIL_TO} avec une copie à {copy_to}."
        return f"Email envoyé avec succès à {MAIL_TO}."

    except Exception as e:
        return f"Envoi impossible : {e}"
