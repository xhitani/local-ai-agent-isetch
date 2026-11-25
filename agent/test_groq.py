import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

key = os.getenv("GROQ_API_KEY")
if not key:
    raise SystemExit("GROQ_API_KEY est absente de .env")
print("Clé trouvée, longueur :", len(key), "| commence par gsk_ :", key.startswith("gsk_"))

c = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
m = os.getenv("API_MODEL", "llama-3.3-70b-versatile")
r = c.chat.completions.create(
    model=m, messages=[{"role": "user", "content": "Dis bonjour en une phrase."}])
print(m, "->", r.choices[0].message.content)
