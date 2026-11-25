"""Choix du fournisseur de LLM : local (Ollama) ou API gratuite, avec repli local.
Les embeddings restent TOUJOURS locaux (bge-m3) : l'index ChromaDB est construit avec."""
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# (url compatible OpenAI, variable de la clé, modèle par défaut)
# Les noms de modèles et les quotas gratuits changent : vérifiez-les au lancement.
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY",
             "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/",
               "GEMINI_API_KEY", "gemini-2.0-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                   "meta-llama/llama-3.3-70b-instruct:free"),
}


class _Completions:
    def __init__(self, owner):
        self.o = owner

    def create(self, **kw):
        o = self.o
        if o.remote is not None:
            try:
                resp = o.remote.chat.completions.create(**{**kw, "model": o.remote_model})
                o.last_backend = f"{o.provider} ({o.remote_model})"
                return resp
            except Exception as e:   # quota, réseau, clé invalide, modèle indisponible
                print(f"[api] {o.provider} indisponible ({type(e).__name__}) : "
                      f"repli sur le modèle local", file=sys.stderr)
        o.last_backend = f"local ({kw.get('model')})"
        return o.local.chat.completions.create(**kw)


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class Llm:
    """Même interface que le client OpenAI : llm.chat.completions.create / llm.embeddings.create."""

    def __init__(self, local_url: str, timeout: float):
        self.local = OpenAI(base_url=local_url, api_key="ollama",
                            timeout=timeout, max_retries=0)
        self.provider = os.getenv("LLM_PROVIDER", "local").strip().lower()
        self.remote = None
        self.remote_model = None
        self.last_backend = "local"
        if self.provider in PROVIDERS:
            url, key_var, default_model = PROVIDERS[self.provider]
            key = os.getenv(key_var)
            if key:
                self.remote = OpenAI(base_url=url, api_key=key, max_retries=0,
                                     timeout=float(os.getenv("API_TIMEOUT", "30")))
                self.remote_model = os.getenv("API_MODEL", default_model)
            else:
                print(f"[api] {key_var} absente de .env : mode local", file=sys.stderr)
        self.chat = _Chat(self)
        self.embeddings = self.local.embeddings   # toujours local


def make_llm(local_url: str, timeout: float) -> Llm:
    return Llm(local_url, timeout)
