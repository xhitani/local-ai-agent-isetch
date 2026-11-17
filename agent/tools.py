"""Outils de l'assistant de scolarité. Aucune logique de modèle ici :
ces fonctions sont réutilisées par l'agent, le serveur MCP, Streamlit et les tests."""
import difflib
import json
import os
import re
import unicodedata
from datetime import datetime, timedelta

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.0"))  # plus petit = plus strict
MARGIN = float(os.getenv("MARGIN", "0.12"))             # écart max avec le meilleur article
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=180)

CONSIGNE = ("\n\nConsigne : réponds en 1 ou 2 phrases, uniquement avec ce que disent "
            "ces informations. N'ajoute aucun conseil ni généralité.")
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi"]

# Question originale de l'étudiant : la recherche l'utilise à la place de la
# requête reformulée par le modèle (qui la déforme parfois).
ORIGINAL_QUESTION = ""


def set_question(q: str) -> None:
    global ORIGINAL_QUESTION
    ORIGINAL_QUESTION = q


def norm(s: str) -> str:
    """minuscules, sans accents, sans underscores ni espaces en trop"""
    s = unicodedata.normalize("NFD", str(s).lower().replace("_", " "))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


# ---------- Calendrier (session de janvier 2027) ----------
CALENDRIER = {
    "algorithmique": {"nom": "Algorithmique", "date": "05/01/2027",
                      "examen": "05/01/2027, 9h, amphi A",
                      "rattrapage": "13/01/2027, 14h, salle B10"},
    "bases de donnees": {"nom": "Bases de données", "date": "08/01/2027",
                         "examen": "08/01/2027, 9h, amphi A",
                         "rattrapage": "15/01/2027, 9h, salle B12"},
    "reseaux": {"nom": "Réseaux", "date": "11/01/2027",
                "examen": "11/01/2027, 9h, amphi B",
                "rattrapage": "18/01/2027, 9h, salle B14"},
    "anglais": {"nom": "Anglais", "date": "12/01/2027",
                "examen": "12/01/2027, 14h, salle B11",
                "rattrapage": "19/01/2027, 10h, salle B11"},
    "projet de fin de module": {"nom": "Projet de fin de module", "date": None,
                                "examen": "Remise le 20/01/2027",
                                "rattrapage": "Pas de rattrapage"},
}


def find_module(name: str):
    """Retrouve la clé d'un module, même avec une faute ou un pluriel."""
    n = norm(name)
    if len(n) < 4:
        return None
    if n in CALENDRIER:
        return n
    for k in CALENDRIER:
        if k in n or n in k:
            return k
    close = difflib.get_close_matches(n, list(CALENDRIER), n=1, cutoff=0.75)
    return close[0] if close else None


def date_limite(key: str, hospitalisation: bool = False) -> datetime:
    """Art. 4 : 3 jours ouvrables (lun-ven), jour de l'examen non compté.
    Art. 6 : 7 jours ouvrables en cas d'hospitalisation."""
    d = datetime.strptime(CALENDRIER[key]["date"], "%d/%m/%Y")
    n = 7 if hospitalisation else 3
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


# ---------- Outil 1 : calendrier ----------
def get_calendrier(module: str) -> str:
    key = find_module(module)
    if key is None:
        return "Module introuvable dans le calendrier."
    m = CALENDRIER[key]
    return json.dumps({"module": m["nom"], "examen": m["examen"],
                       "rattrapage": m["rattrapage"]}, ensure_ascii=False)


# ---------- Outil 2 : recherche dans le règlement (RAG) ----------
SYNONYMES = {
    r"\bpfe\b": "stage de fin d'année",
    r"\bpfa\b": "stage de fin d'année",
    r"\bstage de fin d.etudes\b": "stage de fin d'année",
    r"\bduree\b": "durée",
}
_col = None


def _collection():
    global _col
    if _col is None:
        _col = chromadb.PersistentClient(
            path=os.path.join(ROOT, "chroma_db")).get_collection("reglement")
    return _col


def expand_query(q: str) -> str:
    for pattern, repl in SYNONYMES.items():
        q = re.sub(pattern, repl, q, flags=re.IGNORECASE)
    return q


def search_reglement(query: str) -> str:
    query = expand_query(ORIGINAL_QUESTION or query)
    emb = client.embeddings.create(model="bge-m3", input=[query]).data[0].embedding
    r = _collection().query(query_embeddings=[emb], n_results=3,
                            include=["documents", "metadatas", "distances"])
    rows = list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))
    print("          distances : " + ", ".join(
        f"art.{m['article']}={dist:.2f}" for _, m, dist in rows))
    if not rows or rows[0][2] > MAX_DISTANCE:
        return "Aucun article pertinent trouvé dans le règlement."
    best = rows[0][2]
    out = [f"[Article {m['article']}] {d}" for d, m, dist in rows
           if dist <= MAX_DISTANCE and dist <= best + MARGIN]
    return "\n".join(out) + CONSIGNE


# ---------- Outil 3 : certificat médical ----------
def certificat_medical(module: str, hospitalisation: bool = False) -> str:
    key = find_module(module)
    if key is None:
        return "Module introuvable dans le calendrier."
    if CALENDRIER[key]["date"] is None:
        return ("Ce module est évalué par projet : pas de rattrapage (Article 10), "
                "remise le 20/01/2027.")
    m = CALENDRIER[key]
    d = date_limite(key, hospitalisation)
    art = "Article 6, bulletin d'hospitalisation requis" if hospitalisation else "Article 4"
    return (f"Examen : {m['examen']}. "
            f"Date limite du certificat : {JOURS[d.weekday()]} {d.strftime('%d/%m/%Y')} ({art}). "
            f"Rattrapage : {m['rattrapage']} (Article 7). "
            f"Demande écrite au responsable de scolarité avec le justificatif (Article 8)."
            + CONSIGNE)


# ---------- Outil 4 : demande de rattrapage ----------
def verifier_module_rattrapage(module: str):
    """Renvoie (clé, None) si le module a un rattrapage, sinon (None, message d'erreur)."""
    key = find_module(module)
    dispo = "algorithmique, bases de données, réseaux, anglais"
    if key is None:
        return None, f"Pour quel module ? Modules avec rattrapage : {dispo}."
    if CALENDRIER[key]["date"] is None:
        return None, ("Ce module est évalué uniquement par projet : il n'y a pas de "
                      "session de rattrapage (Article 10).")
    return key, None


def build_demande(key: str, nom: str, numero: str, hospitalisation: bool) -> str:
    """Brouillon de la demande (Article 8). Modèle fixe : le modèle IA n'écrit rien."""
    m = CALENDRIER[key]
    dl = date_limite(key, hospitalisation).strftime("%d/%m/%Y")
    if hospitalisation:
        motif = "hospitalisation"
        pj = f"bulletin d'hospitalisation (Article 6), à remettre avant le {dl}"
    else:
        motif = "maladie"
        pj = ("certificat médical daté du jour de l'examen ou de la veille, signé et "
              f"tamponné (Articles 4 et 5), à remettre avant le {dl}")
    return (
        f"Objet : Demande de session de rattrapage – {m['nom']}\n"
        f"À l'attention du responsable de scolarité, ISET Charguia\n\n"
        f"Je soussigné(e) {nom}, numéro d'étudiant {numero}, n'ai pas pu me présenter "
        f"à l'examen du module « {m['nom']} » prévu le {m['date']}, pour cause de {motif}.\n"
        f"Conformément aux articles 7 et 8 du règlement des études, je sollicite le "
        f"passage de cet examen à la session de rattrapage : {m['rattrapage']}.\n\n"
        f"Pièce jointe : {pj}.\n\n"
        f"Fait à Charguia, le {datetime.now().strftime('%d/%m/%Y')}.\n"
        f"Signature : {nom}"
    )
