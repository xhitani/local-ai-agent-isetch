"""Contrôle rapide des réponses de l'agent (appels réels à Groq).

    python eval/check_answers.py

Chaque cas vérifie la langue de la réponse, ce qu'elle doit contenir et ce
qu'elle ne doit pas affirmer. Les cas reprennent des erreurs observées.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import scolarite_agent as core  # noqa: E402

# Arabizi words mix letters and digits inside the word ("3andek", "sou2el",
# "7aja"). Plain numbers ("4", "2027"), rooms ("B11") and times ("9h") are not.
ARABIZI = re.compile(
    r"\b[a-z]*[a-z][2-9][a-z]+\w*\b|\b[2-9][a-z]{2,}\w*\b"
    r"|\b(?:mta3|mte3|famma|lazem|bech|najem|tnajjem|3andek|kifech|behi)\b", re.I)

CASES = [
    {
        "q": "rasidi fi ghiyebet",
        "why": "situation personnelle en derja : pas d'accès au dossier, règle de l'article 1",
        "lang": "ar",
        "must": [r"(?i)article\s*1|fasl\s*1|الفصل\s*1", r"(?i)scolarit|سكولاريتي|إدارة"],
        "must_not": [r"(?i)rattrapage.*(?:4|quatre) absences|demander un rattrapage"],
    },
    {
        "q": "J'ai plus de 4 absences non justifiées en Réseaux, que se passe-t-il ?",
        "why": "article 1 : pas d'examen ; pas de rattrapage inventé",
        "lang": "fr",
        "must": [r"(?i)article\s*1", r"(?i)ne (?:peu[xt]|pourr)"],
        "must_not": [r"(?i)(?:devez|devrez|pouvez) demander un rattrapage"],
    },
    {
        "q": "Que doit contenir le certificat médical ?",
        "why": "article 5 : daté du jour de l'examen ou de la veille, signé, tamponné, durée",
        "lang": "fr",
        "must": [r"(?i)article\s*5", r"(?i)veille", r"(?i)tamponn", r"(?i)dur[ée]e"],
        "must_not": [r"(?i)la scolarité vous informera"],
    },
    {
        "q": "9adech andi mn 7a9 fi nhar nghib 3al examen?",
        "why": "délai du certificat (article 4), réponse en derja",
        "lang": "ar",
        "must": [r"3", r"(?i)article\s*4|fasl\s*4|الفصل\s*4"],
        "must_not": [],
    },
    {
        "q": "ratit examen te3 algo chna3ml",
        "why": "examen manqué : démarche (certificat, délai, rattrapage), pas seulement les dates",
        "lang": "ar",
        "must": [r"(?i)certificat|شهادة", r"13/01/2027", r"(?i)article\s*4|fasl\s*4|الفصل\s*4"],
        "must_not": [],
    },
    {
        "q": "esmi Ali Ben Test numro 1234567 module anglais w date 12/01/2027",
        "why": "coordonnées pour la demande : brouillon, pas le calendrier",
        "lang": "ar",
        "must": [r"Ali Ben Test", r"1234567", r"(?i)demande de rattrapage"],
        "must_not": [r"(?i)source : calendrier"],
    },
    {
        "q": "How long is the internship?",
        "why": "article 11, réponse en anglais",
        "lang": "en",
        "must": [r"8", r"(?i)article\s*11"],
        "must_not": [],
    },
    {
        "q": "ey aychk",
        "why": "remerciement en derja : réponse polie, jamais un refus",
        "lang": "ar",
        "must": [],
        "must_not": [r"(?i)ne peux pas|désolé"],
    },
    {
        "q": "ok",
        "why": "simple accusé de réception : réponse polie, jamais un refus",
        "lang": "fr",
        "must": [],
        "must_not": [r"(?i)ne peux pas|désolé"],
    },
    {
        "q": "yo",
        "why": "salutation familière en anglais : accueil, pas un refus",
        "lang": "en",
        "must": [r"(?i)hello"],
        "must_not": [r"(?i)sorry|can only|désolé"],
    },
    {
        "q": "english",
        "why": "demande de répondre en anglais, pas le calendrier du module Anglais",
        "lang": "en",
        "must": [r"(?i)english"],
        "must_not": [r"12/01/2027"],
    },
    {
        "q": "Combien de semaines dure le stage ?",
        "why": "stage : article 11, pas le calendrier des examens",
        "lang": "fr",
        "must": [r"8", r"(?i)article\s*11"],
        "must_not": [r"05/01/2027"],
    },
    {
        "q": "Donne-moi une recette de pizza.",
        "why": "hors sujet : refus",
        "lang": "fr",
        "must": [r"(?i)uniquement|seulement|ne peux"],
        "must_not": [r"(?i)farine|levure|tomate"],
    },
]


def answer_language(text: str) -> str:
    arabizi = len(ARABIZI.findall(text))
    derja = len(re.findall(r"\b(?:el|fi|w|ken|ena|enti|tjib|mouch|wala|kima|houni|"
                           r"ba3d|9bal|bel|hedha|hedhi|lel)\b", text, re.I))
    if re.search(r"[؀-ۿ]", text) or arabizi >= 2 or arabizi + derja >= 3:
        return "ar"
    en = len(re.findall(r"\b(?:the|you|must|weeks|your|is)\b", text, re.I))
    fr = len(re.findall(r"\b(?:le|la|les|vous|doit|semaines|est|du)\b", text, re.I))
    return "en" if en > fr else "fr"


def main() -> int:
    if not core.GROQ_API_KEY:
        print("GROQ_API_KEY manquante dans .env")
        return 2
    failed = 0
    for case in CASES:
        answer = core._format_chat_display(core.Agent().ask(case["q"]))
        problems = []
        if answer_language(answer) != case["lang"]:
            problems.append(f"langue attendue {case['lang']}, obtenue {answer_language(answer)}")
        problems += [f"manque : {p}" for p in case["must"] if not re.search(p, answer)]
        problems += [f"interdit : {p}" for p in case["must_not"] if re.search(p, answer)]
        status = "OK  " if not problems else "ÉCHEC"
        failed += bool(problems)
        print(f"\n[{status}] {case['q']}\n        ({case['why']})")
        for p in problems:
            print("        -", p)
        if problems:
            print("        Réponse :", answer.replace("\n", " ")[:400])
    print(f"\n{len(CASES) - failed}/{len(CASES)} cas réussis.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
