import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

CALENDRIER = {
    "algorithmique": {"examen": "05/01/2027, 9h, amphi A",
                      "rattrapage": "13/01/2027, 14h, salle B10"},
    "bases de données": {"examen": "08/01/2027, 9h, amphi A",
                         "rattrapage": "15/01/2027, 9h, salle B12"},
    "réseaux": {"examen": "11/01/2027, 9h, amphi B",
                "rattrapage": "18/01/2027, 9h, salle B14"},
    "anglais": {"examen": "12/01/2027, 14h, salle B11",
                "rattrapage": "19/01/2027, 10h, salle B11"},
    "projet de fin de module": {"examen": "Remise le 20/01/2027",
                                "rattrapage": "Pas de rattrapage"},
}

def get_calendrier(module: str) -> str:
    data = CALENDRIER.get(module.lower().strip())
    if data is None:
        return "Module introuvable dans le calendrier."
    return json.dumps(data, ensure_ascii=False)

TOOL_FUNCTIONS = {"get_calendrier": get_calendrier}

TOOLS = [{"type": "function", "function": {
    "name": "get_calendrier",
    "description": "Dates d'examen et de rattrapage d'un module (session de janvier 2027)",
    "parameters": {"type": "object",
                   "properties": {"module": {"type": "string"}},
                   "required": ["module"]}}}]

SYSTEM = (
    "Tu es l'assistant de scolarité de l'ISET Charguia. "
    "Tu réponds en français, uniquement à partir des outils fournis. "
    "Règles :\n"
    "1. Question sur les dates d'examen ou de rattrapage : utilise get_calendrier.\n"
    "2. Question qui n'a aucun rapport avec la scolarité (culture générale, "
    "personnes, météo...) : réponds exactement « Je ne peux répondre qu'aux "
    "questions de scolarité (absences, rattrapages, stages). » et n'appelle aucun outil.\n"
    "3. Question de scolarité dont la réponse n'est pas dans les outils : "
    "réponds « Je ne sais pas. »\n"
    "4. Ne devine jamais une date et n'invente jamais une information."
)

def run_tool(call) -> str:
    """LE PROGRAMME AGIT : exécution de l'outil demandé par le modèle."""
    name = call.function.name
    if name not in TOOL_FUNCTIONS:
        return f"Erreur : outil inconnu '{name}'."
    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        return "Erreur : arguments JSON invalides."
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:
        return f"Erreur dans l'outil : {e}"

def agent(question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for _ in range(5):  # 5 tours maximum
        # LE MODÈLE DÉCIDE : réponse finale ou demande d'outil
        msg = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS).choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            print(f"  [outil] {call.function.name}({call.function.arguments})")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": run_tool(call)})
    return "Je n'ai pas pu terminer."

if __name__ == "__main__":
    while True:
        q = input("\nQuestion (vide pour quitter) : ").strip()
        if not q:
            break
        print(agent(q))
