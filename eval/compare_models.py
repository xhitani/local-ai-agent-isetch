"""Fair and fast two-model evaluation for the ISET Charguia project.

Compares semantic understanding only: intent, language, and module.
Uses one Ollama call per case/model, JSON response mode, and a few-shot prompt.
"""
from pathlib import Path
import json, os, re, time
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    os.getenv("MODEL_A", "llama3.2:latest"),
    os.getenv("MODEL_B", "qwen2.5:0.5b"),
]
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

CASES = [
    {"q":"متى امتحان الإنجليزية؟","intent":"calendar_exam","module":"anglais","language":"ar"},
    {"q":"Quand est le rattrapage de Réseaux ?","intent":"calendar_resit","module":"réseaux","language":"fr"},
    {"q":"je suis malade pour BDD, je dois faire quoi ?","intent":"medical","module":"bases de données","language":"fr"},
    {"q":"كنت مريضًا يوم اختبار الإنجليزية، ماذا يمكنني أن أفعل؟","intent":"medical","module":"anglais","language":"ar"},
    {"q":"Je n'ai pas justifié mon absence à l'examen d'anglais.","intent":"absence","module":"anglais","language":"fr"},
    {"q":"Que dit l'Article 8 ?","intent":"regulation","module":None,"language":"fr"},
    {"q":"Combien de semaines minimum dure le stage ?","intent":"stage","module":None,"language":"fr"},
    {"q":"Quel est le délai pour une réclamation sur une note ?","intent":"grade_complaint","module":None,"language":"fr"},
    {"q":"أريد طلب استدراك لشبكات","intent":"request","module":"réseaux","language":"ar"},
    {"q":"hello","intent":"greeting","module":None,"language":"en"},
    {"q":"Donne-moi une recette de pizza.","intent":"off_topic","module":None,"language":"fr"},
    {"q":"ما هو استدراك الإنجليزية؟","intent":"calendar_resit","module":"anglais","language":"ar"},
]

SYSTEM = """You classify ISET Charguia student questions.
Return ONLY one JSON object with exactly these keys:
{"intent":"...", "module":"...", "language":"..."}

Allowed intent:
calendar_exam, calendar_resit, medical, absence, regulation,
stage, grade_complaint, request, greeting, off_topic

Allowed module:
anglais, réseaux, bases de données, algorithmique,
projet de fin de module, null

Allowed language: fr, ar, en

Rules:
- A question asking when/where the normal exam is => calendar_exam.
- A question asking about rattrapage/resit => calendar_resit.
- Illness, doctor, medical certificate, hospitalization => medical.
- Ordinary missed exam without medical reason => absence.
- Regulation/article question => regulation.
- Internship duration/rules => stage.
- Grade complaint deadline => grade_complaint.
- Asking to draft a resit request => request.
- Greeting => greeting.
- Clearly unrelated request => off_topic.
- Never invent a module.
- Arabic must be language ar.

Examples:
User: متى امتحان الإنجليزية؟
JSON: {"intent":"calendar_exam","module":"anglais","language":"ar"}

User: كنت مريضًا يوم اختبار الإنجليزية، ماذا يمكنني أن أفعل؟
JSON: {"intent":"medical","module":"anglais","language":"ar"}

User: Quand est le rattrapage de Réseaux ?
JSON: {"intent":"calendar_resit","module":"réseaux","language":"fr"}

User: hello
JSON: {"intent":"greeting","module":null,"language":"en"}
"""

def norm(x):
    return re.sub(r"[^a-z0-9]+", "", str(x or "").lower())

def module_ok(got, expected):
    if expected is None:
        return norm(got) in {"", "null", "none"}
    a, b = norm(got), norm(expected)
    aliases = {
        "anglais": {"anglais", "english"},
        "reseaux": {"reseaux", "network", "networks"},
        "basesdedonnees": {"basesdedonnees", "bdd", "database", "databases"},
        "algorithmique": {"algorithmique", "algorithm", "algo"},
        "projetdefindemodule": {"projetdefindemodule", "projet", "project"},
    }
    for key, vals in aliases.items():
        if b == key:
            return a in vals
    return a == b

def classify(client, model, question):
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system", "content":SYSTEM},
            {"role":"user", "content":question},
        ],
        temperature=0,
        max_tokens=80,
        response_format={"type":"json_object"},
    )
    raw = (r.choices[0].message.content or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}

def run(model):
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    rows = []
    start = time.perf_counter()

    for case in CASES:
        t = time.perf_counter()
        try:
            got = classify(client, model, case["q"])
            intent_ok = got.get("intent") == case["intent"]
            language_ok = got.get("language") == case["language"]
            module_match = module_ok(got.get("module"), case["module"])
            rows.append({
                "question": case["q"],
                "expected": case,
                "got": got,
                "intent_ok": intent_ok,
                "language_ok": language_ok,
                "module_ok": module_match,
                "success": bool(intent_ok and language_ok and module_match),
                "seconds": round(time.perf_counter() - t, 3),
            })
        except Exception as exc:
            rows.append({
                "question": case["q"],
                "expected": case,
                "got": {},
                "success": False,
                "seconds": round(time.perf_counter() - t, 3),
                "error": str(exc),
            })

    elapsed = time.perf_counter() - start
    n = len(rows)
    return {
        "model": model,
        "mode": "fair-semantic-v2",
        "total": n,
        "success": sum(x["success"] for x in rows),
        "intent_accuracy_percent": round(100 * sum(x["intent_ok"] for x in rows) / n, 2),
        "language_accuracy_percent": round(100 * sum(x["language_ok"] for x in rows) / n, 2),
        "module_accuracy_percent": round(100 * sum(x["module_ok"] for x in rows) / n, 2),
        "accuracy_percent": round(100 * sum(x["success"] for x in rows) / n, 2),
        "total_seconds": round(elapsed, 3),
        "average_seconds": round(elapsed / n, 3),
        "results": rows,
    }

def main():
    out = ROOT / "eval" / "model_comparison.json"
    report = {
        "models": [run(model) for model in MODELS],
        "note": "Fair semantic comparison. One LLM call per case/model; no production-agent routing or RAG is included.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = []
    for m in report["models"]:
        summary.append({
            k: m[k] for k in (
                "model", "total", "success", "accuracy_percent",
                "intent_accuracy_percent", "language_accuracy_percent",
                "module_accuracy_percent", "total_seconds", "average_seconds"
            )
        })
    print(json.dumps({"output": str(out), "models": summary}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

