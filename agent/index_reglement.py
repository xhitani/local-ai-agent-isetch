import re
import chromadb
from pypdf import PdfReader
from openai import OpenAI

PDF = "data/Reglement_etudes_ISET_Charguia_2026-2027.pdf"
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def embed(texts):
    r = client.embeddings.create(model="bge-m3", input=texts)
    return [d.embedding for d in r.data]

# 1. Lire le PDF
text = "\n".join(page.extract_text() for page in PdfReader(PDF).pages)
text = text.split("Calendrier des examens")[0]   # le calendrier a son propre outil

# 2. Découper par article : un chunk = un article
parts = re.split(r"(?=Article \d+\.)", text)
chunks = []
for p in parts:
    m = re.match(r"Article (\d+)\.", p)
    if m:
        chunks.append((int(m.group(1)), " ".join(p.split())))

print(f"{len(chunks)} articles trouvés (attendu : 15)")

# 3. Stocker dans ChromaDB
db = chromadb.PersistentClient(path="chroma_db")
try:
    db.delete_collection("reglement")
except Exception:
    pass
col = db.create_collection("reglement")
col.add(
    ids=[f"art{n}" for n, _ in chunks],
    documents=[t for _, t in chunks],
    metadatas=[{"article": n} for n, _ in chunks],
    embeddings=embed([t for _, t in chunks]),
)
print("Index créé dans chroma_db/")
