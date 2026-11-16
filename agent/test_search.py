import chromadb
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
col = chromadb.PersistentClient(path="chroma_db").get_collection("reglement")

def search_reglement(query, k=3):
    emb = client.embeddings.create(model="bge-m3", input=[query]).data[0].embedding
    r = col.query(query_embeddings=[emb], n_results=k)
    return [f"[Article {m['article']}] {d}"
            for d, m in zip(r["documents"][0], r["metadatas"][0])]

for q in ["Quel est le délai pour le certificat médical ?",
          "Que se passe-t-il si je rate le rattrapage ?",
          "Combien de semaines dure le stage ?"]:
    print("\n", q)
    for line in search_reglement(q):
        print("  ", line[:110])
