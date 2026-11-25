# ISET Charguia — Assistant de scolarité IA

Assistant agentique de scolarité pour l'ISET Charguia, développé dans le cadre d'un projet académique. Il répond à partir du règlement des études fourni (PDF) et ne doit rien inventer en dehors de ce document.

## Fonctionnalités

- **Assistant conversationnel** (français, arabe tunisien, anglais) sur le règlement, les absences, les certificats médicaux, les examens, les rattrapages et les stages, avec références aux articles.
- **Calendrier et compte à rebours** : prochain examen (J-n), date de chaque épreuve et date limite du justificatif calculée selon l'article 4 (3 jours ouvrables) ou l'article 6 (7 jours en cas d'hospitalisation).
- **Formulaire de demande de rattrapage** (article 8) : aperçu de la lettre, téléchargement, et envoi par e-mail uniquement après confirmation explicite.
- **Profil étudiant** (nom, numéro, e-mail en copie) pour le message d'accueil et le pré-remplissage. Conservé pendant la session seulement.
- **Interface FR / عربي / EN**, avec mise en page de droite à gauche en arabe.
- **Serveur MCP** exposant `search_reglement`, `get_calendrier` et `generate_demande`. Aucun outil d'envoi n'est exposé au modèle : l'envoi reste une décision humaine.

Toutes les informations académiques (dates, salles, articles) sont lues dans `data/Reglement_etudes_ISET_Charguia_2026-2027.pdf`. Rien n'est codé en dur dans l'interface.

## Architecture

```
app.py                     point d'entrée Streamlit
agent/scolarite_agent.py   agent : lecture du PDF, outils, appel du modèle, e-mail, MCP, terminal
agent/ui.py                interface Streamlit (traductions, calendrier, formulaire)
agent/test_groq.py         vérification rapide de la clé Groq
data/                      règlement des études (PDF)
assets/                    logo
.streamlit/config.toml     thème clair forcé
legacy/                    anciennes versions (Ollama + ChromaDB), conservées pour référence
```

Le modèle est appelé via l'API Groq (compatible OpenAI), par défaut `openai/gpt-oss-20b`. La recherche dans le règlement est faite localement par mots-clés sur les pages du PDF : pas d'embeddings ni de base vectorielle.

## Installation

Python 3.10 ou plus récent.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # puis renseigner GROQ_API_KEY
```

L'envoi d'e-mails est facultatif. Avec Gmail, utilisez un [mot de passe d'application](https://myaccount.google.com/apppasswords) dans `SMTP_PASSWORD` et indiquez l'adresse de la scolarité dans `MAIL_TO`. Sans ces valeurs, les demandes sont préparées et téléchargeables, mais pas envoyées.

## Lancement

```bash
streamlit run app.py                         # interface web (http://localhost:8501)
python agent/scolarite_agent.py --terminal   # mode terminal
python agent/scolarite_agent.py --mcp        # serveur MCP (stdio)
python agent/test_groq.py                    # tester la clé API
```

Pour ouvrir l'interface depuis un autre appareil du réseau local : `streamlit run app.py --server.address 0.0.0.0`.

## Limites connues

- Les conversations sont enregistrées dans `saved_chats/` sur le serveur et visibles par toute personne qui ouvre l'application. À réserver à un usage local ou de démonstration.
- Le règlement et le calendrier sont fictifs, à usage pédagogique.
- L'évaluation dans `legacy/eval/` porte sur les anciens modèles locaux (Ollama), pas sur le modèle Groq actuel.
