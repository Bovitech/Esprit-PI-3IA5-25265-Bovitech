import json
import os
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods


from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


# -----------------------------
# 🔥 INIT
# -----------------------------
client = Groq(api_key=settings.GROQ_API_KEY)


# Qdrant local (SANS DOCKER)
qdrant = QdrantClient(path="qdrant_data")


# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


COLLECTION_NAME = "bovin_chunks"


# -----------------------------
# 🧠 Mémoire conversations
# -----------------------------
CONVERSATIONS = {}
MAX_HISTORY = 6  # garder derniers échanges


# -----------------------------
# 📚 SPLIT
# -----------------------------
def split_text_into_chunks(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0


    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()


        if chunk:
            chunks.append(chunk)


        start += chunk_size - overlap


    return chunks




# -----------------------------
# 📚 LOAD CHUNKS
# -----------------------------
def load_all_chunks():
    knowledge_dir = os.path.join(settings.BASE_DIR, "knowledge_base")
    all_chunks = []


    if not os.path.exists(knowledge_dir):
        return []


    for filename in os.listdir(knowledge_dir):
        file_path = os.path.join(knowledge_dir, filename)


        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                all_chunks.extend(split_text_into_chunks(content))


    return all_chunks




# -----------------------------
# 🔥 EMBEDDING
# -----------------------------
def embed_text(text):
    return embedding_model.encode(text).tolist()




# -----------------------------
# 🚀 INITIALISER QDRANT + INDEX
# -----------------------------
def init_qdrant():


    # créer collection si n'existe pas
    if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:


        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,  # taille embedding MiniLM
                distance=Distance.COSINE
            )
        )


        chunks = load_all_chunks()


        points = []
        for i, chunk in enumerate(chunks):
            vector = embed_text(chunk)


            points.append(
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={"text": chunk}
                )
            )


        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )


        print(f"✅ Qdrant indexé avec {len(points)} chunks")




# 🔥 lancer au démarrage
init_qdrant()




# -----------------------------
# 🔍 SEARCH QDRANT
# -----------------------------
def search_qdrant(query, top_k=3):
    query_vector = embed_text(query)


    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )


    filtered = []


    if hasattr(results, "points"):
        for point in results.points:
            if point.score > 0.5 and point.payload:
                filtered.append(point.payload.get("text", ""))


    return filtered


# -----------------------------
# 🤖 LLM
# -----------------------------
def ask_llm(question, context, history):


    if not settings.GROQ_API_KEY:
        return "Erreur : clé API Groq manquante."


    system_prompt = f"""
Tu es un assistant spécialisé en élevage bovin.


OBJECTIF :
Aider l’utilisateur avec des réponses utiles sur les vaches.


RÈGLES :


- Si la question est liée aux vaches → répondre normalement avec le contexte
- Si l'utilisateur dit au revoir (ex: "au revoir", "bye", "à bientôt") → répondre poliment et terminer la conversation (ne pas poser de nouvelle question)


- Si la question est hors sujet (pas liée aux vaches) :
→ répondre :
"Je suis spécialisé en élevage bovin 🐄. Je ne peux pas répondre à cette question."
- Ne jamais répondre à "au revoir" par une question
- Réponse courte et finale

- Si le CONTEXTE est vide :
→ répondre en utilisant tes connaissances générales sur les vaches
→ rester prudent et ne pas inventer
- Ne jamais inventer d'information
- Ne pas utiliser le contexte s'il ne correspond pas à la question
- Réponse très concise (max 3 lignes)
- Aller directement à l’essentiel
- Utiliser un ton neutre et professionnel
- Si la situation semble sérieuse → recommander de consulter un vétérinaire
- Utiliser des points (•) si nécessaire


CONTEXTE :
{context}
"""


    messages = [{"role": "system", "content": system_prompt}]


    # 🔥 ajouter historique
    messages.extend(history)


    # ajouter question actuelle
    messages.append({"role": "user", "content": question})


    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3,
            max_completion_tokens=150
        )


        return response.choices[0].message.content


    except Exception as e:
        return f"Erreur LLM: {str(e)}"




# -----------------------------
# 🧠 GENERATE RESPONSE
# -----------------------------
def generate_response(message, session_id="default"):


    message_lower = message.lower().strip()


    if len(message_lower) < 2:
        return "Pouvez-vous préciser votre question concernant les vaches ?"


    # récupérer historique
    if session_id not in CONVERSATIONS:
        CONVERSATIONS[session_id] = []


    history = CONVERSATIONS[session_id]


    # 🔍 recherche RAG
    results = search_qdrant(message)


    if not results:
        context = ""
    else:
        context = "\n\n".join(results)

    reply = ask_llm(message, context, history)

    # 🧠 sauvegarder conversation
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})


    # limiter mémoire
    CONVERSATIONS[session_id] = history[-MAX_HISTORY:]


    return reply




# -----------------------------
# 🔥 API
# -----------------------------
@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    try:
        body = json.loads(request.body.decode("utf-8"))
    except:
        return JsonResponse({"error": "Invalid JSON"}, status=400)


    message = body.get("message")
    session_id = body.get("session_id", "default")  # 🔥 important


    if not isinstance(message, str):
        return JsonResponse({"error": "Invalid message"}, status=400)


    reply = generate_response(message, session_id)


    return JsonResponse({"response": reply})

