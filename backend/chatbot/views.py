from cProfile import label
from email import message
import json
import os
from pyexpat.errors import messages
import time
import re
from urllib import response  
import whisper
import subprocess
import threading
import queue
import tempfile
import requests
import math
from django.http import JsonResponse, StreamingHttpResponse, FileResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import render
from .models import Conversation

from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

def frontend(request):
    return render(request, "index.html")

#Conversation.objects.all().delete()
print("🧹 Conversations reset (mode test)")

# -----------------------------
#INIT
# -----------------------------
client = Groq(api_key=settings.GROQ_API_KEY)


# Qdrant local (SANS DOCKER)
qdrant = QdrantClient(path="qdrant_data")


# Embedding model
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

COLLECTION_NAME = "bovin_chunks"

whisper_model = None


# -----------------------------
# CHEMIN VERS PIPER + MODEL TTS
# -----------------------------
PIPER_PATH = r"C:\Users\zeine\Downloads\piper_windows_amd64\piper\piper.exe"
MODEL_FR = r"C:\Users\zeine\Downloads\piper_windows_amd64\piper\fr_FR-upmc-medium.onnx"
MODEL_AR = r"C:\Users\zeine\Downloads\piper_windows_amd64\piper\ar_JO-kareem-medium.onnx"


# -----------------------------
# 🌐 LANGUAGE CHECK
# -----------------------------
def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

# -----------------------------
# 🧹 CLEAN TEXT (AJOUT
# -----------------------------)
def clean_text(text, lang):
    text = text.strip()

    if lang == "fr":
        #garder lettres + accents + chiffres + espaces
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s']", "", text)

    elif lang == "ar":
        #garder arabe + chiffres + espaces
        text = re.sub(r"[^\u0600-\u06FF0-9\s]", "", text)
    return text

# -----------------------------
# 🧹 VALIDATE TEXT
# -----------------------------
def is_valid_text(text, lang):
    if not text or len(text.strip()) < 3:
        return False

    words = text.split()

    # trop peu de mots → suspect
    if len(words) < 2:
        return False

    if lang == "fr":
        # ratio mots "normaux"
        valid_words = sum(1 for w in words if re.match(r"^[a-zA-ZÀ-ÿ']+$", w))
        return valid_words / len(words) > 0.6

    elif lang == "ar":
        valid_words = sum(1 for w in words if re.match(r"^[\u0600-\u06FF]+$", w))
        return valid_words / len(words) > 0.6

    return True

# -----------------------------
# SPLIT
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
# LOAD CHUNKS
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
# EMBEDDING
# -----------------------------
def embed_text(text):
    return embedding_model.encode(text).tolist()




# -----------------------------
# INITIALISER QDRANT + INDEX
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




qdrant_initialized = False

# -----------------------------
#SEARCH QDRANT
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
# 📍 DISTANCE
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # rayon Terre en km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


# -----------------------------
# 🐄 VET AGEN
# -----------------------------
def vet_agent(lat, lon):
    url = "https://overpass-api.de/api/interpreter"
    radii = [5000, 10000, 20000]
    vets = []

    for radius in radii:
        query = f"""
        [out:json];
        node["amenity"="veterinary"](around:{radius},{lat},{lon});
        out;
        """
        try:
            res = requests.post(url, data=query)
            data = res.json()
        except:
            continue

        for el in data.get("elements", []):
            name = el.get("tags", {}).get("name", "Vétérinaire")
            phone = el.get("tags", {}).get("phone", None)
            vlat = el.get("lat")
            vlon = el.get("lon")
            distance = haversine(lat, lon, vlat, vlon)
            vets.append({"name": name, "phone": phone, "lat": vlat, "lon": vlon, "distance": round(distance, 2)})

        if vets:
            break

    if not vets:
        return "<p>❌ Aucun vétérinaire trouvé. Essayez Google Maps ou augmentez la zone de recherche.</p>"

    vets = sorted(vets, key=lambda x: x["distance"])
    best = vets[0]
    others = vets[1:3]

    def map_link(v):
        return f"https://www.google.com/maps/search/?api=1&query={v['lat']},{v['lon']}"

    warning = "<p>⚠️ Aucun vétérinaire très proche, voici les plus proches disponibles :</p>" if best["distance"] > 10 else ""
    phone_str = f"<p>📞 {best['phone']}</p>" if best["phone"] else "<p>📞 Téléphone non disponible</p>"
    
    others_html = ""
    if others:
        items = "".join(f"<li>{v['name']} — {v['distance']} km</li>" for v in others)
        others_html = f"<p><strong>🔹 Autres options :</strong></p><ul>{items}</ul>"

    return f"""{warning}
<div class="vet-card">
  <p>🐄 <strong>Vétérinaire recommandé</strong></p>
  <p>📍 <strong>{best['name']}</strong> — {best['distance']} km</p>
  {phone_str}
  <a href="{map_link(best)}" target="_blank" class="map-btn">📍 Voir sur Google Maps</a>
</div>
{others_html}"""

# -----------------------------
#LLM
# -----------------------------
def correct_stt_with_llm(raw_text, lang):
    if lang == "ar":
        prompt = f"""أنت مصحح للنصوص المنطوقة باللغة العربية.
النص المُفرَّغ (قد يحتوي أخطاء): "{raw_text}"

القواعد الصارمة:
- إذا كان النص يشبه العربية المنطوقة → صحح الإملاء والنحو، طبّع الحروف (أ إ آ)، وأعد النص المصحح فقط
- إذا كان النص غير مفهوم أو ضوضاء أو ليس عربياً → أعد بالضبط: INVALID
- لا تعيد شيئاً غير النص المصحح أو كلمة INVALID"""
    else:
        prompt = f"""Tu es un correcteur de transcription vocale française.
Texte transcrit (peut contenir des erreurs): "{raw_text}"

RÈGLES STRICTES:
- Si le texte ressemble à du français parlé → corrige orthographe et grammaire, retourne UNIQUEMENT le texte corrigé
- Si le texte est incompréhensible, du bruit, ou pas du français → retourne exactement: INVALID
- Ne retourne RIEN d'autre que le texte corrigé ou le mot INVALID
- Exemples: "la banne mange pa" → "La vache ne mange pas" | "xzqr ttt" → INVALID"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_completion_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"STT correction error: {e}")
        return raw_text  # fallback si Groq down
    
def decide_action(question, context, history, lang):
    system_prompt = """Tu es un classificateur strict. Une seule tâche : décider si l'utilisateur veut TROUVER UN VÉTÉRINAIRE PROCHE.

Réponds VET uniquement si l'utilisateur demande explicitement :
- un vétérinaire / vet / veterinaire / دكتور بيطري
- quelqu'un pour soigner son animal
- une clinique vétérinaire

Pour TOUT le reste (questions sur les vaches, maladies, alimentation, salutations, etc.) → réponds NORMAL

Réponds UNIQUEMENT par : VET ou NORMAL"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-2:])
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0,
            max_completion_tokens=5,
        )
        label = response.choices[0].message.content.strip().upper()
        return {"action": "vet_agent"} if "VET" in label else {"action": "answer"}
    except:
        return {"action": "answer"}
    
    
    
def ask_llm(question, context, history, session_id, lang):

    if lang == "ar":
        language_instruction = "Répond uniquement en arabe (العربية)"
    else:
        language_instruction = "Répond uniquement en français"

    if not settings.GROQ_API_KEY:
        yield "Erreur : clé API Groq manquante."
        return

    system_prompt = f"""{language_instruction}

Tu es un assistant IA.

Tu dois comprendre l'intention.

OBJECTIF :
Aider l’utilisateur avec des réponses utiles sur les vaches.

IMPORTANT:
- Ignore complètement la langue des messages précédents
- Répond UNIQUEMENT dans la langue demandée
- Ne te base PAS sur des mots exacts
- Comprends l'intention utilisateur

RÈGLES :

- Si la question est liée aux vaches → répondre normalement avec le contexte
- Si l'utilisateur dit au revoir (ex: "au revoir", "bye", "à bientôt") → répondre poliment et terminer la conversation (ne pas poser de nouvelle question)
- Les salutations comme "bonjour", "salut", "bjr", "bonjouuur" doivent toujours recevoir une réponse de salutation.
- Ne jamais répondre "au revoir" sauf si l'utilisateur dit explicitement qu'il part.
- Ne jamais dire "allez chercher sur Google".
- Si des coordonnées GPS sont présentes, considérer que la localisation est activée

- Si la question n'est pas liée aux vaches :
→ répondre :
FR:
"Je suis spécialisé en élevage bovin 🐄. Je ne peux pas répondre à cette question."
AR:
"أنا متخصص في تربية الأبقار 🐄 ولا يمكنني الإجابة على هذا السؤال."

- Ne jamais répondre à "au revoir" par une question
- Réponse courte et finale
- La langue ne doit jamais influencer la compréhension

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
    messages.extend(history)

    user_content = f"Réponds en {'arabe' if lang == 'ar' else 'français'} uniquement: {question}"
    messages.append({"role": "user", "content": user_content})

    full_response = ""

    try:
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3,
            max_completion_tokens=200,  # ← légèrement augmenté
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                yield token

    except Exception as e:
        yield f"\nErreur LLM: {str(e)}"

    finally:
        # ✅ Sauvegarde garantie même si le stream est interrompu
        if full_response.strip():
            Conversation.objects.create(
                session_id=session_id,
                role="assistant",
                message=full_response
            )


# -----------------------------
#API
# -----------------------------
@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    global qdrant_initialized

    if not qdrant_initialized:
        print("🔄 Initializing Qdrant...")
        init_qdrant()
        qdrant_initialized = True
    try:
        body = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    message = body.get("message")
    session_id = body.get("session_id", "default")
    lang = body.get("lang", "fr")
    lat = body.get("lat")
    lon = body.get("lon")

    if not isinstance(message, str):
        return JsonResponse({"error": "Invalid message"}, status=400)

    # -----------------------------
    #LANGUAGE CONTROL (AJOUT)
    # -----------------------------
    if lang == "fr" and is_arabic(message):
        return StreamingHttpResponse(
            iter(["Veuillez écrire en français 🇫🇷 ou changez la langue"]),
            content_type="text/plain"
)

    if lang == "ar" and not is_arabic(message):
        return StreamingHttpResponse(
            iter(["يرجى الكتابة باللغة العربية 🇸🇦 أو تغيير اللغة"]),
            content_type="text/plain"
)     

    Conversation.objects.create(
        session_id=session_id,
        role="user",
        message=message
)
    # -----------------------------
    #MEMORY
    # -----------------------------
    history_db = Conversation.objects.filter(
        session_id=session_id
    ).order_by('-created_at')[:10]

    history = []
    for msg in reversed(history_db):
        history.append({
            "role": msg.role,
            "content": msg.message
        })
        
    # -----------------------------
    #RAG
    # -----------------------------
    results = search_qdrant(message)
    context = "\n\n".join(results) if results else ""

    # -----------------------------
    # DECISION LLM (PROPRE)
    # -----------------------------
    decision = decide_action(message, context, history, lang)

    # -----------------------------
    # TOOL EXECUTION
    # -----------------------------
    if decision.get("action") == "vet_agent":
        if lat is not None and lon is not None:
            result = vet_agent(float(lat), float(lon))
        else:
            result = "📍 Veuillez activer la localisation pour trouver un vétérinaire proche"

        return StreamingHttpResponse(iter([result]), content_type="text/plain")
    
    # -----------------------------
    # 🤖 LLM NORMAL
    # -----------------------------
    return StreamingHttpResponse(
        ask_llm(message, context, history, session_id, lang),
        content_type="text/plain"
    )



# -----------------------------
# STT
# -----------------------------
@csrf_exempt
@require_http_methods(["POST"])
def stt(request):
    global whisper_model

    if whisper_model is None:
        print("🔄 Loading Whisper model...")
        whisper_model = whisper.load_model("base")
    lang = request.POST.get("lang", "fr")
    if lang not in ["fr", "ar"]:
        lang = "fr"
    try:
        audio_file = request.FILES.get("audio")
        if not audio_file:
            return JsonResponse({"error": "No audio file"}, status=400)

        # ✅ chemin compatible Windows/Linux/Mac
        tmp_dir = tempfile.gettempdir()
        input_path = os.path.join(tmp_dir, "input.webm")
        output_path = os.path.join(tmp_dir, "output.wav")

        with open(input_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print("FILE SIZE:", os.path.getsize(output_path))

        result_queue = queue.Queue()

        def transcribe():
            try:
                if lang == "ar":
                    result = whisper_model.transcribe(output_path, language="ar")
                else:
                    result = whisper_model.transcribe(output_path, language="fr")
                result_queue.put(result["text"])
            except Exception as e:
                result_queue.put("")

        t = threading.Thread(target=transcribe)
        t.start()
        t.join(timeout=30)
        
        if not result_queue.empty():
            text = result_queue.get().strip()
            text = clean_text(text, lang)

            if not is_valid_text(text, lang):
                return JsonResponse({"text": "", "status": "incomprehensible"})

            # 🔥 Correction LLM
            corrected = correct_stt_with_llm(text, lang)
            print(f"STT brut: {text} → corrigé: {corrected}")

            if corrected.strip().upper().replace(".", "").replace("!", "") == "INVALID":
                return JsonResponse({"text": "", "status": "incomprehensible"})

            return JsonResponse({"text": corrected, "status": "ok"})
        else:
            return JsonResponse({"error": "Timeout"}, status=504)
        

    except Exception as e:
        print("ERROR:", e)
        return JsonResponse({"error": str(e)}, status=500)
    

# -----------------------------
# TTS
# -----------------------------
def generate_tts(text, lang):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = tmp_file.name

    # 🔥 choix du modèle
    if lang == "ar":
        model_path = MODEL_AR
    else:
        model_path = MODEL_FR

    process = subprocess.Popen(
        [PIPER_PATH, "--model", model_path, "--output_file", output_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    process.communicate(input=text.encode("utf-8"))

    return output_path


@csrf_exempt
@require_http_methods(["POST"])
def tts(request):
    body = json.loads(request.body.decode("utf-8"))
    text = body.get("text", "")
    lang = body.get("lang", "fr")  

    if not text:
        return JsonResponse({"error": "No text"}, status=400)

    audio_path = generate_tts(text, lang)

    response = FileResponse(open(audio_path, "rb"), content_type="audio/wav")
    response["Content-Disposition"] = "inline; filename=tts.wav"

    def cleanup(file_path):
        try:
            os.remove(file_path)
        except:
            pass

    threading.Thread(target=cleanup, args=(audio_path,)).start()

    return response

