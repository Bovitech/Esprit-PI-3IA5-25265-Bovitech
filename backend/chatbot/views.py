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
# INIT
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
# 🧹 CLEAN TEXT
# -----------------------------
def clean_text(text, lang):
    text = text.strip()
    if lang == "fr":
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s']", "", text)
    elif lang == "ar":
        text = re.sub(r"[^\u0600-\u06FF0-9\s]", "", text)
    return text

# -----------------------------
# 🧹 VALIDATE TEXT
# -----------------------------
def is_valid_text(text, lang):
    if not text or len(text.strip()) < 3:
        return False
    words = text.split()
    if len(words) < 2:
        return False
    if lang == "fr":
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
    if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        chunks = load_all_chunks()
        points = []
        for i, chunk in enumerate(chunks):
            vector = embed_text(chunk)
            points.append(PointStruct(id=i, vector=vector, payload={"text": chunk}))
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Qdrant indexé avec {len(points)} chunks")

qdrant_initialized = False

# -----------------------------
# SEARCH QDRANT
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
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


# -----------------------------
# 🐄 VET AGENT
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
# 🌦️ METEO AGENT (time-aware)
# -----------------------------
def meteo_agent(lat, lon, lang):
    from datetime import datetime, timezone, timedelta

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,precipitation,windspeed_10m,weathercode,is_day"
            f"&hourly=temperature_2m,precipitation_probability"
            f"&forecast_days=1"
            f"&windspeed_unit=kmh&timezone=auto"
        )
        res = requests.get(url)
        data = res.json()
        current = data["current"]

        temp   = current["temperature_2m"]
        rain   = current["precipitation"]
        wind   = current["windspeed_10m"]
        wcode  = current["weathercode"]
        is_day = current.get("is_day", 1)

        hourly = data.get("hourly", {})
        prec_prob_list = hourly.get("precipitation_probability", [])
        next_3h_rain   = max(prec_prob_list[:3]) if prec_prob_list else 0

    except Exception as e:
        msg = "تعذّر الحصول على بيانات الطقس." if lang == "ar" else "Impossible de récupérer la météo."
        return msg

    time_context = "nuit" if not is_day else "jour"

    context = f"""
Heure : {time_context}
Température : {temp}°C
Précipitations actuelles : {rain} mm
Probabilité de pluie dans les 3 prochaines heures : {next_3h_rain}%
Vent : {wind} km/h
Code météo WMO : {wcode}
"""

    if lang == "ar":
        prompt = f"""أنت خبير في تربية الأبقار. بناءً على بيانات الطقس التالية، قرر إذا كان يمكن إخراج الأبقار.

{context}

القواعد:
- إذا كان الوقت ليلاً → لا تُخرج الأبقار أبداً (خطر وبرودة)
- درجة حرارة > 32°C → إجهاد حراري، خروج محدود صباحاً أو مساءً فقط
- درجة حرارة < 0°C → برد شديد، ابقِ في الداخل
- أمطار > 2mm أو احتمال مطر > 60% → أرض موحلة، خطر إصابة
- رياح > 50 km/h → خطر ذعر
- وإلا → الظروف مواتية

أجب فقط بـ JSON هذا بدون أي شيء آخر:
{{"decision": "out", "reason": "...", "tip": "..."}}
الـ decision يجب أن يكون: "out" أو "in" أو "limited"
"""
    else:
        prompt = f"""Tu es un expert en élevage bovin. Décide si les vaches peuvent sortir.

{context}

Règles :
- Nuit → ne jamais sortir les vaches (danger, froid, déstabilisation)
- Température > 32°C → stress thermique, sortie limitée matin/soir
- Température < 0°C → froid extrême, garder à l'intérieur
- Précipitations > 2mm OU probabilité pluie > 60% → sol boueux, blessure
- Vent > 50 km/h → panique et stress
- Sinon → conditions favorables

Réponds UNIQUEMENT avec ce JSON :
{{"decision": "out", "reason": "...", "tip": "..."}}
"decision" doit être : "out", "in" ou "limited"
"""

    try:
        llm_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=150,
        )
        raw = llm_res.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        parsed   = json.loads(raw)
        decision = parsed.get("decision", "out")
        reason   = parsed.get("reason", "")
        tip      = parsed.get("tip", "")
    except:
        decision, reason, tip = "out", "Données insuffisantes", "Surveiller les conditions"

    # Night override — safety net regardless of LLM
    if not is_day:
        decision = "in"
        if lang == "ar":
            reason = "الوقت ليلاً — الأبقار يجب أن تكون في الداخل"
            tip    = "أعد التحقق صباحاً"
        else:
            reason = "Il fait nuit — les vaches doivent rester à l'intérieur"
            tip    = "Revérifiez le matin"

    # Build card
    if decision == "out":
        header  = ("🌤️ الطقس مناسب",       "🌤️ Conditions favorables")[lang != "ar"]
        verdict = ("✅ يمكن إخراج الأبقار", "✅ Les vaches peuvent sortir")[lang != "ar"]
        color   = "#2d6a4f"
    elif decision == "in":
        if not is_day:
            header  = ("🌙 وقت الليل",              "🌙 C'est la nuit")[lang != "ar"]
            verdict = ("❌ الأبقار تبقى في الداخل", "❌ Les vaches restent à l'intérieur")[lang != "ar"]
        else:
            header  = ("🌧️ الطقس غير مناسب",        "🌧️ Conditions défavorables")[lang != "ar"]
            verdict = ("❌ أبقِ الأبقار في الداخل", "❌ Gardez les vaches à l'intérieur")[lang != "ar"]
        color = "#c0392b"
    else:
        header  = ("⚠️ خروج محدود",                  "⚠️ Sortie limitée")[lang != "ar"]
        verdict = ("⚠️ الخروج صباحاً أو مساءً فقط", "⚠️ Sortir tôt matin / soir uniquement")[lang != "ar"]
        color   = "#e67e22"

    stats = f"🌡️ {temp}°C &nbsp;|&nbsp; 🌧️ {rain}mm &nbsp;|&nbsp; 💨 {wind} km/h &nbsp;|&nbsp; {'🌙 Nuit' if not is_day else '☀️ Jour'}"

    rain_warning = ""
    if next_3h_rain > 60:
        rain_warning = f"<p class='meteo-rain-warn'>🌂 {'احتمال مطر خلال 3 ساعات' if lang == 'ar' else 'Pluie probable dans 3h'} ({next_3h_rain}%)</p>"

    return f"""<div class="meteo-card" style="border-color:{color}">
  <p class="meteo-header">{header}</p>
  <p class="meteo-verdict" style="color:{color}">{verdict}</p>
  <p class="meteo-stats">{stats}</p>
  {rain_warning}
  <p class="meteo-reason">• {reason}</p>
  <p class="meteo-tip">👉 {tip}</p>
</div>"""


# -----------------------------
# 🌱 FEED AGENT
# -----------------------------
def feed_agent(lat, lon, lang):
    from datetime import datetime

    month = datetime.now().month
    if month in [12, 1, 2]:
        season = "hiver"
    elif month in [3, 4, 5]:
        season = "printemps"
    elif month in [6, 7, 8]:
        season = "été"
    else:
        season = "automne"

    # ✅ None guard — feed agent works even without GPS
    try:
        if lat is not None and lon is not None:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,precipitation,is_day"
                f"&timezone=auto"
            )
            res    = requests.get(url)
            data   = res.json()
            temp   = data["current"]["temperature_2m"]
            rain   = data["current"]["precipitation"]
            is_day = data["current"].get("is_day", 1)
        else:
            temp, rain, is_day = 20, 0, 1  # neutral defaults without GPS
    except:
        temp, rain, is_day = 20, 0, 1

    time_of_day = "matin" if is_day else "soir/nuit"

    if lang == "ar":
        prompt = f"""أنت خبير تغذية أبقار. بناءً على البيانات التالية، أعطِ توصيات تغذية اليوم.

الفصل: {season}
الوقت: {time_of_day}
درجة الحرارة: {temp}°C
الأمطار: {rain}mm

اعتبر:
- في الصيف وعند الحرارة > 28°C → زد الماء، قلل الحبوب، أضف الأملاح
- في الشتاء والبرد < 5°C → زد الطاقة (الشعير/الذرة)، أضف دهون
- في الربيع → مراعاة الانتقال لمرعى جديد (خطر انتفاخ)
- في الخريف → زد الاحتياطي قبل الشتاء

أجب فقط بـ JSON هذا:
{{"main_feed": "...", "supplement": "...", "water": "...", "warning": "...", "tip": "..."}}
"""
    else:
        prompt = f"""Tu es un expert en nutrition bovine. Donne des recommandations d'alimentation pour aujourd'hui.

Saison : {season}
Moment : {time_of_day}
Température : {temp}°C
Pluie : {rain}mm

Considère :
- Été / chaleur > 28°C → augmenter eau, réduire concentrés, ajouter sel minéral
- Hiver / froid < 5°C → augmenter énergie (orge/maïs), supplément lipidique
- Printemps → attention transition pâturage (risque météorisation)
- Automne → constituer réserves corporelles avant hiver

Réponds UNIQUEMENT avec ce JSON :
{{"main_feed": "...", "supplement": "...", "water": "...", "warning": "...", "tip": "..."}}
"""

    try:
        llm_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=200,
        )
        raw    = llm_res.choices[0].message.content.strip()
        raw    = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(raw)

        main_feed  = parsed.get("main_feed",  "Foin à volonté")
        supplement = parsed.get("supplement", "Minéraux standards")
        water      = parsed.get("water",      "Eau fraîche en permanence")
        warning    = parsed.get("warning",    "")
        tip        = parsed.get("tip",        "")
    except:
        main_feed, supplement = "Foin à volonté", "Minéraux standards"
        water, warning, tip   = "Eau fraîche", "", ""

    season_emoji = {"hiver": "❄️", "printemps": "🌱", "été": "☀️", "automne": "🍂"}.get(season, "🌿")
    warning_html = f'<p class="feed-warning">⚠️ {warning}</p>' if warning else ""

    if lang == "ar":
        labels = {
            "title":       f"{season_emoji} توصيات التغذية اليوم",
            "main_label":  "🌾 العلف الأساسي",
            "supp_label":  "💊 المكملات",
            "water_label": "💧 الماء",
            "tip_label":   "👉 نصيحة"
        }
    else:
        labels = {
            "title":       f"{season_emoji} Alimentation recommandée aujourd'hui",
            "main_label":  "🌾 Fourrage principal",
            "supp_label":  "💊 Compléments",
            "water_label": "💧 Eau",
            "tip_label":   "👉 Conseil"
        }

    return f"""<div class="feed-card">
  <p class="feed-title">{labels['title']}</p>
  <p class="feed-stats">🌡️ {temp}°C &nbsp;|&nbsp; {season_emoji} {season.capitalize()} &nbsp;|&nbsp; 🕐 {time_of_day}</p>
  <div class="feed-row"><span class="feed-label">{labels['main_label']}</span><span>{main_feed}</span></div>
  <div class="feed-row"><span class="feed-label">{labels['supp_label']}</span><span>{supplement}</span></div>
  <div class="feed-row"><span class="feed-label">{labels['water_label']}</span><span>{water}</span></div>
  {warning_html}
  <p class="feed-tip">{labels['tip_label']} {tip}</p>
</div>"""


# -----------------------------
# LLM — STT CORRECTION
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
        return raw_text


# -----------------------------
# DECISION CLASSIFIER
# -----------------------------
def decide_action(question, context, history, lang):
    system_prompt = """Tu es un classificateur strict. Classe dans UNE catégorie :

VET    → trouver un vétérinaire / clinique proche
METEO  → météo, sortir les vaches dehors, conditions extérieures
FEED   → alimentation, que donner à manger, ration, fourrage, eau, nutrition
NORMAL → tout le reste

Réponds UNIQUEMENT par : VET, METEO, FEED ou NORMAL"""

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
        if "VET"   in label: return {"action": "vet_agent"}
        if "METEO" in label: return {"action": "meteo_agent"}
        if "FEED"  in label: return {"action": "feed_agent"}
        return {"action": "answer"}
    except:
        return {"action": "answer"}


# -----------------------------
# 🧠 CONVERSATION SUMMARIZER
# -----------------------------
SUMMARY_THRESHOLD = 12
MESSAGES_TO_KEEP  = 4

def summarize_conversation(session_id, messages_to_summarize, lang):
    conversation_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages_to_summarize
    )

    if lang == "ar":
        prompt = f"""لخّص هذا الحوار بإيجاز شديد (3-5 جمل) مع الحفاظ على المعلومات المهمة عن الأبقار والمشاكل المذكورة:

{conversation_text}

الملخص:"""
    else:
        prompt = f"""Résume cette conversation très brièvement (3-5 phrases) en conservant les informations importantes sur les vaches et problèmes mentionnés :

{conversation_text}

Résumé :"""

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=200,
        )
        summary_text = res.choices[0].message.content.strip()
    except:
        summary_text = "Résumé non disponible."

    from .models import ConversationSummary
    ConversationSummary.objects.create(session_id=session_id, summary=summary_text)

    old_ids = [m["id"] for m in messages_to_summarize if "id" in m]
    if old_ids:
        Conversation.objects.filter(id__in=old_ids).delete()

    return summary_text


def get_history_with_summary(session_id, lang):
    from .models import ConversationSummary

    all_messages = list(
        Conversation.objects.filter(session_id=session_id)
        .order_by('created_at')
        .values('id', 'role', 'message', 'created_at')
    )

    history = []

    if len(all_messages) > SUMMARY_THRESHOLD:
        old_msgs = all_messages[:-MESSAGES_TO_KEEP]

        old_formatted = [
            {"id": m["id"], "role": m["role"], "content": m["message"]}
            for m in old_msgs
        ]

        summarize_conversation(session_id, old_formatted, lang)

        all_messages = list(
            Conversation.objects.filter(session_id=session_id)
            .order_by('created_at')
            .values('id', 'role', 'message')
        )

    latest_summary = ConversationSummary.objects.filter(session_id=session_id).first()

    if latest_summary:
        summary_msg = (
            f"ملخص المحادثة السابقة: {latest_summary.summary}"
            if lang == "ar"
            else f"Résumé de la conversation précédente : {latest_summary.summary}"
        )
        history.append({"role": "user",      "content": summary_msg})
        history.append({"role": "assistant", "content": "Compris, je garde ce contexte en mémoire." if lang == "fr" else "حسناً، سأتذكر هذا السياق."})

    for m in all_messages:
        history.append({"role": m["role"], "content": m["message"]})

    return history


# -----------------------------
# LLM — MAIN
# -----------------------------
def ask_llm(question, context, history, session_id, lang):

    if lang == "ar":
        language_instruction = "Répond uniquement en arabe (العربية)"
    else:
        language_instruction = "Répond uniquement en français"

    if not settings.GROQ_API_KEY:
        yield "Erreur : clé API Groq manquante."
        return

    system_prompt = f"""{language_instruction}

Tu es un assistant IA spécialisé en élevage bovin.

RÈGLES :
- Si la question concerne les vaches → réponds avec le contexte fourni
- Si l'utilisateur salue → réponds poliment
- Si l'utilisateur dit au revoir → réponds brièvement sans poser de question
- Si la question n'a rien à voir avec les vaches → réponds :
  FR: "Je suis spécialisé en élevage bovin 🐄. Je ne peux pas répondre à cette question."
  AR: "أنا متخصص في تربية الأبقار 🐄 ولا يمكنني الإجابة على هذا السؤال."
- Si le contexte est vide → utilise tes connaissances générales sur les vaches
- Ne jamais inventer d'informations
- Réponse concise (max 4 lignes), ton professionnel
- Si situation grave → recommander un vétérinaire
- Utiliser des points (•) si nécessaire
- IMPORTANT : réponds TOUJOURS dans la langue demandée ({lang}), ignore la langue des messages précédents

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
            max_completion_tokens=200,
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
        if full_response.strip():
            Conversation.objects.create(
                session_id=session_id,
                role="assistant",
                message=full_response
            )


# -----------------------------
# API — CHAT
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

    message    = body.get("message")
    session_id = body.get("session_id", "default")
    lang       = body.get("lang", "fr")
    lat        = body.get("lat")
    lon        = body.get("lon")

    if not isinstance(message, str):
        return JsonResponse({"error": "Invalid message"}, status=400)

    # Language control
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

    Conversation.objects.create(session_id=session_id, role="user", message=message)

    # Memory with summarization
    history = get_history_with_summary(session_id, lang)

    # RAG
    results = search_qdrant(message)
    context = "\n\n".join(results) if results else ""

    # Decision
    decision = decide_action(message, context, history, lang)

    # Tool routing
    if decision.get("action") == "vet_agent":
        if lat is not None and lon is not None:
            result = vet_agent(float(lat), float(lon))
        else:
            result = "📍 Veuillez activer la localisation pour trouver un vétérinaire proche"
        return StreamingHttpResponse(iter([result]), content_type="text/plain")

    if decision.get("action") == "meteo_agent":
        if lat is not None and lon is not None:
            result = meteo_agent(float(lat), float(lon), lang)
        else:
            result = "📍 يرجى تفعيل الموقع الجغرافي للحصول على توقعات الطقس" if lang == "ar" else "📍 Veuillez activer la localisation pour obtenir la météo"
        return StreamingHttpResponse(iter([result]), content_type="text/plain")

    if decision.get("action") == "feed_agent":
        # ✅ feed_agent handles None lat/lon gracefully
        result = feed_agent(
            float(lat) if lat is not None else None,
            float(lon) if lon is not None else None,
            lang
        )
        return StreamingHttpResponse(iter([result]), content_type="text/plain")

    # Normal LLM
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

        tmp_dir     = tempfile.gettempdir()
        input_path  = os.path.join(tmp_dir, "input.webm")
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
    tmp_file    = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = tmp_file.name

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
    response   = FileResponse(open(audio_path, "rb"), content_type="audio/wav")
    response["Content-Disposition"] = "inline; filename=tts.wav"

    def cleanup(file_path):
        try:
            os.remove(file_path)
        except:
            pass

    threading.Thread(target=cleanup, args=(audio_path,)).start()
    return response