import os
import re
import json
import difflib
import httpx
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from google import genai
from google.genai import types

app = FastAPI(title="Quranic Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALQURAN_BASE_URL = "https://api.alquran.cloud/v1"

# -----------------------------------------------------------------------------
# AI Client Credentials -- read ONLY from environment variables (set these in
# Vercel: Project -> Settings -> Environment Variables). Never hardcode keys
# here: this file is committed to a public/shared repo.
# -----------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

groq_client = Groq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Groq deprecated llama-3.3-70b-versatile on 2026-08-16.
GROQ_CHAT_MODEL = "openai/gpt-oss-120b"
GEMINI_CHAT_MODEL = "gemini-3.6-flash"

# -----------------------------------------------------------------------------
# In-Memory Cache (RAM)
# -----------------------------------------------------------------------------
surah_list_cache = None
ayah_cache = {}
quran_arabic_cache = None  # flat list for /recognize fuzzy matching


# -----------------------------------------------------------------------------
# 1. Quran Endpoints
# -----------------------------------------------------------------------------

@app.get("/surahs")
async def get_surahs():
    global surah_list_cache
    if surah_list_cache is not None:
        return surah_list_cache

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{ALQURAN_BASE_URL}/surah")
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch surahs")

        data = resp.json().get("data", [])
        surah_list_cache = [
            {
                "id": item["number"],
                "name_english": item["englishName"],
                "name_arabic": item["name"],
                "ayah_count": item["numberOfAyahs"],
                "revelation_type": item["revelationType"].lower(),
            }
            for item in data
        ]
        return surah_list_cache


@app.get("/surahs/{surah_id}/ayahs")
async def get_ayahs(surah_id: int):
    if surah_id < 1 or surah_id > 114:
        raise HTTPException(status_code=400, detail="Invalid Surah ID")

    if surah_id in ayah_cache:
        return ayah_cache[surah_id]

    url = f"{ALQURAN_BASE_URL}/surah/{surah_id}/editions/quran-uthmani,en.sahih,ur.jalandhry"
    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to load Surah ayahs")

        editions = resp.json().get("data", [])
        ar_ayahs = editions[0]["ayahs"]
        en_ayahs = editions[1]["ayahs"]
        ur_ayahs = editions[2]["ayahs"]

        out = []
        for i in range(len(ar_ayahs)):
            ayah_no = ar_ayahs[i]["numberInSurah"]
            out.append({
                "surah_id": surah_id,
                "ayah_number": ayah_no,
                "arabic": ar_ayahs[i]["text"],
                "english": en_ayahs[i]["text"],
                "urdu": ur_ayahs[i]["text"],
                "audio_url": f"https://everyayah.com/data/Alafasy_128kbps/{str(surah_id).zfill(3)}{str(ayah_no).zfill(3)}.mp3",
            })

        ayah_cache[surah_id] = out
        return out


@app.get("/search")
async def search(q: str = Query(..., min_length=1)):
    url = f"{ALQURAN_BASE_URL}/search/{q}/all/en.sahih"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return []

        matches = resp.json().get("data", {}).get("matches", [])
        return [
            {
                "surah_id": m["surah"]["number"],
                "ayah_number": m["numberInSurah"],
                "english": m["text"],
            }
            for m in matches[:20]
        ]


# -----------------------------------------------------------------------------
# 2. AI Chat Endpoint
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str | None = None
    text: str | None = None
    language: str = "en"


SYSTEM_PROMPT = """You are an Islamic scholar assistant. Answer concisely and accurately according to the Holy Quran.
You MUST reply strictly with compact JSON, all on one line, in this exact format:
{"text": "<2-3 sentence explanation in the requested language>", "citations": ["<surah>:<ayah>", ...]}
Do not include markdown, code fences, or any text outside the JSON object."""

CHAT_FALLBACK = {
    "en": "I couldn't generate an answer right now. Please try again in a moment.",
    "ur": "اس وقت جواب نہیں بن سکا۔ براہ کرم تھوڑی دیر بعد دوبارہ کوشش کریں۔",
    "ar": "تعذر إنشاء إجابة الآن. حاول مرة أخرى بعد قليل.",
}


def _parse_json_reply(raw: str) -> dict | None:
    """Recovers a usable {text, citations} dict even from truncated or
    markdown-wrapped model output, instead of letting json.loads() crash."""
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "{" in raw:
            raw = raw[raw.find("{"):]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(raw + '"}')
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(raw + '"]}')
    except json.JSONDecodeError:
        pass

    m = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
    if not m:
        return None
    text = m.group(1).encode().decode("unicode_escape", errors="ignore")
    refs = re.findall(r"\b(\d{1,3}:\d{1,3})\b", raw)
    return {"text": text, "citations": refs}


@app.post("/chat")
async def chat(req: ChatRequest):
    query_text = (req.message or req.text or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    prompt_content = f"Language: {req.language}\nQuestion: {query_text}"

    # 1. Primary: Groq
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=700,
            temperature=0.2,
        )
        parsed = _parse_json_reply(response.choices[0].message.content)
        if parsed and parsed.get("text"):
            return {"text": parsed["text"], "citations": parsed.get("citations", [])}
        print("[GROQ ERROR] Could not parse a usable reply from Groq output")
    except Exception as groq_err:
        print(f"\n[GROQ ERROR]: {groq_err}\n")

    # 2. Fallback: Gemini
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=700,
            ),
        )
        parsed = _parse_json_reply(response.text)
        if parsed and parsed.get("text"):
            return {"text": parsed["text"], "citations": parsed.get("citations", [])}
        print("[GEMINI ERROR] Could not parse a usable reply from Gemini output")
    except Exception as gemini_err:
        print(f"\n[GEMINI ERROR]: {gemini_err}\n")

    return {"text": CHAT_FALLBACK.get(req.language, CHAT_FALLBACK["en"]), "citations": []}


# -----------------------------------------------------------------------------
# 3. Audio: Speech-to-Text & Verse Recitation Recognition
# -----------------------------------------------------------------------------

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """Transcribes user voice input into text using Groq Whisper."""
    try:
        audio_bytes = await file.read()
        transcription = groq_client.audio.transcriptions.create(
            file=(file.filename or "audio.m4a", audio_bytes),
            model="whisper-large-v3",
            response_format="json",
        )
        return {"text": transcription.text.strip()}
    except Exception as e:
        print(f"\n[STT ERROR]: {e}\n")
        raise HTTPException(status_code=500, detail="speech_to_text_failed")


_DIACRITICS = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]"
)


def _normalize_arabic(text: str) -> str:
    """Diacritic-free, letter-unified Arabic used for verse matching. Also
    drops spaces, since Whisper's word-splitting rarely matches the
    Uthmani script's spacing exactly."""
    text = _DIACRITICS.sub("", text)
    for a, b in (("ٱ", "ا"), ("أ", "ا"), ("إ", "ا"), ("آ", "ا"),
                 ("ؤ", "و"), ("ئ", "ي"), ("ء", ""), ("ى", "ي"), ("ة", "ه")):
        text = text.replace(a, b)
    return re.sub(r"\s+", "", text)


async def _get_quran_arabic_flat():
    """Loads the whole Quran's Arabic text once (cached across warm serverless
    invocations, same pattern as surah_list_cache), for fuzzy-matching
    recitations against REAL verse text -- never guessed by an LLM."""
    global quran_arabic_cache
    if quran_arabic_cache is not None:
        return quran_arabic_cache

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{ALQURAN_BASE_URL}/quran/quran-uthmani")
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to load Quran text")
        surahs = resp.json().get("data", {}).get("surahs", [])

    flat = []
    for s in surahs:
        for a in s.get("ayahs", []):
            flat.append({
                "surah": s["number"],
                "ayah": a["numberInSurah"],
                "text": a["text"],
                "text_norm": _normalize_arabic(a["text"]),
            })
    if len(flat) < 6000:  # sanity check: the Quran has 6236 ayahs
        raise HTTPException(status_code=502, detail="Incomplete Quran text loaded")
    quran_arabic_cache = flat
    return flat


@app.post("/recognize")
async def recognize_recitation(file: UploadFile = File(...)):
    """Transcribes recited Quranic audio, then matches it against the REAL
    Quran text using string similarity -- not an LLM guess, so the surah,
    ayah, and confidence returned are never hallucinated."""
    try:
        audio_bytes = await file.read()
        transcription = groq_client.audio.transcriptions.create(
            file=(file.filename or "audio.m4a", audio_bytes),
            model="whisper-large-v3",
            language="ar",
            response_format="json",
        )
        transcript = transcription.text.strip()
    except Exception as e:
        print(f"\n[RECOGNIZE STT ERROR]: {e}\n")
        raise HTTPException(status_code=500, detail="speech_to_text_failed")

    if not transcript:
        raise HTTPException(status_code=400, detail="no_speech")

    query_norm = _normalize_arabic(transcript)
    if len(query_norm) < 3:
        raise HTTPException(status_code=400, detail="no_speech")

    try:
        verses = await _get_quran_arabic_flat()
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n[RECOGNIZE INDEX ERROR]: {e}\n")
        raise HTTPException(status_code=503, detail="index_unavailable")

    best = None
    best_score = 0.0
    for v in verses:
        score = difflib.SequenceMatcher(None, query_norm, v["text_norm"]).ratio()
        if score > best_score:
            best_score = score
            best = v

    if best is None or best_score < 0.45:
        raise HTTPException(status_code=404, detail="no_match")

    return {
        "transcript": transcript,
        "confidence": round(best_score, 3),  # real similarity, never guessed
        "ayah": {
            "surah_id": best["surah"],
            "ayah_number": best["ayah"],
            "arabic": best["text"],
        },
    }
