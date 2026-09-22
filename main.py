import os
import re
import json
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from google import genai
from google.genai import types

app = FastAPI(title="Quranic Chatbot Backend")

# Allow requests from mobile emulators and local devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALQURAN_BASE_URL = "https://api.alquran.cloud/v1"

# -----------------------------------------------------------------------------
# AI Client Credentials
# -----------------------------------------------------------------------------
# IMPORTANT: your old keys were hardcoded here and were shared in a chat log.
# Regenerate both keys in the Groq and Google AI Studio dashboards, then set
# them as real environment variables (e.g. in a .env file loaded by your
# process manager, or `export GROQ_API_KEY=...` before starting the server).
# Do NOT put the actual key values back into this file.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

groq_client = Groq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------------------------------------------------------
# In-Memory Cache (RAM) to eliminate repeated fetches and network lag
# -----------------------------------------------------------------------------
surah_list_cache = None
ayah_cache = {}


# -----------------------------------------------------------------------------
# 1. Quran Endpoints (Optimized with In-Memory Caching)
# -----------------------------------------------------------------------------

@app.get("/surahs")
async def get_surahs():
    """Returns the list of 114 Surahs instantly from RAM cache."""
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
    """Returns Arabic, English, and Urdu text cached after the first fetch."""
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
    """Searches English translation by query."""
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
# 2. Hybrid AI Chat (Groq Primary -> Gemini Fallback)
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str | None = None
    text: str | None = None
    language: str = "en"


SYSTEM_PROMPT = """You are an Islamic scholar assistant. Answer concisely and accurately according to the Holy Quran.
You MUST reply strictly with compact JSON, all on one line, in this exact format:
{"text": "<2-3 sentence explanation in the requested language>", "citations": ["<surah>:<ayah>", ...]}
Do not include markdown, code fences, or any text outside the JSON object."""

FALLBACK_TEXT = {
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

    # 1. Straightforward case.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Response got cut off mid-string (too few max_tokens): try closing it.
    try:
        return json.loads(raw + '"}')
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(raw + '"]}')
    except json.JSONDecodeError:
        pass

    # 3. Last resort: pull the text value and any "surah:ayah" refs out by hand.
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
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=700,  # raised from 350 - too small caused truncated JSON
            temperature=0.2,
        )
        parsed = _parse_json_reply(response.choices[0].message.content)
        if parsed and parsed.get("text"):
            return {
                "text": parsed["text"],
                "citations": parsed.get("citations", []),
            }
        print("[GROQ ERROR] Could not parse a usable reply from Groq output")

    except Exception as groq_err:
        print(f"\n[GROQ ERROR]: {groq_err}\n")

    # 2. Fallback: Gemini
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=700,
            ),
        )
        parsed = _parse_json_reply(response.text)
        if parsed and parsed.get("text"):
            return {
                "text": parsed["text"],
                "citations": parsed.get("citations", []),
            }
        print("[GEMINI ERROR] Could not parse a usable reply from Gemini output")

    except Exception as gemini_err:
        print(f"\n[GEMINI ERROR]: {gemini_err}\n")

    # 3. Both providers failed: the user always gets a normal sentence,
    # never a raw exception/status string inside the chat bubble.
    return {
        "text": FALLBACK_TEXT.get(req.language, FALLBACK_TEXT["en"]),
        "citations": [],
    }