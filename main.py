import asyncio
import os

import httpx
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

load_dotenv()

app = FastAPI(title="e-call-transcripts")

GHL_API_KEY = os.getenv("GHL_API_KEY", "")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID", "")
GHL_USER_ID = os.getenv("GHL_USER_ID", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
MIN_CALL_DURATION = int(os.getenv("MIN_CALL_DURATION_SECONDS", "60"))
RECORDING_DELAY = int(os.getenv("RECORDING_DELAY_SECONDS", "60"))

# Tracks contact IDs currently being processed to prevent duplicate notes
_processing: set[str] = set()

GHL_BASE = "https://services.leadconnectorhq.com"


def ghl_headers(version: str = "2021-04-15") -> dict:
    return {
        "Authorization": f"Bearer {GHL_API_KEY}",
        "Content-Type": "application/json",
        "Version": version,
    }


@app.get("/")
async def health():
    return {"status": "ok", "service": "e-call-transcripts"}


@app.post("/webhook/call-completed")
async def call_completed(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()

    custom_data = payload.get("customData", {})
    duration_raw = custom_data.get("Call Duration", "0")

    try:
        duration = float(duration_raw)
    except (ValueError, TypeError):
        duration = 0.0

    if duration < MIN_CALL_DURATION:
        return {
            "status": "skipped",
            "reason": f"duration {duration}s below minimum {MIN_CALL_DURATION}s",
        }

    contact_id = (
        payload.get("contactId")
        or payload.get("contact_id")
        or payload.get("id")
    )
    if not contact_id:
        raise HTTPException(status_code=400, detail="contactId not found in payload")

    if contact_id in _processing:
        return {"status": "skipped", "reason": "already processing", "contactId": contact_id}

    background_tasks.add_task(process_call, contact_id)
    return {"status": "accepted", "contactId": contact_id}


async def process_call(contact_id: str):
    _processing.add(contact_id)
    try:
        print(f"[{contact_id}] Waiting {RECORDING_DELAY}s for GHL to process recording...")
        await asyncio.sleep(RECORDING_DELAY)

        audio_url = await get_recording_url(contact_id)
        if not audio_url:
            print(f"[{contact_id}] No recording URL found — aborting")
            return

        print(f"[{contact_id}] Transcribing: {audio_url}")
        transcript = await transcribe(audio_url)
        if not transcript:
            print(f"[{contact_id}] Empty transcription — aborting")
            return

        await post_note(contact_id, transcript)
        print(f"[{contact_id}] Done")
    finally:
        _processing.discard(contact_id)


async def get_recording_url(contact_id: str) -> str | None:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{GHL_BASE}/conversations/search",
            headers=ghl_headers(),
            params={"locationId": GHL_LOCATION_ID, "contactId": contact_id},
        )

    if resp.status_code != 200:
        print(f"[{contact_id}] GHL search error {resp.status_code}: {resp.text[:300]}")
        return None

    data = resp.json()
    conversations = data.get("conversations", [])

    for conv in conversations:
        messages = conv.get("messages", [])
        # GHL sometimes wraps messages in a nested dict
        if isinstance(messages, dict):
            messages = messages.get("messages", [])

        for msg in messages:
            msg_type = str(msg.get("type") or msg.get("messageType") or "").upper()
            if "CALL" in msg_type:
                meta = msg.get("meta", {})
                url = (
                    meta.get("url")
                    or meta.get("recordingUrl")
                    or msg.get("url")
                    or msg.get("recordingUrl")
                )
                if url:
                    return url

    return None


async def transcribe(audio_url: str) -> str | None:
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        punctuate=True,
        language="es",
    )

    response = deepgram.listen.rest.v("1").transcribe_url({"url": audio_url}, options)
    results = response.results

    channels = getattr(results, "channels", None) or []
    if channels:
        return channels[0].alternatives[0].transcript

    return None


async def post_note(contact_id: str, transcript: str):
    body: dict = {"body": f"📞 Transcripción de llamada:\n\n{transcript}"}
    if GHL_USER_ID:
        body["userId"] = GHL_USER_ID

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{GHL_BASE}/contacts/{contact_id}/notes",
            headers=ghl_headers(version="2021-07-28"),
            json=body,
        )

    if resp.status_code not in (200, 201):
        print(f"[{contact_id}] Note error {resp.status_code}: {resp.text[:300]}")
    else:
        print(f"[{contact_id}] Note posted successfully")
