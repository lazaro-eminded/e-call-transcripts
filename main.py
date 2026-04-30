import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

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

GHL_BASE = "https://services.leadconnectorhq.com"


@dataclass
class LocationConfig:
    location_id: str
    api_key: str
    user_id: str


def load_locations() -> dict[str, LocationConfig]:
    registry: dict[str, LocationConfig] = {}

    # Fuente 1: pares LOCATION_ID_<SUFFIX> + GHL_API_KEY_<SUFFIX> en env vars
    # Ej: LOCATION_ID_EMINDED + GHL_API_KEY_EMINDED
    for key, loc_id in os.environ.items():
        if not key.startswith("LOCATION_ID_"):
            continue
        suffix = key[len("LOCATION_ID_"):]
        api_key = os.getenv(f"GHL_API_KEY_{suffix}", "")
        if loc_id and api_key:
            registry[loc_id] = LocationConfig(
                location_id=loc_id,
                api_key=api_key,
                user_id=GHL_USER_ID,
            )

    # Fuente 2: variable de entorno LOCATIONS_JSON (JSON completo como string)
    locations_json_env = os.getenv("LOCATIONS_JSON", "")
    if locations_json_env:
        data = json.loads(locations_json_env)
        for loc_id, cfg in data.items():
            registry[loc_id] = LocationConfig(
                location_id=loc_id,
                api_key=cfg["api_key"],
                user_id=cfg.get("user_id", GHL_USER_ID),
            )

    # Fuente 3: archivo locations.json (desarrollo local)
    json_path = Path("locations.json")
    if json_path.exists():
        data = json.loads(json_path.read_text())
        for loc_id, cfg in data.items():
            registry.setdefault(
                loc_id,
                LocationConfig(
                    location_id=loc_id,
                    api_key=cfg["api_key"],
                    user_id=cfg.get("user_id", GHL_USER_ID),
                ),
            )

    # Fuente 4: env vars legacy GHL_API_KEY + GHL_LOCATION_ID (una sola location)
    if GHL_API_KEY and GHL_LOCATION_ID:
        registry.setdefault(
            GHL_LOCATION_ID,
            LocationConfig(location_id=GHL_LOCATION_ID, api_key=GHL_API_KEY, user_id=GHL_USER_ID),
        )

    return registry


LOCATIONS: dict[str, LocationConfig] = load_locations()


def ghl_headers(api_key: str, version: str = "2021-04-15") -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Version": version,
    }


@app.get("/")
async def health():
    return {"status": "ok", "service": "e-call-transcripts"}


def _handle_call_webhook(payload: dict, loc_cfg: LocationConfig, background_tasks: BackgroundTasks):
    custom_data = payload.get("customData", {})
    duration_raw = custom_data.get("Call Duration", "0")
    try:
        duration = float(duration_raw)
    except (ValueError, TypeError):
        duration = 0.0

    contact_id = (
        payload.get("contactId")
        or payload.get("contact_id")
        or payload.get("id")
    )

    print(f"[{loc_cfg.location_id}] contactId={contact_id} duration={duration}s")

    if duration < MIN_CALL_DURATION:
        print(f"[{loc_cfg.location_id}] skipped — duration {duration}s below minimum {MIN_CALL_DURATION}s")
        return {"status": "skipped", "reason": f"duration {duration}s below minimum {MIN_CALL_DURATION}s"}

    if not contact_id:
        raise HTTPException(status_code=400, detail="contactId not found in payload")

    background_tasks.add_task(process_call, contact_id, loc_cfg)
    return {"status": "accepted", "contactId": contact_id, "locationId": loc_cfg.location_id}


@app.post("/webhook/call-completed/{location_id}")
async def call_completed_by_location(location_id: str, request: Request, background_tasks: BackgroundTasks):
    loc_cfg = LOCATIONS.get(location_id)
    if loc_cfg is None:
        raise HTTPException(status_code=404, detail=f"locationId '{location_id}' is not configured")
    payload = await request.json()
    return _handle_call_webhook(payload, loc_cfg, background_tasks)


@app.post("/webhook/call-completed")
async def call_completed(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    location_id = payload.get("locationId") or payload.get("location_id") or payload.get("Location_id")
    if not location_id:
        if len(LOCATIONS) == 1:
            loc_cfg = next(iter(LOCATIONS.values()))
        else:
            raise HTTPException(status_code=400, detail="locationId not found in payload — use /webhook/call-completed/{locationId}")
    else:
        loc_cfg = LOCATIONS.get(location_id)
        if loc_cfg is None:
            raise HTTPException(status_code=400, detail=f"locationId '{location_id}' is not configured")
    return _handle_call_webhook(payload, loc_cfg, background_tasks)


async def process_call(contact_id: str, loc_cfg: LocationConfig):
    print(f"[{loc_cfg.location_id}][{contact_id}] Waiting {RECORDING_DELAY}s for GHL to process recording...")
    await asyncio.sleep(RECORDING_DELAY)

    audio_url = await get_recording_url(contact_id, loc_cfg)
    if not audio_url:
        print(f"[{loc_cfg.location_id}][{contact_id}] No recording URL found — aborting")
        return

    print(f"[{loc_cfg.location_id}][{contact_id}] Transcribing: {audio_url}")
    transcript = await transcribe(audio_url)
    if not transcript:
        print(f"[{loc_cfg.location_id}][{contact_id}] Empty transcription — aborting")
        return

    await post_note(contact_id, transcript, loc_cfg)
    print(f"[{loc_cfg.location_id}][{contact_id}] Done")


async def get_recording_url(contact_id: str, loc_cfg: LocationConfig) -> str | None:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{GHL_BASE}/conversations/search",
            headers=ghl_headers(loc_cfg.api_key),
            params={"locationId": loc_cfg.location_id, "contactId": contact_id},
        )

    if resp.status_code != 200:
        print(f"[{loc_cfg.location_id}][{contact_id}] GHL search error {resp.status_code}: {resp.text[:300]}")
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
        diarize=True,
        punctuate=True,
        language="es",
    )

    response = deepgram.listen.rest.v("1").transcribe_url({"url": audio_url}, options)
    results = response.results

    utterances = getattr(results, "utterances", None) or []
    if utterances:
        lines = []
        for utt in utterances:
            # Speaker 0 is assumed to be the agent (answers the call first)
            label = "Agente" if utt.speaker == 0 else "Cliente"
            lines.append(f"{label}: {utt.transcript}")
        return "\n".join(lines)

    # Fallback: plain transcript without diarization
    channels = getattr(results, "channels", None) or []
    if channels:
        return channels[0].alternatives[0].transcript

    return None


async def post_note(contact_id: str, transcript: str, loc_cfg: LocationConfig):
    body: dict = {"body": f"📞 Transcripción de llamada:\n\n{transcript}"}
    if loc_cfg.user_id:
        body["userId"] = loc_cfg.user_id

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{GHL_BASE}/contacts/{contact_id}/notes",
            headers=ghl_headers(loc_cfg.api_key, version="2021-07-28"),
            json=body,
        )

    if resp.status_code not in (200, 201):
        print(f"[{loc_cfg.location_id}][{contact_id}] Note error {resp.status_code}: {resp.text[:300]}")
    else:
        print(f"[{loc_cfg.location_id}][{contact_id}] Note posted successfully")
