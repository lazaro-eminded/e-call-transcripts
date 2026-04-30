import asyncio
import json
import os
from dataclasses import dataclass, field
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
MIN_CALL_DURATION = int(os.getenv("MIN_CALL_DURATION_SECONDS", "0"))
RECORDING_DELAY = int(os.getenv("RECORDING_DELAY_SECONDS", "60"))

GHL_BASE = "https://services.leadconnectorhq.com"

CALL_TYPE_IDS = {"1", "10", "TYPE_CALL"}
_AUDIO_EXTENSIONS = (".mp3", ".mp4", ".wav", ".ogg", ".m4a", ".webm", ".aac")


@dataclass
class LocationConfig:
    location_id: str
    api_key: str
    user_id: str
    name: str = field(default="")


def load_locations() -> dict[str, LocationConfig]:
    registry: dict[str, LocationConfig] = {}

    # Pares LOCATION_ID_<SUFFIX> + GHL_API_KEY_<SUFFIX>
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

    # LOCATIONS_JSON como string JSON
    locations_json_env = os.getenv("LOCATIONS_JSON", "")
    if locations_json_env:
        data = json.loads(locations_json_env)
        for loc_id, cfg in data.items():
            registry[loc_id] = LocationConfig(
                location_id=loc_id,
                api_key=cfg["api_key"],
                user_id=cfg.get("user_id", GHL_USER_ID),
            )

    # Archivo locations.json (desarrollo local)
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

    # Fallback: env vars legacy (una sola location)
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


def _extract_location_id(payload: dict) -> str | None:
    for key, val in payload.items():
        if key.lower().replace("_", "") == "locationid" and val:
            return val
    return None


def _looks_like_audio_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    v = value.lower()
    return v.startswith("http") and (
        any(v.endswith(ext) for ext in _AUDIO_EXTENSIONS)
        or "recording" in v
        or "twilio.com/2010" in v
        or "calltools" in v
        or "storage.googleapis" in v
        or "s3.amazonaws" in v
    )


def _find_recording_url(payload: dict, custom_data: dict) -> str | None:
    candidates = [
        payload.get("recordingUrl"),
        payload.get("recording_url"),
        payload.get("Call Tools Call"),
        payload.get("subir llamada"),
        payload.get("Last Call"),
        custom_data.get("Recording URL"),
        custom_data.get("recordingUrl"),
        custom_data.get("recording_url"),
        custom_data.get("Call Tools Call"),
        custom_data.get("subir llamada"),
        custom_data.get("Last Call"),
    ]
    for v in candidates:
        if _looks_like_audio_url(v):
            return v

    for source in (payload, custom_data):
        for k, v in source.items():
            if _looks_like_audio_url(v):
                print(f"[payload-scan] Found audio URL in field {k!r}: {v}")
                return v

    return None


async def get_location_info(loc_cfg: LocationConfig) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{GHL_BASE}/locations/{loc_cfg.location_id}",
            headers=ghl_headers(loc_cfg.api_key, version="2023-02-21"),
        )
    if resp.status_code == 200:
        data = resp.json()
        return data.get("location", data)
    return {"error": resp.status_code, "detail": resp.text[:200]}


@app.on_event("startup")
async def startup():
    print(f"=== e-call-transcripts started — {len(LOCATIONS)} location(s) configured ===")
    for loc_cfg in LOCATIONS.values():
        info = await get_location_info(loc_cfg)
        name = info.get("name", "unknown")
        loc_cfg.name = name
        print(f"  ✓ {loc_cfg.location_id} → {name}")


@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": "e-call-transcripts",
        "locations": [{"id": c.location_id, "name": c.name} for c in LOCATIONS.values()],
    }


@app.get("/debug/messages/{contact_id}")
async def debug_messages(contact_id: str, location_id: str | None = None):
    loc_cfg = _resolve_location(location_id)
    if not loc_cfg:
        return {"error": "location not found"}

    async with httpx.AsyncClient(timeout=30) as client:
        conv_resp = await client.get(
            f"{GHL_BASE}/conversations/search",
            headers=ghl_headers(loc_cfg.api_key),
            params={"locationId": loc_cfg.location_id, "contactId": contact_id, "limit": 5},
        )
        if conv_resp.status_code != 200:
            return {"error": conv_resp.status_code, "body": conv_resp.text}

        conversations = conv_resp.json().get("conversations", [])
        result = []
        for conv in conversations:
            conv_id = conv.get("id")
            msg_resp = await client.get(
                f"{GHL_BASE}/conversations/{conv_id}/messages",
                headers=ghl_headers(loc_cfg.api_key),
                params={"limit": 20},
            )
            messages = msg_resp.json() if msg_resp.status_code == 200 else {"error": msg_resp.text}
            result.append({"conversationId": conv_id, "raw": messages})

    return result


@app.api_route("/webhook/debug", methods=["GET", "POST"])
async def debug_webhook(request: Request):
    if request.method == "GET":
        return {"status": "debug endpoint active — send POST to inspect payload"}
    payload = await request.json()
    print("=== DEBUG PAYLOAD ===")
    for k, v in payload.items():
        print(f"  {k!r}: {v!r}")
    print("=== END PAYLOAD ===")
    return {"received_keys": list(payload.keys()), "payload": payload}


def _resolve_location(location_id: str | None) -> LocationConfig | None:
    if location_id:
        return LOCATIONS.get(location_id)
    if len(LOCATIONS) == 1:
        return next(iter(LOCATIONS.values()))
    return None


def _handle_webhook(payload: dict, loc_cfg: LocationConfig, background_tasks: BackgroundTasks):
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

    if MIN_CALL_DURATION > 0 and duration < MIN_CALL_DURATION:
        print(f"[{loc_cfg.location_id}] skipped — duration {duration}s < minimum {MIN_CALL_DURATION}s")
        return {"status": "skipped", "reason": f"duration {duration}s below minimum {MIN_CALL_DURATION}s"}

    if not contact_id:
        raise HTTPException(status_code=400, detail="contactId not found in payload")

    recording_url = _find_recording_url(payload, custom_data)
    print(f"[{loc_cfg.location_id}][{contact_id}] Recording URL from payload: {recording_url!r}")

    background_tasks.add_task(process_call, contact_id, loc_cfg, recording_url)
    return {"status": "accepted", "contactId": contact_id, "locationId": loc_cfg.location_id}


@app.post("/webhook/call-completed/{location_id}")
async def call_completed_by_location(location_id: str, request: Request, background_tasks: BackgroundTasks):
    loc_cfg = LOCATIONS.get(location_id)
    if loc_cfg is None:
        raise HTTPException(status_code=404, detail=f"locationId '{location_id}' not configured")
    payload = await request.json()
    return _handle_webhook(payload, loc_cfg, background_tasks)


@app.post("/webhook/call-completed")
async def call_completed(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    location_id = _extract_location_id(payload)
    loc_cfg = _resolve_location(location_id)

    if loc_cfg is None:
        print(f"[webhook] WARNING: locationId={location_id!r} not found — payload keys: {list(payload.keys())}")
        return {"status": "skipped", "reason": "locationId not found or not configured"}

    return _handle_webhook(payload, loc_cfg, background_tasks)


async def process_call(contact_id: str, loc_cfg: LocationConfig, recording_url: str | bytes | None = None):
    if recording_url:
        print(f"[{loc_cfg.location_id}][{contact_id}] Recording from webhook payload")
    else:
        delays = [RECORDING_DELAY, 120, 120]
        for attempt, wait in enumerate(delays, start=1):
            print(f"[{loc_cfg.location_id}][{contact_id}] Attempt {attempt}: waiting {wait}s...")
            await asyncio.sleep(wait)
            recording_url = await get_recording_url(contact_id, loc_cfg)
            if recording_url:
                break
            print(f"[{loc_cfg.location_id}][{contact_id}] Attempt {attempt}: no recording found yet")

    if not recording_url:
        print(f"[{loc_cfg.location_id}][{contact_id}] No recording found after all retries — aborting")
        return

    try:
        if isinstance(recording_url, bytes):
            print(f"[{loc_cfg.location_id}][{contact_id}] Transcribing audio bytes ({len(recording_url)} bytes)")
            transcript = await transcribe_bytes(recording_url)
        else:
            print(f"[{loc_cfg.location_id}][{contact_id}] Transcribing URL: {recording_url}")
            transcript = await transcribe(recording_url)
    except Exception as e:
        print(f"[{loc_cfg.location_id}][{contact_id}] Transcription error: {type(e).__name__}: {e}")
        return

    if not transcript:
        print(f"[{loc_cfg.location_id}][{contact_id}] Empty transcription — aborting")
        return

    await post_note(contact_id, transcript, loc_cfg)
    print(f"[{loc_cfg.location_id}][{contact_id}] Done")


async def get_recording_url(contact_id: str, loc_cfg: LocationConfig) -> str | bytes | None:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{GHL_BASE}/conversations/search",
            headers=ghl_headers(loc_cfg.api_key),
            params={"locationId": loc_cfg.location_id, "contactId": contact_id, "limit": 10},
        )

        if resp.status_code != 200:
            print(f"[{loc_cfg.location_id}][{contact_id}] Conversations error {resp.status_code}: {resp.text[:300]}")
            return None

        conversations = resp.json().get("conversations", [])
        if not conversations:
            print(f"[{loc_cfg.location_id}][{contact_id}] No conversations found")
            return None

        print(f"[{loc_cfg.location_id}][{contact_id}] Found {len(conversations)} conversation(s)")

        for conv in conversations:
            conv_id = conv.get("id")
            if not conv_id:
                continue

            msg_resp = await client.get(
                f"{GHL_BASE}/conversations/{conv_id}/messages",
                headers=ghl_headers(loc_cfg.api_key),
                params={"limit": 50},
            )
            if msg_resp.status_code != 200:
                continue

            messages = msg_resp.json().get("messages", [])
            if isinstance(messages, dict):
                messages = messages.get("messages", [])

            messages = sorted(messages, key=lambda m: m.get("dateAdded", ""), reverse=True)

            for msg in messages:
                raw_type = str(msg.get("type") or msg.get("messageType") or "").upper()
                if "CALL" not in raw_type and raw_type not in CALL_TYPE_IDS:
                    continue

                msg_id = msg.get("id")
                if msg_id:
                    single = await client.get(
                        f"{GHL_BASE}/conversations/{conv_id}/messages/{msg_id}",
                        headers=ghl_headers(loc_cfg.api_key),
                    )
                    if single.status_code == 200:
                        msg = single.json()

                meta = msg.get("meta") or {}
                call_meta = meta.get("call") or {}
                attachments = msg.get("attachments") or []
                body = msg.get("body") or ""

                url = (
                    call_meta.get("url")
                    or call_meta.get("recordingUrl")
                    or meta.get("url")
                    or meta.get("recordingUrl")
                    or msg.get("url")
                    or msg.get("recordingUrl")
                    or (attachments[0].get("url") if attachments and isinstance(attachments[0], dict) else None)
                    or (attachments[0] if attachments and isinstance(attachments[0], str) else None)
                    or (body if body.startswith("http") else None)
                )

                if not url and msg_id:
                    url = await _fetch_recording_bytes(client, contact_id, msg_id, loc_cfg)

                if url:
                    print(f"[{loc_cfg.location_id}][{contact_id}] Recording found in conv {conv_id}")
                    return url

    print(f"[{loc_cfg.location_id}][{contact_id}] No recording found in any conversation")
    return None


async def _fetch_recording_bytes(
    client: httpx.AsyncClient,
    contact_id: str,
    msg_id: str,
    loc_cfg: LocationConfig,
) -> bytes | None:
    endpoint = f"{GHL_BASE}/conversations/messages/{msg_id}/locations/{loc_cfg.location_id}/recording"
    try:
        r = await client.get(endpoint, headers=ghl_headers(loc_cfg.api_key, version="2023-02-21"))
        print(f"[{loc_cfg.location_id}][{contact_id}] Recording endpoint → {r.status_code} size={len(r.content)}")
        if r.status_code == 200 and len(r.content) > 1000:
            return r.content
    except Exception as e:
        print(f"[{loc_cfg.location_id}][{contact_id}] Recording endpoint error: {e}")
    return None


def _parse_transcript(results) -> str | None:
    utterances = getattr(results, "utterances", None) or []
    if utterances:
        lines = []
        for utt in utterances:
            label = "Agente" if utt.speaker == 0 else "Cliente"
            lines.append(f"{label}: {utt.transcript}")
        return "\n".join(lines)
    channels = getattr(results, "channels", None) or []
    if channels:
        return channels[0].alternatives[0].transcript
    return None


async def transcribe(audio_url: str) -> str | None:
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        diarize=True,
        punctuate=True,
        detect_language=True,
    )
    response = deepgram.listen.rest.v("1").transcribe_url({"url": audio_url}, options)
    return _parse_transcript(response.results)


async def transcribe_bytes(audio_bytes: bytes) -> str | None:
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        diarize=True,
        punctuate=True,
        detect_language=True,
    )
    response = deepgram.listen.rest.v("1").transcribe_file(
        {"buffer": audio_bytes, "mimetype": "audio/x-wav"},
        options,
    )
    return _parse_transcript(response.results)


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
