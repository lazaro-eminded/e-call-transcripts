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
MIN_CALL_DURATION = int(os.getenv("MIN_CALL_DURATION_SECONDS", "5"))
RECORDING_DELAY = int(os.getenv("RECORDING_DELAY_SECONDS", "60"))

GHL_BASE = "https://services.leadconnectorhq.com"

# GHL message type IDs that represent calls
# GHL type '1' = outbound/inbound call message
CALL_TYPE_IDS = {"1", "10", "TYPE_CALL"}


def ghl_headers(version: str = "2021-04-15") -> dict:
    return {
        "Authorization": f"Bearer {GHL_API_KEY}",
        "Content-Type": "application/json",
        "Version": version,
    }


_AUDIO_EXTENSIONS = (".mp3", ".mp4", ".wav", ".ogg", ".m4a", ".webm", ".aac")


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
    # Explicit well-known keys first
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

    # Fallback: scan all top-level and customData values for anything that
    # looks like an audio URL
    for source in (payload, custom_data):
        for k, v in source.items():
            if _looks_like_audio_url(v):
                print(f"[payload-scan] Found audio URL in field {k!r}: {v}")
                return v

    return None


@app.get("/")
async def health():
    return {"status": "ok", "service": "e-call-transcripts"}


@app.get("/debug/messages/{contact_id}")
async def debug_messages(contact_id: str):
    """Fetches raw GHL messages for a contact — use to inspect call message structure."""
    async with httpx.AsyncClient(timeout=30) as client:
        conv_resp = await client.get(
            f"{GHL_BASE}/conversations/search",
            headers=ghl_headers(),
            params={"locationId": GHL_LOCATION_ID, "contactId": contact_id, "limit": 5},
        )
        if conv_resp.status_code != 200:
            return {"error": conv_resp.status_code, "body": conv_resp.text}

        conversations = conv_resp.json().get("conversations", [])
        result = []
        for conv in conversations:
            conv_id = conv.get("id")
            msg_resp = await client.get(
                f"{GHL_BASE}/conversations/{conv_id}/messages",
                headers=ghl_headers(),
                params={"limit": 20},
            )
            messages = msg_resp.json() if msg_resp.status_code == 200 else {"error": msg_resp.text}
            result.append({"conversationId": conv_id, "raw": messages})

    return result


@app.api_route("/webhook/debug", methods=["GET", "POST"])
async def debug_webhook(request: Request):
    """Dump the full payload to logs — use this to inspect what GHL sends."""
    if request.method == "GET":
        return {"status": "debug endpoint active — send POST to inspect payload"}
    payload = await request.json()
    print("=== DEBUG PAYLOAD ===")
    for k, v in payload.items():
        print(f"  {k!r}: {v!r}")
    print("=== END PAYLOAD ===")
    return {"received_keys": list(payload.keys()), "payload": payload}


@app.post("/webhook/call-completed")
async def call_completed(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    print(f"[webhook] Received payload keys: {list(payload.keys())}")

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

    # Use recording URL from payload if GHL/Call Tools includes it directly.
    # We search both the top-level payload and customData for any key that
    # looks like a URL pointing to an audio file.
    recording_url = _find_recording_url(payload, custom_data)
    print(f"[{contact_id}] Recording URL from payload: {recording_url!r}")

    background_tasks.add_task(process_call, contact_id, recording_url)
    return {"status": "accepted", "contactId": contact_id}


async def process_call(contact_id: str, recording_url: str | bytes | None = None):
    if recording_url:
        print(f"[{contact_id}] Recording from webhook payload")
    else:
        delays = [
            int(os.getenv("RECORDING_DELAY_SECONDS", "60")),
            120,
            120,
        ]
        for attempt, wait in enumerate(delays, start=1):
            print(f"[{contact_id}] Attempt {attempt}: waiting {wait}s for GHL to process recording...")
            await asyncio.sleep(wait)
            recording_url = await get_recording_url(contact_id)
            if recording_url:
                break
            print(f"[{contact_id}] Attempt {attempt}: no recording found yet")

    if not recording_url:
        print(f"[{contact_id}] No recording found after all retries — aborting")
        return

    if isinstance(recording_url, bytes):
        print(f"[{contact_id}] Transcribing audio bytes ({len(recording_url)} bytes)")
        transcript = await transcribe_bytes(recording_url)
    else:
        print(f"[{contact_id}] Transcribing URL: {recording_url}")
        transcript = await transcribe(recording_url)
    if not transcript:
        print(f"[{contact_id}] Empty transcription — aborting")
        return

    await post_note(contact_id, transcript)
    print(f"[{contact_id}] Done")


async def get_recording_url(contact_id: str) -> str | bytes | None:
    """
    Finds the most recent call recording URL for a contact by:
    1. Searching conversations for the contact
    2. Fetching messages for each conversation via /conversations/{id}/messages
    3. Returning the URL from the most recent call message that has a recording
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{GHL_BASE}/conversations/search",
            headers=ghl_headers(),
            params={"locationId": GHL_LOCATION_ID, "contactId": contact_id, "limit": 10},
        )

        if resp.status_code != 200:
            print(f"[{contact_id}] GHL conversations search error {resp.status_code}: {resp.text[:300]}")
            return None

        data = resp.json()
        conversations = data.get("conversations", [])

        if not conversations:
            print(f"[{contact_id}] No conversations found for contact")
            return None

        print(f"[{contact_id}] Found {len(conversations)} conversation(s)")

        # Iterate conversations (already sorted by lastMessage desc in GHL)
        for conv in conversations:
            conv_id = conv.get("id")
            if not conv_id:
                continue

            msg_resp = await client.get(
                f"{GHL_BASE}/conversations/{conv_id}/messages",
                headers=ghl_headers(),
                params={"limit": 50},
            )

            if msg_resp.status_code != 200:
                print(f"[{contact_id}] Messages fetch error for conv {conv_id}: {msg_resp.status_code}: {msg_resp.text[:200]}")
                continue

            msg_data = msg_resp.json()
            # GHL wraps messages in a nested object: { messages: { messages: [...] } }
            messages = msg_data.get("messages", [])
            if isinstance(messages, dict):
                messages = messages.get("messages", [])

            print(f"[{contact_id}] Conv {conv_id}: {len(messages)} message(s)")

            # Sort by dateAdded descending so we pick the most recent call
            messages = sorted(messages, key=lambda m: m.get("dateAdded", ""), reverse=True)

            for msg in messages:
                raw_type = str(msg.get("type") or msg.get("messageType") or "").upper()
                is_call = "CALL" in raw_type or raw_type in CALL_TYPE_IDS

                if not is_call:
                    continue

                msg_id = msg.get("id")
                # Fetch the individual message — the list endpoint may omit fields
                if msg_id:
                    single_resp = await client.get(
                        f"{GHL_BASE}/conversations/{conv_id}/messages/{msg_id}",
                        headers=ghl_headers(),
                    )
                    if single_resp.status_code == 200:
                        msg = single_resp.json()

                print(f"[{contact_id}]   CALL msg id={msg.get('id')} type={msg.get('type')} meta={msg.get('meta')} attachments={msg.get('attachments')}")

                meta = msg.get("meta") or {}
                call_meta = meta.get("call") or {}
                attachments = msg.get("attachments") or []
                body = msg.get("body") or ""
                alt_id = msg.get("altId") or ""  # Twilio Call SID (CA...)
                conv_id_msg = msg.get("conversationId") or conv_id

                url = (
                    call_meta.get("url")
                    or call_meta.get("recordingUrl")
                    or call_meta.get("recording_url")
                    or meta.get("url")
                    or meta.get("recordingUrl")
                    or meta.get("recording_url")
                    or msg.get("url")
                    or msg.get("recordingUrl")
                    or (attachments[0].get("url") if attachments and isinstance(attachments[0], dict) else None)
                    or (attachments[0] if attachments and isinstance(attachments[0], str) else None)
                    or (body if body.startswith("http") else None)
                )

                # Use the official GHL recording endpoint
                if not url and msg_id:
                    url = await fetch_recording_by_message_id(client, contact_id, msg_id)

                if url:
                    size_info = f"{len(url)} bytes (WAV)" if isinstance(url, bytes) else url
                    print(f"[{contact_id}] Found recording in conv {conv_id}: {size_info}")
                    return url
                else:
                    call_duration = call_meta.get("duration", 0)
                    print(f"[{contact_id}]   Call (duration={call_duration}s, altId={alt_id!r}) — no recording URL")

    print(f"[{contact_id}] No call recording found in any conversation")
    return None


async def fetch_recording_by_message_id(
    client: httpx.AsyncClient,
    contact_id: str,
    msg_id: str,
) -> bytes | None:
    """
    Official GHL endpoint: returns WAV audio bytes directly.
    GET /conversations/messages/{messageId}/locations/{locationId}/recording
    Version: 2023-02-21
    """
    endpoint = f"{GHL_BASE}/conversations/messages/{msg_id}/locations/{GHL_LOCATION_ID}/recording"
    try:
        r = await client.get(endpoint, headers=ghl_headers(version="2023-02-21"))
        content_type = r.headers.get("content-type", "")
        print(f"[{contact_id}]   Recording endpoint → {r.status_code} content-type={content_type} size={len(r.content)}")
        if r.status_code == 200 and len(r.content) > 1000:
            return r.content
    except Exception as e:
        print(f"[{contact_id}]   Recording endpoint error: {e}")
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


async def transcribe_bytes(audio_bytes: bytes) -> str | None:
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        diarize=True,
        punctuate=True,
        language="es",
    )
    response = deepgram.listen.rest.v("1").transcribe_file(
        {"buffer": audio_bytes, "mimetype": "audio/wav"},
        options,
    )
    results = response.results

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
