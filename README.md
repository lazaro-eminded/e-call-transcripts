# e-call-transcripts

Servidor que recibe webhooks de HighLevel cuando se completa una llamada, descarga la grabación, la transcribe con Deepgram, y publica la transcripción como nota en el contacto — con separación de hablantes **A / B**.

Soporta múltiples subcuentas (locations) en una sola instancia.

---

## Cómo funciona

```
HighLevel (llamada completada)
        │
        ▼ POST /webhook/call-completed/{locationId}
  e-call-transcripts
        │
        ├─ Busca URL de grabación en el payload
        ├─ Si no la encuentra → consulta GHL Conversations API (3 intentos)
        │
        ▼ Deepgram nova-2
  Transcripción con diarización (hablantes A y B)
        │
        ▼ POST /contacts/{contactId}/notes
  Nota publicada en el contacto de HighLevel
```

---

## Variables de entorno

### Una sola subcuenta

```env
GHL_API_KEY=pit-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GHL_LOCATION_ID=xxxxxxxxxxxxxxxxxxxxxxxx
GHL_USER_ID=                          # opcional
DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MIN_CALL_DURATION_SECONDS=0           # 0 = sin filtro de duración
RECORDING_DELAY_SECONDS=60
```

### Múltiples subcuentas (Railway / cloud)

Agrega un par de variables por cada subcuenta. El sufijo puede ser cualquier nombre descriptivo:

```env
LOCATION_ID_EMINDED=xxxxxxxxxxxxxxxxxxxxxxxx
GHL_API_KEY_EMINDED=pit-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

LOCATION_ID_SAF=yyyyyyyyyyyyyyyyyyyyyyyy
GHL_API_KEY_SAF=pit-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy

GHL_USER_ID=                          # opcional, compartido entre subcuentas
DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Al arrancar el servidor verifica cada subcuenta con `GET /locations/{locationId}` y muestra su nombre en los logs.

### Múltiples subcuentas (desarrollo local)

Copia `locations.json.example` a `locations.json` (está en `.gitignore`) y rellena los valores:

```json
{
  "LOCATION_ID_AQUI": {
    "api_key": "pit-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "user_id": ""
  }
}
```

---

## Configuración del webhook en HighLevel

Cada subcuenta debe apuntar a su URL específica — esto garantiza que el servidor sepa a qué subcuenta pertenece la llamada:

```
POST https://tu-dominio.railway.app/webhook/call-completed/{locationId}
```

Evento: **Call Status**

---

## Endpoints

| Método | Path | Descripción |
|--------|------|-------------|
| `GET` | `/` | Health check — muestra las subcuentas configuradas |
| `POST` | `/webhook/call-completed/{locationId}` | Webhook principal (recomendado) |
| `POST` | `/webhook/call-completed` | Webhook genérico — detecta locationId del payload |
| `GET` | `/debug/messages/{contactId}?location_id=` | Inspecciona mensajes de un contacto |
| `POST` | `/webhook/debug` | Muestra el payload completo en logs |

---

## Formato de la transcripción

La nota se publica en el contacto con separación por hablante:

```
📞 Transcripción de llamada:

A: Hola, te llamo para hablar sobre el programa.
B: Sí, dime en qué consiste.
A: Básicamente lo que hacemos es...
B: Entiendo, ¿y cuál es el siguiente paso?
```

Speaker `0` → **A** (quien contesta primero)
Speaker `1+` → **B**

---

## Despliegue en Railway

1. Conecta el repositorio en Railway
2. Rama de despliegue: `main`
3. Agrega las variables de entorno (ver sección anterior)
4. Railway genera la URL — úsala para configurar los webhooks en HighLevel

---

## Stack

- **Python 3.13** / FastAPI / uvicorn
- **Deepgram** nova-2 — transcripción con diarización
- **HighLevel API** v2 — Conversations, Contacts/Notes, Locations
