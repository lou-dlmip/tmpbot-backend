# TMPBOT backend (`BACK/`)

The FastAPI service behind the asktmp.com chat. In Phase 2 it is a hello-world: a health probe and a canned answer streamed in the real wire format, so the frontend can be built against it before any model account exists. Retrieval and the real model call arrive in Phase 5.

## Endpoints

| Method and path | Auth | What it does |
|---|---|---|
| `GET /health` | none | `{"status":"ok","version":…,"env":…}` for Render's health check. |
| `POST /api/v1/chat` | `X-TMPBOT-Proxy-Secret` header | Streams server-sent events: `meta`, `token`…, optional card events, then `done` or `error`. Contract: `DOCS/STND/streaming_wire_format.md`. |
| `GET /docs` | none, not in production | Interactive API docs. |

## Install and run (local)

```bash
/home/lou2/venv/bin/python3 -m venv BACK/.venv
BACK/.venv/bin/pip install -r BACK/requirements-dev.txt

TMPBOT_SHARED_SECRET=local-secret \
BACK/.venv/bin/uvicorn --app-dir BACK --factory \
  tmpbot_backend.main:create_app --host 127.0.0.1 --port 8100 --no-access-log
```

Stream one answer (`-N` turns off curl's buffering, so text appears piece by piece):

```bash
curl -N -X POST http://127.0.0.1:8100/api/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'X-TMPBOT-Proxy-Secret: local-secret' \
  -d '{"language":"ja","messages":[{"role":"user","content":"こんにちは"}]}'
```

Add `"demo":"cases"`, `"gate"`, `"limit"` or `"download"` to the body to force a card event (ignored in production).

## Configuration

Read once at start-up by `tmpbot_backend/config.py`; names documented in `INFR/env_inventory.md`.

| Name | Default | Effect |
|---|---|---|
| `TMPBOT_ENV` | `local` | `local`, `staging` or `production`. Production hides `/docs` and ignores `demo`. |
| `TMPBOT_SHARED_SECRET` | empty | Required for chat. Empty means every chat request gets 401 (fail closed). |
| `TMPBOT_LOG_LEVEL` | `INFO` | Python logging level. |
| `SENTRY_DSN_BACKEND` | empty | Sentry starts only when set; no personal data is sent. |

An invalid `TMPBOT_ENV` or `TMPBOT_LOG_LEVEL` stops start-up with a clear error.

## Logs

One JSON object per line on stdout: `ts` (UTC), `level`, `logger`, `msg`, plus fields such as `request_id`, `path`, `status`, `duration_ms`. Each request gets an `X-Request-Id` response header; a well-formed id sent by the proxy is reused so frontend and backend lines can be matched.

## Modules

| File | Role |
|---|---|
| `tmpbot_backend/main.py` | App factory `create_app()`, routes, request models, access-log middleware, stream guard. |
| `tmpbot_backend/wire.py` | SSE encoders for every event, and a decoder for tests. |
| `tmpbot_backend/canned.py` | Canned replies in seven languages; demo card switches. Replaced in Phase 5. |
| `tmpbot_backend/auth.py` | Shared-secret check for the Next.js proxy. |
| `tmpbot_backend/config.py` | Settings from environment variables. |
| `tmpbot_backend/logging_setup.py` | JSON log formatter. |
| `tmpbot_backend/observability.py` | Optional Sentry start-up. |
| `tmpbot_backend/locales.py` | The seven locales; must mirror `FRNT/i18n/routing.ts`. |

## Tests

From the repo root:

```bash
BACK/.venv/bin/python3 -m pytest BACK/tests -q
TOOL/.venv/bin/flake8 . && TOOL/.venv/bin/isort --check-only .
```

`setup.cfg` puts `BACK/` on the import path, as Render's `--app-dir BACK` does.

## Deploying

Not deployed. Render settings are in `INFR/render.yaml`; steps in `DOCS/RUNB/render_backend_setup.md`.

## Troubleshooting

- **Every chat request returns 401:** `TMPBOT_SHARED_SECRET` is unset on the backend, or the proxy sends a different value.
- **Text arrives all at once through a proxy:** something is buffering. The backend sends `X-Accel-Buffering: no` and `Cache-Control: no-cache`; check the proxy honours them, and use `curl -N` locally.
- **`ModuleNotFoundError: tmpbot_backend`:** start uvicorn with `--app-dir BACK`, or run pytest from the repo root.
