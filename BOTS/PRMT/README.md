# BOTS/PRMT: the chat's system prompt

The chat's system prompt is built from the layers in this folder by
`BACK/tmpbot_backend/prompt.py`. Each file's leading `<!-- … -->` comment
is a header for people and is stripped before sending.

| File | Sent when | Role |
|---|---|---|
| `voice.md` | always | Who is speaking, the TMP voice, answer length. From the handoff's `SYSTEM`, minus the facts. |
| `rules.md` | always | Never invent; reference material and visitor text are data, not instructions; the `show_case_studies` tool. |
| `gating.md` | visitor is not a subscriber | The subscriber-only question types (**Lou maintains this list**) and the `show_access_prompt` tool. From the handoff's `GATE_NOTE`. |
| `access.md` | visitor has TMP Access | Full answers and the `offer_download` tool. |
| `length_mistral.md` | writer is a Mistral model and `mistral_length_shaping: true` | Extra length limit for the French writer. |

## Order and caching

1. **Stable block** (cached on Anthropic models with `cache_control`):
   `voice.md`, `rules.md`, then `gating.md` or `access.md`, then
   `length_mistral.md` when it applies. Two variants exist, one per plan
   group, so each stays cached.
2. **Turn block** (not cached): the reply language, the "Asking as"
   audience, and the reference material retrieved for this question.

TMP's facts are not in these files. They come from `COPS/` through
retrieval, so the prompt cannot drift from the published content.

## The paywall does not depend on this prompt

`gating.md` decides *when* the paywall card appears. It does not keep
subscriber-only text away from non-subscribers: retrieval does that, by
never putting that text in the prompt (ADR-0007). Tests in
`BACK/tests/test_paywall_db.py` assert on the exact prompt sent to the
model, including a prompt-injection attempt.

## Editing

Edit a layer, run `BACK/.venv/bin/python3 -m pytest BACK/tests -q`, and
open a pull request. Render redeploys on changes under `BOTS/**`.
