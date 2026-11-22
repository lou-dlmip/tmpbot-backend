# COPS: content for the site and the chat

One copy of every piece of content that is both a page and chat grounding (ADR-0006). The Next.js build reads it to make pages; the ingest tool (M4.4) chunks and embeds it for retrieval.

| Folder | What | Files |
|---|---|---|
| `CASE/` | Case studies | `{id}_{lang}.md`, 15 ids |
| `LEAD/` | Leadership bios | `{id}_{lang}.md`, 3 ids |
| `ABOU/` | About, Our Story, timeline, practice areas | `about`, `our-story`, `timeline`, `practice-areas` |
| `PUBL/` | Subscriber-only publications (Phase 7) | empty for now |

- **Format:** `DOCS/STND/frontmatter_schema.md`. Check with `TOOL/.venv/bin/python3 TOOL/validate_frontmatter.py` (CI runs it).
- **Languages:** one file per language, sharing the `id`. English is the source and is `active`. Every other language is `draft` until a native speaker reviews it and sets `translation: "reviewed"`; drafts are neither shown nor retrieved, and the page falls back to English.
- **Where it came from (2026-11-13):** `node TOOL/extract_handoff.mjs content` imported the English case studies from the scrape of asktmp.com (`DOCS/DSGN/webflow_export_2026-11/`), and the English bios plus all About, timeline and practice-area files (seven languages) from the design handoff. Case-study bodies and bios in ja, es, pt, fr, de and it are machine drafts (`translation: "machine"`); their titles come from the handoff's `CASE_L`.
- **Editing:** COPS is now the source of truth. Edit the Markdown directly. On a substantive change to English, bump `version` in every language of that id and set the translations back to `draft`. The importer never overwrites an existing file unless run with `--force`.
- **Images:** named in the frontmatter (`hero_image`, `portrait`) and stored as real files in `FRNT/public/images/`.
