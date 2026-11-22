# COPS/AUDI: audience pages (chat grounding only)

TMP's topic copy written for one "Asking as" audience: funders, investors, corporates, governments and non-profits. The chat reads these pages so it can answer in the visitor's terms. **They are not site pages:** the Next.js build reads only `CASE`, `LEAD` and `ABOU`, and the case-study cards come only from `CASE`.

| Audience (`audience`) | Topics |
|---|---|
| `funder` | climate-risk, transition-minerals, security, sovereign-debt, local-resilience, food-security |
| `investor` | climate-risk, transition-minerals, sovereign-debt, food-security |
| `corporate` | climate-risk, transition-minerals, food-security |
| `government` | climate-risk, transition-minerals, security, sovereign-debt, local-resilience, food-security |
| `non_profit` | climate-risk, transition-minerals, sovereign-debt, local-resilience, food-security |

- **Files:** `{audience}-{topic}_en.md` (the audience in kebab-case, so `non-profit-…`). The `audience` values are exactly the backend's "Asking as" keys; "just curious" has no pages. Format: `DOCS/STND/frontmatter_schema.md`, type `audience_page`.
- **Source:** Lou's Google Doc `1jAL3zlud2m0jiNBtghqxq6MVJ7VcBrlN` (the old asktmp.com audience copy), received 2026-11-22 as a plain-text export. The text is copied as written. Only export artefacts were removed: the `\-30-` end-of-section marks. Each page's headline is its first paragraph; its short sub-headings are `##`. Every file records the source in its `source` field.
- **Languages:** English only for now. A translation follows the usual COPS rules (draft until a native speaker reviews it).
- **Retrieval:** a visitor who picked an audience reads only that audience's pages, slightly boosted; with no audience, or "just curious", at most one page per topic. Either way at most two pages per search, so case studies keep their places. See `BACK/tmpbot_backend/retrieval.py`.
- **Copy notes for Lou** (not changed here, because the copy is kept as supplied):
  - `funder-transition-minerals`, "Unique data and analysis": the paragraph ends with a clause repeated from "Community empowerment" ("…and giving local people the agency they need to reach fair and lasting deals over minerals.").
  - Relative dates such as "Three years ago" (funder minerals page) and "a three-year initiative" (non-profit minerals page) date from the old site.
