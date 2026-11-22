# Translation review notes (D12, M3.5)

Every non-English file in `COPS/` and every non-English string in `FRNT/messages/` is a **draft**. Drafts are neither shown nor retrieved; the site shows English with an "English only" note until a native speaker reviews the file.

## What needs review

| Source | Files | `translation` |
|---|---|---|
| Design handoff copy | `COPS/ABOU/about_*`, `our-story_*`, `timeline_*`; all of `FRNT/messages/{ja,es,pt,fr,de,it}.json`; case-study titles | `handoff` |
| Machine drafts (2026-11-13) | `COPS/CASE/*_{ja,es,pt,fr,de,it}.md` (bodies, dates, translated partner descriptions), `COPS/LEAD/*_{…}.md`, `COPS/ABOU/practice-areas_*` (title only), and the supplement strings in `TOOL/data/ui_supplement.json` ("English only" note, fair-use note) | `machine` |

## How to sign a file off

1. Read the file next to its `_en.md` original and correct the text in place.
2. Set `translation: "reviewed"` and `source_lifecycle: "active"`.
3. Run `TOOL/.venv/bin/python3 TOOL/validate_frontmatter.py`. It must report 0 errors.

UI strings in `FRNT/messages/` are generated, so fix them in the handoff's place of record instead: either a new handoff export, or `TOOL/data/ui_supplement.json` for the supplement strings. Then run `npm run extract:strings` in `FRNT/`.

## Points the drafts flagged for reviewers

**All languages**
- Partner and funder **organisation names stay in English**, including notes in brackets such as "MSCI (financial services company based in New York, USA)" and "ODI (UK Think Tank)". Only the Landesa entry (a full sentence) is translated. Decide whether the bracketed notes should be translated.
- **Numbers use local formats.** For example, 85,260 is written 85.260 in de, es and it and 85 260 in fr; 5.5 is written 5,5. Japanese uses 万 and 億: $5m becomes 500万ドル.
- In the Angel case study, "Transition Minerals" (the project) follows each language's handoff title.
- Fixed 2026-11-15 (Lou approved): "Brumadhino" is now Brumadinho in every language, and the English "more 300 processing assets" now reads "more than 300". Both were copied from the live site. `transition-minerals` is at version 1.1.
- Fixed 2026-11-15 (Lou approved): the two Dryad "Lessons learned" items are now `###` sub-headings, each followed by its sentence, in every language. `dryad` is at version 1.1. Reviewers: check the sub-heading wording.
- Willie Munden's bio, "biometric data": this probably means forest or field measurements, not data about people. Check the wording in every language.

**ja**
- Job titles: マネージングディレクター (Yumiko), チーフエンジニア, 創業者. Japanese companies are written in Japanese (住友グループ, 三菱グループ, 天理大学), and people's names stay in Latin script. Check the official name of Yumiko's degree course at Tenri University (ポルトガル語・ブラジル研究).
- Terms: IPLCs → 先住民族や地域コミュニティ, fortress conservation → 要塞型保全, tailings ponds → 鉱滓ダム.

**de**
- Titles: Geschäftsführerin, Chefingenieur, Gründer. "Côte d'Ivoire" uses a plain apostrophe. The degree is rendered „Luso-Brasilianistik“.

**es / pt**
- IPLC and EWE are kept as English abbreviations. The UN may use local forms (for example PICL); confirm which to use.
- es: "costo" (not "coste"), "búsqueda de rentas", "autorreporte", "herramienta de manejo de plagas y enfermedades".
- pt (Brazilian): "due diligence" and "advocacy" stay as English loanwords; "busca de renda", "autorrelato", RSPO mills → "usinas de processamento".
- Titles: Directora general / Diretora-geral, Ingeniero jefe / Engenheiro-chefe, Fundador.

**fr / it**
- fr uses PACL for IPLC, and « conservation forteresse ». "Due diligence" is kept in both languages; fr could use « vérification diligente » instead.
- The fr body follows the handoff title "Mission Climat"; it keeps "Mission Climate Project".
- Titles: fr Directrice générale; it Amministratrice delegata, Ingegnere capo. "Based in" became it "ha base in".
