# Batch integrity summary

Generated 2026-11-21T11:18:54+00:00 · pipeline v1.0.0 · device `cpu`

**15 files** · 13 analysed · 2 flagged as unprocessable · risk counts: HIGH 8, MEDIUM 1, LOW 4, UNPROCESSABLE 2

## Per-file results

| File | Status | Risk | Peer % | External % | AI % | Top peer match | Report |
|---|---|---|---|---|---|---|---|
| `pitch_01.mp4` | OK | 🔴 HIGH | 87.1 | 0.0 | – | submission_05.pdf | [md](pitch_01.mp4.md) |
| `pitch_02.mp4` | OK | 🔴 HIGH | 0.0 | 86.2 | – | – | [md](pitch_02.mp4.md) |
| `submission_01.pdf` | OK | 🔴 HIGH | 100.0 | 0.0 | 4.8 | submission_02.docx | [md](submission_01.pdf.md) |
| `submission_02.docx` | OK | 🔴 HIGH | 100.0 | 0.0 | 4.8 | submission_01.pdf | [md](submission_02.docx.md) |
| `submission_03.docx` | OK | 🔴 HIGH | 81.4 | 0.0 | 7.8 | submission_04.docx | [md](submission_03.docx.md) |
| `submission_04.docx` | OK | 🔴 HIGH | 81.5 | 0.0 | 4.0 | submission_03.docx | [md](submission_04.docx.md) |
| `submission_06.pdf` | OK | 🔴 HIGH | 0.0 | 0.0 | 93.0 | – | [md](submission_06.pdf.md) |
| `submission_09.pdf` | OK | 🔴 HIGH | 0.0 | 52.0 | 7.5 | – | [md](submission_09.pdf.md) |
| `submission_05.pdf` | OK | 🟠 MEDIUM | 22.7 | 0.0 | 8.8 | pitch_01.mp4 | [md](submission_05.pdf.md) |
| `submission_07.pdf` | OK | 🟢 LOW | 0.0 | 0.0 | 4.1 | – | [md](submission_07.pdf.md) |
| `submission_08.docx` | OK | 🟢 LOW | 0.0 | 0.0 | 16.8 | – | [md](submission_08.docx.md) |
| `submission_10.pdf` | OK | 🟢 LOW | 0.0 | 0.0 | 4.0 | – | [md](submission_10.pdf.md) |
| `submission_11.docx` | OK | 🟢 LOW | 0.0 | 0.0 | 12.6 | – | [md](submission_11.docx.md) |
| `pitch_03.mp4` | NO_AUDIO_TRACK | ⚪ UNPROCESSABLE | – | – | – | – | [md](pitch_03.mp4.md) |
| `submission_12.pdf` | EMPTY_FILE | ⚪ UNPROCESSABLE | – | – | – | – | [md](submission_12.pdf.md) |

## Peer similarity matrix

Cell = % of the **row** file's assessed (non-template) words that match the **column** file. Asymmetric by design: a short file copied into a long one scores high in its own row only.

| | P01 | P02 | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 | S11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **P01** | · |  |  |  |  |  | **87** |  |  |  |  |  |  |
| **P02** |  | · |  |  |  |  |  |  |  |  |  |  |  |
| **S01** |  |  | · | **100** |  |  |  |  |  |  |  |  |  |
| **S02** |  |  | **100** | · |  |  |  |  |  |  |  |  |  |
| **S03** |  |  |  |  | · | **81** |  |  |  |  |  |  |  |
| **S04** |  |  |  |  | **82** | · |  |  |  |  |  |  |  |
| **S05** | 23 |  |  |  |  |  | · |  |  |  |  |  |  |
| **S06** |  |  |  |  |  |  |  | · |  |  |  |  |  |
| **S07** |  |  |  |  |  |  |  |  | · |  |  |  |  |
| **S08** |  |  |  |  |  |  |  |  |  | · |  |  |  |
| **S09** |  |  |  |  |  |  |  |  |  |  | · |  |  |
| **S10** |  |  |  |  |  |  |  |  |  |  |  | · |  |
| **S11** |  |  |  |  |  |  |  |  |  |  |  |  | · |

Legend: P01 = `pitch_01.mp4` · P02 = `pitch_02.mp4` · S01 = `submission_01.pdf` · S02 = `submission_02.docx` · S03 = `submission_03.docx` · S04 = `submission_04.docx` · S05 = `submission_05.pdf` · S06 = `submission_06.pdf` · S07 = `submission_07.pdf` · S08 = `submission_08.docx` · S09 = `submission_09.pdf` · S10 = `submission_10.pdf` · S11 = `submission_11.docx`

## Files that could not be analysed

- `pitch_03.mp4`: **NO_AUDIO_TRACK** – Video has no audio track; nothing to transcribe. The file was flagged and skipped.
- `submission_12.pdf`: **EMPTY_FILE** – File is empty (0 bytes); nothing to analyse.

## Run details

- Boilerplate rule: sentences shared by ≥ 3 files were discounted (45 sentences).
- Timings (s): extraction 13.24, embedding_and_indexing 16.92, reference_corpus 0.47, matching 0.1, ai_and_scoring 32.27
