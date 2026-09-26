# The 99-entry dataset catalogue

This is every dataset the project's research passes (`research/00`–`05`) surveyed and verified
against a primary source, in one place. It answers "what did the research actually find?" without
requiring a read of the 2,000+ line PRD.

**This is a research catalogue, not the corpus.** It lists candidates checked for license,
size, label space and primitive fit — most are not loaded into `better-jev-bench` yet. For what
is actually built and downloadable today, run `bjb catalogue` (built datasets only, PRD §6.3) or
see the table in [`README.md`](README.md#whats-here-right-now). Tracking which of the 99 are built
is [`STATUS.md`](STATUS.md) and [`better-jev-bench_PRD.md`](better-jev-bench_PRD.md) §12.

Every row below is sourced from a live check of a Hugging Face card, Kaggle page, UCI record,
GitHub `LICENSE` file, or official project/government page — never assumed from a dataset's name
or reputation. `verified_how` in each row names that source; the full detail (exact page checked,
transform difficulty, per-dataset caveats) lives in the linked `research/0N-*.md` file.

## License tiers, used throughout

| Tier | Meaning |
|---|---|
| **A** | Freely redistributable (MIT / Apache-2.0 / BSD / CC0 / CC BY / public domain). |
| **A-SA** | Tier A for permission purposes, but copyleft/share-alike (CC BY-SA, GFDL, DbCL, CDLA-Sharing) — obligations propagate to anything derived from it. |
| **B** | Research-only / non-commercial. Usable for research training and a research-tier eval slice; never a commercially-redistributable release. |
| **C** | Access-gated (DUA, credentialing, a request form). Can only ever be a pointer + instructions, never redistributed. |
| **D** | License unclear, unverified, or under competition/no-license terms. **Do not include without the follow-up work named in the row.** Being famous is not a license. |

Full tier rationale and the machine-readable enforcement (`manifest.toml`'s `license_tier` field,
CI's refusal to merge `tier = "D"`): PRD §4.

## Headline numbers

| | Count |
|---|---|
| Catalogued dataset entries (table rows, this file) | **99** |
| Unique sources (BANKING77, CFPB, and Jigsaw are each counted once but appear in two domain passes) | **≈96** |
| Tier A (incl. A-SA) | **~51** |
| Tier B | **~12** |
| Tier C | **3** |
| Tier D | **~31** |

---

## 1. Finance and quantitative decisions (18 entries)

Source: [`research/01-finance-quant.md`](research/01-finance-quant.md)

| Dataset | Tier | License | Size | Label space | Primitive |
|---|---|---|---|---|---|
| CFPB Consumer Complaint Database | A | US Gov public domain / CC0-1.0 (HF mirror) | 4M+ complaints, ~8GB, live-updating | Product/Sub-product (~18/~80), Issue/Sub-issue, Company response (5), Timely (Y/N) | `choice` + `noul` |
| BANKING77 | A | CC BY 4.0 | 13,083 | 77 banking intents | `choice` |
| Twitter Financial News Topic | A | MIT | 21,107 | 20 topics | `choice` |
| Twitter Financial News Sentiment | A | MIT | 9,938 / 2,486 | Bearish/Bullish/Neutral | `choice` |
| FiQA Sentiment | A | MIT | 822/117/234 | continuous −0.94…0.94 | `score` |
| StockNet | A | MIT | 88 stocks × 2y price + aligned tweets | next-day up/down (sometimes 3-class) | `choice` / `noul` |
| German Credit (Statlog) | A | CC BY 4.0 (UCI) | 1,000 × 20 attrs | good/bad credit risk | `noul` |
| Taiwanese Bankruptcy | A | CC BY 4.0 (UCI) | 6,819 × 95 ratios | bankrupt / not | `noul` |
| Default of Credit Card Clients | A | CC BY 4.0 (UCI) / CC0 (Kaggle mirror) | 30,000 × 24 | default next month | `noul` |
| Credit Card Approval Prediction | A | CC0 | ~438K applicants / ~1M history rows | derived approval/default (needs derivation) | `noul` |
| Vehicle Insurance Claim Fraud | A | CC0 | 15,420 | fraud / not fraud | `noul` |
| Credit Risk Dataset | A | CC0 | ~32.6K | loan_status; loan_grade A–G; loan_intent (6) | `noul` / `choice` |
| All Lending Club Loan Data | A | CC0 | ~1.36GB, millions of rows, 2007–2018 | grade/sub_grade (35-way), loan status | `choice` |
| S&P 500 ESG Risk Ratings | A | CC0 | ~500 | ESG score + 5-band category | `score` / `choice` |
| Credit Card Fraud (ULB) | A-SA | Database Contents License v1.0 | 284,807 txns, 492 fraud (0.172%) | fraud / not | `noul` |
| PaySim (synthetic mobile money) | A-SA | CC BY-SA 4.0 | ~6.36M txns, 8,213 fraud | fraud / not | `noul` |
| Financial PhraseBank | B | CC BY-NC-SA 3.0 | 4,846 (50agree) → 2,264 (allagree) | positive/negative/neutral | `choice` |
| Give Me Some Credit | D — blank/"Unknown" license field | unverified | 150,000 × 11 | serious delinquency in 2y | `noul` |

**Structural problem in this domain**: 6 of the 18 are rows of numbers with no natural-language
`state` at all — legitimate for label space, but need `synthetic_narrative` templating (PRD §2.4)
to have a `state`. No verified, openly-licensed dataset was found for algorithmic-trading-signal
or financial-statement-fraud classification; that gap was left empty rather than padded.
Excluded rather than listed with an inflated license: Home Credit Default Risk, IEEE-CIS Fraud
Detection, Porto Seguro (Kaggle competition terms).

## 2. Academic, scientific and STEM reasoning (20 entries)

Source: [`research/02-academic-stem.md`](research/02-academic-stem.md)

| Dataset | Tier | License | Size | Label space | Primitive |
|---|---|---|---|---|---|
| MMLU | A | MIT (wrapper; underlying exam provenance mixed) | 115,858 total, ~14K test | 4-way A–D | `choice` |
| MedMCQA | A | Apache-2.0 | 193,155 | 4-way, 21 subjects | `choice` |
| AQuA-RAT | A | Apache-2.0 | ~98,000 | 5-way + rationale | `choice` |
| MathQA | A | Apache-2.0 | ~37,000 | 5-way + operation program | `choice` |
| PRM800K | A | MIT | ~800K step-labels over ~100K solutions | per-step {−1, 0, +1} | `score` / `noul` |
| MATH (Hendrycks) | A | MIT | 12,500 | `\boxed{}` answer; level 1–5; 7 subjects | `score` (level) / `noul` (verify) |
| GSM8K | A | MIT | 8.5K | free-text CoT + numeric answer | `noul` via answer-verification |
| PubMedQA | A | MIT | 1K expert + 211.3K heuristic + 61.2K unlabeled | yes/no/maybe | `choice` |
| QASC | A | CC BY 4.0 | 9,980 | 8-way | `choice` |
| ASAP 2.0 (2024) | A | CC BY | ~13,000 essays | integer essay score | `score` |
| AI2 ARC (Challenge + Easy) | A-SA | CC BY-SA 4.0 | 7,787 | 3–5-way `answerKey` | `choice` |
| FOLIO | A-SA | CC BY-SA 4.0 | 1,430 over 487 premise sets | True/False/Uncertain | `choice` |
| LogiQA 2.0 | A-SA | CC BY-SA 4.0 | 8,678 | 4-way, civil-service exam | `choice` |
| SciQ | B | CC BY-NC 3.0 | 13,679 | correct + 3 distractors, no answer key | `choice` |
| SciFact | B | CC BY-NC 2.0 | 1,409 claims / ~5,183 abstracts | SUPPORTS/REFUTES/NOT_ENOUGH_INFO | `choice` / `noul` |
| Devign / CodeXGLUE Defect Detection | B | C-UDA (non-standard, usage restrictions) | 27,318 C functions | secure / vulnerable | `noul` |
| ReClor | C | CC BY-NC 4.0, password-gated request form | 6,138 (LSAT/GMAT), EASY/HARD | 4-way | `choice` |
| OpenBookQA | D — HF card literally says "unknown" | unverified | 5,957 | 4-way | `choice` |
| SciCite | D — HF "unknown" vs. GitHub Apache-2.0 badge, conflicting | unverified | 10,969 | method/background/result | `choice` |
| PeerRead | D — no LICENSE, opt-in consent collection, README notes withheld sections | unverified — highest legal risk surveyed | 14,000+ papers, 10,000+ reviews | accept/reject + per-aspect 1–5 scores | `noul` + `score` |
| ASAP-AES (2012) | D — Kaggle competition rules | unverified | ~13,000 essays / 8 prompts | integer score, range differs per prompt | `score` |
| LogiQA 1.0 | D — unclear (prefer 2.0) | unverified | 8,678 (same corpus as 2.0) | 4-way | `choice` |

Also checked but excluded: ACL-ARC (no authoritative license, copyrighted source text), ProofWriter
(license claimed on secondary sources only, not independently confirmed on an AI2-controlled page).

Findings: premium academic benchmarks skew non-commercial (SciQ, ReClor, SciFact all CC BY-NC);
peer-review/citation-intent data has the murkiest licensing surveyed (PeerRead, SciCite); `noul`
math data is thin (only PRM800K is natively step-level at scale); code is the thinnest sub-area
(Devign is the one solid binary-classification set, non-standard license); ASAP-AES (2012, Kaggle)
and ASAP 2.0 (2024, CC BY) are different datasets under different licenses — do not conflate.

## 3. NLP decisions beyond NLI (26 entries)

Source: [`research/03-nlp-beyond-nli.md`](research/03-nlp-beyond-nli.md) — the domain closest to
the ekVachan failure that motivated this project. ★ marks 10+ class label sets (schema-generality
stress tests).

| Dataset | Tier | License | Size | Label space | Primitive |
|---|---|---|---|---|---|
| ★ CLINC150 | A | CC-BY-3.0 | 59,275 | 150 intents × 10 domains + explicit `oos` class | `choice` |
| ★ BANKING77 *(dup of §1)* | A | CC-BY-4.0 | 13,083 | 77 intents | `choice` |
| ★ MASSIVE | A | CC-BY-4.0 | ~1M / 52 languages | 60 intents × 18 scenarios | `choice` |
| ★ CFPB Consumer Finance Complaints *(dup of §1)* | A | CC0-1.0 | 1M–10M, live-updating | ~18 Products, nested Sub-product/Issue | `choice` |
| Civil Comments | A | CC0-1.0 | ~2M | 7 continuous 0–1 fields (toxicity, threat, insult, …) | `score` |
| ★ HateXplain | A | MIT | 20,148 | hate / normal / offensive + target-community labels | `choice` |
| Hate Speech & Offensive (Davidson) | A | MIT | 24,783 | hate-speech / offensive / neither | `choice` |
| Phishing Dataset (aggregated) | A | Apache 2.0 | 10K–100K, 1.7GB | Phishing / Benign across URL/SMS/email/HTML | `choice` |
| ★ GoEmotions | A | Apache 2.0 | 58k curated (211k+ raw) | 27 emotions + neutral, multi-label | `choice` (multi-label) |
| ★ DBpedia-14 | A-SA | CC-BY-SA + GFDL (share-alike obligations) | 630,000 | 14 ontology classes | `choice` |
| Jigsaw Toxic Comment *(dup — also §4)* | A-SA | CC0 (Wikipedia source text CC-BY-SA 3.0 — dual) | 100K–1M | 6 multi-label binary flags | `choice` (multi-label) |
| ★ Customer Support Tickets | B | CC-BY-NC-4.0 | 61,800 | queue (52), priority (5), type (4), language (2) | `choice` × 2 |
| ★ dair-ai/emotion | B — card "other"; text says "educational and research purposes only" | flagged research-only | 436,809 | 6 emotions | `choice` |
| Yelp Review Full | B — Yelp Dataset Agreement restricts commercial use | flagged | 700,000 | 1–5 stars | `score` |
| ★ TweetEval (7 subtasks) | D — mixed per subtask, verify each | sentiment CC-BY-3.0, hate requires permission, rest unspecified | 200,785 | sentiment(3), emotion(4), irony(2), offensive(2), hate(2), stance(3×5), emoji(20) | `choice` |
| ★ Yahoo Answers Topics | D — unspecified | unverified | 1.46M | 10 classes | `choice` |
| ★ AG News | D — unspecified | unverified | 127,600 | 4 classes | `choice` |
| ★ 20 Newsgroups | D — not stated | unverified | 18,846 | 20 classes, hierarchical taxonomy | `choice` |
| App Reviews (F-Droid) | D — unspecified | unverified | 288,065 | 1–5 stars | `score` |
| SST-5 | D — not stated | unverified | 11,855 | 5-point ordinal sentiment | `choice` / ordinal |
| LIAR | D — unspecified | unverified | ~12.8k | 6-point truthfulness (pants-fire…true) | `choice` / ordinal |
| Language Identification | D — unspecified | unverified | 90,000 | 20 language codes | `choice` |
| Sarcasm News Headlines | D — unspecified | unverified | 55,328 | binary | `choice` |
| Enron Spam | D — not stated | unverified | 33,716 | spam / ham | `choice` |
| SMS Spam Collection | D — unspecified | unverified | 5,574 | ham / spam | `choice` |
| SNIPS (DeepPavlov mirror) | D — not stated | unverified | 14,498 | 7 intents | `choice` |

Rejected during verification: `amazon_reviews_multi` — confirmed **defunct**, pulled by the
provider and moved to `defunct-datasets/` on HF. The core finding: "classification" in production
text systems almost never means a fixed 3-way schema — the real distribution runs from 2 to 150+
classes, sometimes multi-label, sometimes continuous. AG News, 20 Newsgroups, SST-5, SMS Spam,
Enron Spam and SNIPS are all famous and all Tier D on the source checked — fame is not a license.

## 4. Operational and safety-adjacent decisions (20 entries)

Source: [`research/04-operational-safety.md`](research/04-operational-safety.md) — closest to what
a deployed decision layer does, and carrying the most serious responsible-use constraints (see
[README § Responsible use](README.md#responsible-use) and PRD §8.3).

| Dataset | Tier | License | Size | Label space | Primitive | Sensitivity |
|---|---|---|---|---|---|---|
| LEDGAR (LexGLUE) | A | CC BY 4.0 | 80,000 provisions | 100 provision types | `choice` | Legal — not attorney review |
| CUAD | A | CC BY 4.0 | 510 contracts, 13,000+ clauses, 84,325 rows | 41 categories (33 binary) | `choice` + span | Legal — flag risk, don't auto-approve |
| PhiUSIIL Phishing URL | A | CC BY 4.0 | 235,795 URLs | binary | `choice` | Security — one signal, not sole blocker |
| BODMAS | A | BSD 2-Clause | 57,293 malware + 77,142 benign | binary + 581 families + 14 categories | `choice` | Security — priority-affecting |
| CAIL2018 | A | MIT | 2,676,075 criminal cases | 202 charges, 183 articles, prison term | `choice` + `score` | Criminal justice — human judge in loop |
| Jigsaw Toxic Comment Challenge *(dup of §3)* | A-SA | CC0 comp data; Wikipedia text CC-BY-SA 3.0 | ~159,000 | 6 non-exclusive labels | `choice` / `score` | Moderation — needs appeal path |
| Bitext Customer Support | A-SA | CDLA-Sharing-1.0 | 26,872 | 27 intents × 10 categories | `choice` | Low-stakes |
| CIC-IDS2017 | B — research use w/ citation, no formal open license | flagged | ~2.8M flows, ~51GB | 7 attack categories + benign | `choice` | Security — detection aid, not autonomous block |
| UNSW-NB15 | B — academic research free; commercial needs author permission | flagged | 2,540,044 | 9 attack categories + normal | `choice` | Security — same |
| CheXpert | C — Stanford Research Use Agreement, non-commercial DUA | credentialed | 224,316 radiographs, 65,240 patients | 14 findings × {positive, negative, uncertain, unmentioned} | `noul` | Medical — training only, never diagnosis |
| MIMIC-IV-ED / MIETIC | C — PhysioNet Credentialed Health Data License 1.5.0 | credentialed | ~425,000 ED visits; MIETIC subset 9,629 | ESI acuity 1–5 + vitals + chief complaint | `score` (ordinal) | Medical — never autonomous ESI assignment |
| CaseHOLD | D — no explicit license file found | unverified | 53,000+ | 5-way holding selection | `choice` | Legal — holding ≠ legal correctness |
| ECtHR / ECHR | D — not stated; HUDOC redistribution terms unverified | unverified | 11,478 cases | binary violation per article | `choice` / `score` | Legal — must not predict real case outcomes |
| SOREL-20M | D — Apache 2.0 covers code only; dataset ToU unread | unverified | ~20M files | binary + multi-vendor tags | `choice` | Security |
| EMBER (2017/2018/2024) | D — LICENSE.txt exists, text unconfirmed | unverified | 1.1M / 1M / 3.2M | binary + file type | `choice` | Security |
| HateModerate | D — no LICENSE file found | unverified | test-suite scale, exact count unconfirmed | 41 Facebook community-standard categories × 4 severity tiers | `choice` | Moderation — policy category, not generic toxicity |
| Amazon Employee Access Challenge | D — Kaggle competition rules | unverified | 32,769 | binary ACTION over resource+role | `choice` | Access control — advisory only |
| Allstate Claims Severity | D — Kaggle competition rules | unverified | ~188,000 × 132 | continuous loss (needs binning) | `score` | Insurance — adjuster in loop |
| Resume Dataset (snehaanbhawal) | D — permissive on page but not OSI; scraped from LiveCareer | unverified | 2,400+ | 24 job categories | `choice` | HR/hiring — known proxy-bias risk |
| gretelai/symptom_to_diagnosis | D — unverified license; LLM-generated, not real patient data | unverified | 1,065 | 22 diagnosis classes | `choice` | Medical + synthetic |

Gaps found and not papered over: access-control decision data barely exists publicly (Amazon
Employee Access is close to the only public one, under competition terms); policy-category
moderation data tied to *named* platform policies is rare (HateModerate is the one solid find, and
its license is unverified); no open insurance/administrative-tribunal decision data exists; HR/hiring
is flagged most cautiously (no bias-audited hiring dataset justified inclusion). Diagnosis-prediction
datasets were deliberately excluded even with a caveat — a `choice`-typed diagnosis classifier is
closer to practicing medicine than to triage/routing, and that judgment is policy here, not
re-litigated per dataset (PRD §8.3).

## 5. Multimodal (15 entries)

Source: [`research/05-multimodal.md`](research/05-multimodal.md)

| Dataset | Tier | Modality | License | Size | Label space | Primitive |
|---|---|---|---|---|---|---|
| EuroSAT | A | Satellite (Sentinel-2) | MIT | 27,000 | 10 land-use classes | `choice` |
| AI2D | A-SA | Diagram + text | CC BY-SA 4.0 | ~5,000 diagrams, 15,000+ MCQs | 4-way | `choice` |
| HAM10000 | A | Dermatoscopic image | CC0 (Harvard Dataverse) | 10,015 | 7 lesion classes | `choice` |
| Chest X-Ray (Pneumonia) | A | X-ray image | CC BY 4.0 | 5,863 | Normal / Pneumonia (pediatric) | `choice` |
| SROIE (ICDAR2019) | A | Scanned receipt | CC-BY-4.0 | 973 receipts | 4 entity types | mixed (extraction; entity-type is `choice`) |
| `deepghs/nsfw_detect` | A | Image (moderation) | MIT | 10K–100K | multi-level moderation tags (taxonomy needs pinning) | `choice` |
| Google Speech Commands v0.02 | A | Audio | CC BY 4.0 | ~105,829 clips | 35 keywords + silence + unknown | `choice` (37-way) |
| MedMNIST v2 | A — aggregation CC BY 4.0; **each of 18 subsets keeps its own license, verify per-subset** | Medical image (18 sub-datasets) | see caveat | 708,069 2D + 9,998 3D | varies: binary → 14-way, some ordinal | `choice` / `score` |
| ScienceQA (image subset) | B — HF mirror says CC BY-SA 4.0, paper says CC BY-NC-SA; treat as NC-SA | Image + text | discrepancy, see caveat | 21,208 total, 10,332 with image | 2–5 answer choices | `choice` |
| FUNSD | B — custom research terms, explicit no-re-identification clause, not OSI | Scanned form | flagged | 199 forms, 9,707 entities | 4-way entity label | `choice` |
| ESC-50 | B — mixed per-clip Freesound licenses (CC-BY / CC-BY-NC / CC0) | Audio | flagged | 2,000 clips, 50 classes | 50 environmental sounds | `choice` |
| RVL-CDIP | D — "other"; UCSF Industry Documents terms, not OSI | Document image | unverified | 400,000 | 16 document types | `choice` |
| NWPU-RESISC45 | D — conflicting across mirrors | Aerial scene | unverified | 31,500 | 45 scene classes | `choice` |
| GTSRB | D — primary site citation-only; Kaggle mirror claims CC0 | Traffic sign | unverified | 39,209 + 12,630 | 43 sign classes | `choice` |
| ChartQA | D — repo GPL-3.0; charts are third-party copyright (Statista/Pew/OWID/PPIC) | Chart + text | unverified | tens of thousands of QA over ~4.8K charts | short extractive answers | `score` / `noul` natively |

Excluded after checking: DocVQA (gated behind RRC-portal registration, no published redistribution
license) and Food-101 (HF license "unknown", Foodspotting source terms restrict use beyond research
fair use). This is the messiest domain surveyed — vision-dataset licensing is far less standardized
than the HF-hub-native text world, and the recommendation is to ship per-dataset provenance notes,
never a blanket "open" claim.

### Multimodal extension pass (2026-09-24) — not counted in the 99

A follow-up pass found the on-distribution slice (GUI/agent/game data) the original multimodal pass
was missing. These 8 entries are additional and post-date the 99-entry count — several are already
built into the corpus (see [`STATUS.md`](STATUS.md)). Detail: `research/05-multimodal.md` §"Extension
pass, 2026-09-24".

| Dataset | Tier | Modality | License | Size | Primitive | Caveat |
|---|---|---|---|---|---|---|
| Atari-HEAD | A | Game frame + action | CC BY 4.0 (Zenodo record) | 117h, 20 games, ~8M actions | `choice` | Must split by trial, never by frame |
| Mind2Web | A | HTML (no images) | CC BY 4.0 | 6.74GB, ~2,350 tasks / 137 sites | `choice` | Not multimodal — text corpus |
| ScreenSpot-v2 | A | GUI screenshot + instruction | Apache-2.0 | ~1.2k items | `choice` | Too small to train on — eval-only |
| OS-Atlas-data | A (aggregate) — verify subset | GUI screenshots | apache-2.0 declared, aggregates other terms | 2.3M screenshots, 816GB | `choice` | Aggregate license ≠ subset license |
| GUIAct / GUICourse | D — HF says apache-2.0, paper says CC BY 4.0 (discrepancy) | GUI screenshot + metadata | unresolved | 67k web + 9.1k mobile steps | `choice` | Resolve with maintainers first |
| Multimodal-Mind2Web | D — `openrail`, use-restricted | Screenshot + HTML | not Tier A | 13.6GB, 14,193 action steps | `choice` | Best-fit dataset that can't ship permissively |
| Android in the Wild (AitW) | D — no LICENSE file, no terms in README | Android screenshots + actions | blocked | 715k episodes | `choice` | Blocked despite scale |
| ShowUI-desktop | D — no license stated; GPT-4o-augmented OmniACT derivative | Desktop screenshot + instruction | blocked | 7,496 samples | `choice` | Blocked on two independent grounds |

---

## How to use this file

- **Building something new?** Start with the Tier A / A-SA rows — CI will refuse anything tagged
  Tier D, and Tier B/C need explicit handling (research-only labeling, coverage disclosure).
- **Contributing a loader for one of these?** See [README § Add a dataset](README.md#add-a-dataset)
  and PRD §7. `bjb validate` runs the 8 gates, license tier included.
- **Something here needs re-verifying?** Every Tier D row names the specific follow-up needed —
  that's a standing invitation to do it and send a PR, not a rejection.
