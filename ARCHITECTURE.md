# Architecture

## Pipeline

```
                 ┌─────────────────────────────────────────────┐
                 │              corpus/gdpr/*.html              │
                 │   (EUR-Lex consolidated GDPR, committed)     │
                 └───────────────────────┬───────────────────────┘
                                          │ parse.py
                                          ▼
                          one chunk per article
                       {reg, article, title, text}
                                          │ ingest.py
                                          │ mistral-embed (one call per article,
                                          │ throttled to the account's rate limit)
                                          ▼
                     index/regulations.faiss + metadata.json
                          (committed — never rebuilt at runtime)

────────────────────────────── at query time ──────────────────────────────

  user question
       │
       ▼
  ┌─────────────┐   mistral-embed    ┌──────────────┐
  │  retrieve   │ ─────────────────▶ │  FAISS top-k │
  └─────────────┘                    └──────┬───────┘
                                             │ k retrieved articles
                                             ▼
                                    ┌─────────────────┐
                                    │      judge      │  mistral-small:
                                    │ (sufficient?)    │  "do these articles
                                    └────────┬────────┘   answer this?"
                                             │
                        ┌────────────────────┴────────────────────┐
                        │ no                                      │ yes
                        ▼                                         ▼
                ┌───────────────┐                        ┌─────────────────┐
                │    abstain    │                        │     answer      │  mistral-small,
                │ "not covered  │                        │ grounded only in │  grounded on the
                │ by the loaded │                        │ retrieved text,  │  retrieved articles
                │ regulation"   │                        │  with citations  │  only
                └───────────────┘                        └─────────────────┘
```

## Why retrieve → judge → abstain → cite, not a single call

A single "retrieve then generate" pass has no way to say "I don't actually
know" — it will paraphrase whatever text got retrieved, confidently, even when
that text doesn't answer the question. The judge step is a separate,
dedicated decision: *do the retrieved articles actually support an answer?*
Only a "yes" proceeds to generation. This is what makes the abstain gate real
rather than cosmetic — it's a distinct model call with one job, not a prompt
instruction hoping the same call also polices itself.

## Prompt-injection separation

Retrieved article text is rendered into clearly delimited `<article>` blocks
and only ever placed in the *user* turn, never the system role — see
`delimit_articles()` in `rag.py`. The system prompt instructs the model to
treat that content as reference data, not instructions. Risk is low for
statute text, but the architecture is corpus-agnostic (see below), and a
future corpus could be arbitrary documents.

## Corpus-agnostic by design

Nothing in `rag.py` hard-codes "GDPR" — the active regulation list, display
names, and full names come from `config.json`, and every chunk carries a
`reg` id from ingestion onward. Adding a second regulation (ePrivacy, DSA,
NIS2, the AI Act) is a data-and-config change: drop the source file in
`corpus/<reg>/`, add an entry to `config.json`, re-run `ingest.py`. No changes
to the retrieve/judge/generate code path.

The one deliberately inert piece is `check_cross_regulation_interplay()` in
`rag.py` — a documented no-op stub for the case where two loaded regulations
both plausibly answer a question (e.g. GDPR and ePrivacy on cookie consent).
With one regulation loaded, that logic is untestable, so it stays a
documented placeholder rather than speculative code — see the docstring for
what it needs to do once a second corpus is active.

## Deployment

Single stateless container (`Dockerfile`): the FAISS index and corpus are
baked into the image at build time, so there's no runtime ingestion step and
no persistent volume. The app binds to the host's injected `$PORT`. Secrets
(`MISTRAL_API_KEY`, `DEMO_USER`, `DEMO_PASS`) are supplied as environment
variables by the host, never committed.
