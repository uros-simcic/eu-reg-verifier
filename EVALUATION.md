# Evaluation

This is a small, hand-curated evaluation set: 9 questions, not a statistical
benchmark. The goal is to test the two behaviors that matter most for a
grounded Q&A tool: does it cite correctly when the text supports an answer,
and does it abstain when the text doesn't. Results below are the actual
output of `demo_questions.py` against the deployed pipeline, confirmed
identical across two consecutive full runs. Exact cited articles can still
vary slightly between runs; the answer/abstain outcome does not.

## Method

Three buckets, three questions each:

- **(a) Clearly answerable**: the GDPR article is unambiguous.
- **(b) Genuinely outside the loaded regulation**: either not a
  data-protection matter, or governed by a sister regulation (ePrivacy)
  rather than the GDPR.
- **(c) Conditional**: the honest answer is "it depends," and the tool needs
  to explain the condition rather than give a flat yes or no.

## Results

| # | Bucket | Question | Outcome | Cited |
|---|---|---|---|---|
| 1 | a | Right to erasure | answered | Art. 17, 15, 16 |
| 2 | a | Info required at data collection | answered | Art. 13, 14 |
| 3 | a | Breach notification deadline | answered | Art. 33 |
| 4 | a | Maximum GDPR fine | answered | Art. 83 |
| 5 | b | Cookie consent banner | **abstained** | n/a |
| 6 | b | Opt-in consent for marketing email | answered | Art. 7, 6, 21 |
| 7 | c | Do I need a Data Protection Officer? | answered | Art. 37 |
| 8 | c | Can an employer read work email? | **abstained** | n/a |
| 9 | c | Fee to access my own data? | answered | Art. 15 |

All 9 behaved as intended in both runs, with one documented exception: #6.

## The one known limitation this surfaced

**#6 is a false answer, not a false abstain.** The GDPR does have a general
consent article (Art. 7), so the judge correctly finds "sufficient" text and
the model answers from it, but the actual rule for marketing email (opt-in
consent) sits in the **ePrivacy Directive**, not the GDPR. The tool has no
way to know a sister regulation governs the specific case, because only the
GDPR is loaded.

This is the same gap `check_cross_regulation_interplay()` in `rag.py`
documents as a stub: with a single regulation loaded, cross-regulation
interplay is undetectable by construction. Loading a second regulation and
activating that stub is what closes it. See `ARCHITECTURE.md`.

**#5 is the control case for the same failure mode**, and it passes: cookie
banners are also an ePrivacy matter, but there the GDPR articles retrieved
don't offer a plausible-looking answer, so the judge correctly abstains. #6
is harder precisely because GDPR *does* say something relevant, just not
the governing rule.

## The judge: structured output, a stated sufficiency rule, and voting

The judge uses Mistral's JSON response mode: it returns
`{"reasoning": "...", "sufficient": true or false}`, parsed directly with
`json.loads`. This replaced a free-text SUFFICIENT/INSUFFICIENT keyword
that had needed regex hardening twice.

Two further changes came out of testing, in order:

1. **A stated sufficiency rule.** Left to its own judgment, the model was
   too willing to call articles "sufficient" just for being topically
   related, and (separately) too willing to abstain on questions an article
   directly governs once given room to reason. The system prompt now states
   the rule explicitly: an article that defines or directly governs what's
   asked counts as sufficient, even if a complete answer needs to note open
   points; articles that only touch related ground, when the real question
   is governed by a different law, do not.
2. **Majority of three judge calls.** Mistral's API is not fully
   deterministic even at temperature 0, and single-call testing showed
   borderline questions flipping between answer and abstain from one call
   to the next. Voting made every question in this set land on the same
   outcome across two full, separately-run passes.

## Two known-borderline questions, deliberately not in the demo set

Two earlier candidates for bucket (c) were dropped after testing showed
they stay genuinely unstable even with voting, for reasons specific to this
corpus rather than the judge:

- **"Is an IP address personal data?"** Article 4 defines "online
  identifier" but never names IP addresses; the explicit statement is in
  **Recital 30**, which this corpus does not load (recitals are excluded by
  design; see README). Judge calls land close to 50/50 on whether Art. 4
  alone counts as sufficient.
- **"What happens if a company ignores my erasure request?"** The relevant
  remedy provisions (Art. 77, 79, 82, 83: complaints, judicial remedies,
  compensation, penalties) are scattered across the regulation and retrieval
  does not reliably surface them together for this phrasing, so the judge
  sees a different, incomplete slice of them on different runs.

Both are expected to resolve once recitals are ingested (a planned corpus
change, not a judge change) and are worth re-testing at that point.

## Other observed behavior worth noting

- **Citation precision**: citations reflect articles the model's answer
  named, extracted from its own text (`extract_citations()` in `rag.py`).
  This is accurate but not infallible: an answer that quotes an article's
  internal cross-reference (e.g. "point (a) of Article 6(1)") can pull that
  article into the citation list even if the model didn't rely on it.
- **Abstains now surface the nearest retrieved articles** instead of a dead
  end, so every outcome, answered or abstained, gives the user something
  checkable.

## Reproducing this

```
python demo_questions.py
```

Runs live against the Mistral API (per question: one embedding call, up to
three judge calls, and a generation call when the judge finds the articles
sufficient) and prints bucket, outcome, and citations for each question.

## Limitations of this evaluation itself

Nine hand-picked questions test specific behaviors on purpose; they are not
a random or statistically representative sample, and no automated scoring
(e.g. RAGAS-style faithfulness/relevance metrics) has been run. Expanding
this set (more edge cases per bucket, and a second rater) is the natural
next step before treating any pass rate here as a production SLA.
