# Evaluation

This is a small, hand-curated evaluation set: 9 questions, not a statistical
benchmark. The goal is to test the two behaviors that matter most for a
grounded Q&A tool: does it cite correctly when the text supports an answer,
and does it abstain when the text doesn't. Results below are the actual
output of `demo_questions.py` against the deployed pipeline. This is one
representative run; the exact cited articles can vary slightly between
runs, even at temperature 0.

## Method

Three buckets, three questions each:

- **(a) Clearly answerable**: the GDPR article is unambiguous.
- **(b) Genuinely outside the loaded regulation**: either not a
  data-protection matter, or governed by a sister regulation (ePrivacy)
  rather than the GDPR.
- **(c) Borderline**: tests judge precision; either an answer or an abstain
  can be the honest outcome, depending on how the articles read.

## Results

| # | Bucket | Question | Outcome | Cited |
|---|---|---|---|---|
| 1 | a | Right to erasure | answered | Art. 17, 15, 16, 5 |
| 2 | a | Info required at data collection | answered | Art. 13, 14 |
| 3 | a | Breach notification deadline | answered | Art. 33 |
| 4 | a | Maximum GDPR fine | answered | Art. 83 |
| 5 | b | Cookie consent banner | **abstained** | n/a |
| 6 | b | Opt-in consent for marketing email | answered | Art. 7, 6, 21 |
| 7 | c | Is an IP address personal data? | answered | Art. 4 |
| 8 | c | Can an employer read work email? | **abstained** | n/a |
| 9 | c | Remedies if erasure request is ignored | answered | Art. 17, 19, 79 |

**8 of 9 behaved as intended.** The one documented exception is #6.

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
(Art. 5-8) don't offer a plausible-looking answer, so the judge correctly
abstains. #6 is harder precisely because GDPR *does* say something relevant,
just not the governing rule.

## Other observed behavior worth noting

- **#7 (IP address)** answers from Art. 4's "online identifier" language and
  states, unprompted, that the articles don't directly settle the question.
  The clearest textual basis is actually Recital 30, which is not loaded
  (recitals are excluded by design; see README).
- **Citation precision**: citations reflect articles the model's answer
  named, extracted from its own text (`extract_citations()` in `rag.py`).
  This is accurate but not infallible: an answer that quotes an article's
  internal cross-reference (e.g. "point (a) of Article 6(1)") can pull that
  article into the citation list even if the model didn't rely on it.

## Reproducing this

```
python demo_questions.py
```

Runs live against the Mistral API (9 calls) and prints bucket, outcome, and
citations for each question.

## Limitations of this evaluation itself

Nine hand-picked questions test specific behaviors on purpose; they are not
a random or statistically representative sample, and no automated scoring
(e.g. RAGAS-style faithfulness/relevance metrics) has been run. Expanding
this set (more edge cases per bucket, and a second rater) is the natural
next step before treating any pass rate here as a production SLA.
