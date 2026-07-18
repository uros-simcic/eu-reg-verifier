# The Business Case

## Who this is for

Any organization operating in or with the EU that needs a quick, reliable
answer on what the GDPR says.

## The focus: the abstain gate

Most AI tools answer everything confidently, right or wrong. This one does
not guess. It cites the article behind an answer, or says the text doesn't
cover it. See [EVALUATION.md](EVALUATION.md).

## Why Mistral

EU-based provider for an EU-law tool. Open weights: the pipeline can run
against a self-hosted model instead of the API when data cannot leave the
buyer's infrastructure, and nothing in `rag.py` is tied to the hosted
endpoint. The judge and answer calls are small, well-defined tasks that a
compact model handles well, which keeps the cost per question low.

Not legal advice. Not novel technology: a transparent, grounded reference
implementation.
