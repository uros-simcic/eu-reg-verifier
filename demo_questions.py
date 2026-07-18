"""A small demo/eval set for the grounded Q&A loop, in three buckets:

  (a) clearly answerable  - expect a cited answer
  (b) outside the loaded regulation - the abstain gate must fire
  (c) borderline          - tests judge precision; either outcome can be honest

Run it to see current behavior:  python demo_questions.py
This is an eval aid, not part of the app, and it makes live API calls.
"""

from rag import answer, load_resources

# (bucket, question, note on expected/observed behavior)
DEMO_SET = [
    ("a", "Do I have the right to have my personal data erased?",
     "erasure -> Art 17"),
    ("a", "What information must a company give me when it collects my personal data?",
     "transparency at collection -> Art 13"),
    ("a", "When must a personal data breach be reported to the supervisory authority?",
     "breach notification -> Art 33 (72 hours)"),
    ("a", "What is the maximum fine for infringing the GDPR?",
     "penalties -> Art 83 (EUR 20m / 4%)"),
    ("b", "Do I need to display a cookie consent banner on my website?",
     "cookie specifics are ePrivacy, not GDPR -> should abstain"),
    ("b", "Do I need opt-in consent to send a marketing newsletter?",
     "electronic marketing is ePrivacy -> known false-answer risk (answers GDPR consent angle)"),
    ("c", "Is an IP address personal data?",
     "clearest basis is Recital 30; recitals are not loaded -> answers from Art 4 and flags the gap"),
    ("c", "Can my employer read my work emails?",
     "no workplace-monitoring article -> may abstain"),
    ("c", "What happens if a company ignores my erasure request?",
     "remedies span Art 79/82/83 across chunks"),
]


def main():
    index, meta, cfg, api_key = load_resources()
    for bucket, question, note in DEMO_SET:
        result = answer(question, index, meta, cfg, api_key)
        print(f"\n[{bucket}] {question}\n     ({note})")
        if result["abstained"]:
            print("     -> ABSTAINED")
        else:
            cited = ", ".join(str(c["article"]) for c in result["citations"]) or "none"
            print(f"     -> answered; cited articles: {cited}")


if __name__ == "__main__":
    main()
