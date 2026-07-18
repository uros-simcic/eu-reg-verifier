"""Redact structured PII from a question before it leaves the machine.

The API calls in rag.py send the user's question text to Mistral (for
embedding, judging, and answering). On a privacy-law tool, people paste real
personal data into questions -- an email, an IBAN, a card number. This strips
those patterns out locally, before any network call, so they never reach a
third-party API.

This catches structured, pattern-matchable PII only: email addresses, IBANs,
card numbers, IPv4 addresses. It does NOT catch freeform identifiers like
names or addresses -- that needs NER, not regex (e.g. Microsoft Presidio),
which pulls in ML models too heavy for this deployment's memory budget. Don't
oversell this as "anonymization" -- it's a narrow, honest first layer.
"""

import re

_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def scrub_pii(text):
    """Redact structured PII in text. Returns (clean_text, kinds_found)."""
    found = []
    clean = text
    for label, pattern in _PATTERNS.items():
        if pattern.search(clean):
            found.append(label)
            clean = pattern.sub(f"[{label} REDACTED]", clean)
    return clean, found
