"""Parse a committed EUR-Lex consolidated-HTML regulation into per-article chunks.

One chunk per article: {"reg", "article", "title", "text"}. This lives apart from
ingest.py on purpose, so the parse can be verified without making any API calls.

EUR-Lex consolidated HTML tags each article heading with two predictable classes:
    title-article-norm   -> "Article 17"
    stitle-article-norm  -> "Right to erasure (...)"
We slice the document at each heading and take everything up to the next heading
as the article body. Recitals sit before Article 1, so they fall outside every
slice and are excluded for free.
"""

import re

from bs4 import BeautifulSoup

# A line that is *only* a paragraph/point marker: "1.", "(a)", "(12)", "(iv)".
_ENUMERATOR = re.compile(r"^(?:\d+\.|\([a-z]+\)|\(\d+\)|\([ivxlcdm]+\))$", re.I)


def _join_enumerators(text: str) -> str:
    """Merge a lone marker line onto the text line that follows it.

    EUR-Lex puts each "1." / "(a)" in its own element, so get_text() leaves the
    marker on its own line. Joining it back to its content reads naturally and
    keeps each point's marker attached to its text for retrieval. Meaning is
    unchanged. A marker followed by another marker is left alone (rare edge).
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if _ENUMERATOR.match(cur) and nxt and not _ENUMERATOR.match(nxt):
            out.append(f"{cur} {nxt}")
            i += 2
        else:
            out.append(cur)
            i += 1
    return "\n".join(out)


def parse_eurlex_html(reg_id: str, html_path: str) -> list[dict]:
    """Return an ordered list of per-article chunks for one regulation."""
    with open(html_path, encoding="utf-8") as fh:
        raw = fh.read()

    # Byte offset of every article heading, plus a tail bound so the last article
    # stops before the closing "This Regulation shall be binding..." formula.
    starts = [
        raw.rfind("<", 0, m.start())
        for m in re.finditer(r'class="title-article-norm"', raw)
    ]
    tail = raw.find("This Regulation shall be binding")
    if tail == -1:
        # a Directive ends with a different formula; without a tail bound the
        # last article would swallow annexes/signatures, so make that visible
        print(f"warning: {reg_id}: closing formula not found; "
              f"last article may include trailing non-article text")
    bounds = starts + [tail if tail != -1 else len(raw)]

    chunks = []
    for i in range(len(starts)):
        seg = BeautifulSoup(raw[bounds[i] : bounds[i + 1]], "html.parser")
        number = seg.find(class_="title-article-norm").get_text(" ", strip=True)
        stitle = seg.find(class_="stitle-article-norm")
        title = stitle.get_text(" ", strip=True) if stitle else ""
        # Drop the two heading elements; whatever remains is the article body.
        for tag in seg.find_all(class_=["title-article-norm", "stitle-article-norm"]):
            tag.extract()
        body = _join_enumerators(seg.get_text("\n", strip=True))
        chunks.append(
            {
                "reg": reg_id,
                "article": int(re.search(r"\d+", number).group()),
                "title": title,
                "text": body,
            }
        )
    return chunks


if __name__ == "__main__":
    # Self-check when run directly: python parse.py [reg_id] [html_path]
    import json
    import sys

    reg = sys.argv[1] if len(sys.argv) > 1 else "gdpr"
    path = sys.argv[2] if len(sys.argv) > 2 else "regulations/gdpr/gdpr-consolidated.html"
    articles = parse_eurlex_html(reg, path)
    nums = [c["article"] for c in articles]
    ok = nums == list(range(1, nums[-1] + 1))
    print(f"{len(articles)} articles parsed; sequential 1..{nums[-1]} with no gaps: {ok}")
    print(json.dumps(articles[0], ensure_ascii=False, indent=2))
