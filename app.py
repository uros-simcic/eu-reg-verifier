"""Gradio chat UI over the grounded Q&A loop.

Chat-style thread, single-turn under the hood: the visible history is display
only; every question is answered independently and nothing from earlier turns
reaches the model. That keeps each answer an auditable unit -- one question,
one retrieval, one judged, cited (or abstained) answer.

Launches with username/password auth when DEMO_USER and DEMO_PASS are set (as
on the Hugging Face Space); without them it starts open, for local dev.
"""

import os

import gradio as gr
import requests

from rag import answer, load_resources

index, meta, cfg, api_key = load_resources()
REG_NAME = {rid: cfg["regulations"][rid]["display_name"]
            for rid in cfg["active_regulations"]}
LOADED = ", ".join(
    f'{cfg["regulations"][rid]["display_name"]} ({cfg["regulations"][rid]["full_name"]})'
    for rid in cfg["active_regulations"]
)

NOTICE = ("Ask one self-contained question at a time — each answer is "
          "independently grounded in the regulation text and cited.")
DISCLAIMER = ("Answers state what the regulation text says, with citations. "
              "This is not legal advice.")
FOLLOWUP_REPLY = (
    "This tool doesn't carry context between turns — each question is answered "
    "independently against the regulation text. Please ask a full, self-contained "
    "question (for example: “Does the GDPR give me the right to have my data "
    "deleted?”)."
)
BUSY_REPLY = ("The model API didn't respond (rate limit or a transient error). "
              "Please try again in a few seconds.")

# Openers that only make sense with earlier context. Deliberately conservative:
# a wrong "please rephrase" is annoying, but a guessed answer is off-brand.
FOLLOWUP_OPENERS = ("and ", "but ", "what about", "how about", "also ", "then ")


def looks_like_followup(q):
    lower = q.lower()
    return lower.startswith(FOLLOWUP_OPENERS) or len(lower.split()) < 3


def source_block(reg_name, article, title, text):
    """One expandable cited/retrieved article with its full source text."""
    return (f"<details><summary>{reg_name}, Article {article} — {title}"
            f"</summary>\n\n{text}\n\n</details>")


def format_reply(result):
    if result["abstained"]:
        return (f"**{result['text']}**\n\n"
                "The retrieved articles do not contain enough to answer this, "
                "so no answer is given rather than a guess.")
    parts = [result["text"]]
    if result["citations"]:
        parts.append("**Cited articles** (click to read the source text):")
        parts += [source_block(c["reg_name"], c["article"], c["title"], c["text"])
                  for c in result["citations"]]
    else:
        # the model answered without naming an article; stay transparent
        parts.append("_The answer did not cite a specific article. "
                     "Retrieved articles, for transparency:_")
        parts += [source_block(REG_NAME.get(h["reg"], h["reg"]), h["article"],
                               h["title"], h["text"])
                  for h in result["retrieved"]]
    return "\n\n".join(parts)


def respond(q):
    if looks_like_followup(q):
        return FOLLOWUP_REPLY
    try:
        return format_reply(answer(q, index, meta, cfg, api_key))
    except requests.RequestException:
        return BUSY_REPLY


def on_submit(question, history):
    q = (question or "").strip()
    if not q:
        # nothing typed; don't add an empty bubble to the thread
        return "", history or []
    history = (history or []) + [
        {"role": "user", "content": q},
        {"role": "assistant", "content": respond(q)},
    ]
    return "", history


with gr.Blocks(title="Grounded EU-regulation Q&A") as demo:
    gr.Markdown(
        "# Grounded EU-regulation Q&A\n"
        f"Answers come only from the loaded regulation text — currently **{LOADED}** — "
        "with a citation to the exact article, or the tool explicitly abstains. "
        "It never answers from model memory.\n\n"
        f"*{DISCLAIMER}*"
    )
    chatbot = gr.Chatbot(label="Q&A")
    gr.Markdown(f"**{NOTICE}**")
    question = gr.Textbox(show_label=False, submit_btn=True,
                          placeholder="e.g. Do I have the right to have my personal data erased?")
    question.submit(on_submit, [question, chatbot], [question, chatbot])

if __name__ == "__main__":
    user, password = os.environ.get("DEMO_USER"), os.environ.get("DEMO_PASS")
    # bind to the port the host injects ($PORT); fall back to 7860 for local dev
    launch = {"server_name": "0.0.0.0", "server_port": int(os.environ.get("PORT", 7860))}
    if user and password:
        demo.launch(auth=(user, password), **launch)
    else:
        print("warning: DEMO_USER/DEMO_PASS not set - launching without auth (local dev only)")
        demo.launch(**launch)
