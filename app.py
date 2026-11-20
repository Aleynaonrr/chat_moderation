"""Real-time chat moderation demo (Gradio).

Every message is labeled live (clean / offensive) with a confidence score, flagged messages
are masked, and a side panel logs the response time of every message and lets the user
switch between the five models. The default model is the quantized (INT8) BERTurk.

Run:  python app.py   -> opens http://127.0.0.1:7860
"""
import html
import time

import gradio as gr
import pandas as pd

from src.bert_model import LATENCY_BUDGET_MS
from src.config import MODELS_DIR
from src.edit_distance import LexiconMatcher
from src.evaluate import load_models
from src.normalize import normalize, normalize_tokens

print("Loading models (about 10-20 s)...")
MODELS = {name: (predict, thr) for name, _area, _prec, predict, thr, _size in load_models()}
MODEL_NAMES = list(MODELS)
DEFAULT_MODEL = "BERTurk quantized"
LEXICON = LexiconMatcher.load(MODELS_DIR / "lexicon.json")   # used to find which words to mask
for _predict, _thr in MODELS.values():          # warm-up: the first call of a model is always slow
    for _ in range(5):
        _predict(["ısınma mesajı, warm-up message"])
LOG_COLUMNS = ["#", "Model", "Label", "Confidence", "Latency (ms)", f"<= {LATENCY_BUDGET_MS:.0f} ms"]
CMP_COLUMNS = ["Model", "Label", "P(offensive)", "Latency (ms)"]
print("Ready.")


def classify(text: str, model_name: str):
    predict, thr = MODELS[model_name]
    start = time.perf_counter()
    p_off = float(predict([text])[0])
    ms = (time.perf_counter() - start) * 1000
    is_off = p_off >= thr
    confidence = p_off if is_off else 1 - p_off
    return is_off, confidence, p_off, ms


def mask_message(text: str) -> str:
    """Replace offensive words with *****; if no single word is identified, hide the whole message."""
    words, masked = text.split(), 0
    out = []
    for w in words:
        toks = [t for t in normalize_tokens(w, fold=True) if t.isalpha() and len(t) >= 3]
        if any(LEXICON.match_word(t) for t in toks):
            out.append("*" * len(w))
            masked += 1
        else:
            out.append(w)
    return " ".join(out) if masked else "[message hidden by moderation]"


def render_chat(history: list) -> str:
    if not history:
        return "<div style='padding:24px;color:#888;text-align:center'>Type a message below to start chatting.</div>"
    bubbles = []
    for h in history:
        color = "#d93025" if h["off"] else "#188038"
        label = "OFFENSIVE" if h["off"] else "CLEAN"
        bubbles.append(
            f"<div style='margin:8px 0;padding:10px 14px;border-radius:12px;background:#f1f3f4;"
            f"border-left:5px solid {color};color:#202124'>"
            f"<div style='font-size:16px'>{html.escape(h['shown'])}</div>"
            f"<div style='font-size:12px;color:#5f6368;margin-top:4px'>"
            f"<b style='color:{color}'>{label} {h['conf']:.0%}</b> · {html.escape(h['model'])} · "
            f"{h['ms']:.1f} ms · normalized: <i>{html.escape(h['norm'])}</i></div></div>")
    return "<div style='max-height:520px;overflow-y:auto'>" + "".join(bubbles) + "</div>"


def send(text, model_name, compare, history, log):
    text = (text or "").strip()
    if not text:
        return render_chat(history), pd.DataFrame(log, columns=LOG_COLUMNS), gr.update(), "", history, log, stats(log)

    is_off, conf, p_off, ms = classify(text, model_name)
    history = history + [{
        "shown": mask_message(text) if is_off else text, "off": is_off, "conf": conf,
        "model": model_name, "ms": ms, "norm": normalize(text),
    }]
    log = log + [[len(log) + 1, model_name, "offensive" if is_off else "clean", f"{conf:.0%}",
                  round(ms, 2), "yes" if ms <= LATENCY_BUDGET_MS else "NO"]]

    cmp_df = pd.DataFrame(columns=CMP_COLUMNS)
    if compare:   # run the same message through all five models, side by side
        rows = []
        for name in MODEL_NAMES:
            o, _c, p, t = classify(text, name)
            rows.append([name, "offensive" if o else "clean", round(p, 3), round(t, 2)])
        cmp_df = pd.DataFrame(rows, columns=CMP_COLUMNS)

    return render_chat(history), pd.DataFrame(log[::-1], columns=LOG_COLUMNS), cmp_df, "", history, log, stats(log)


def stats(log) -> str:
    if not log:
        return f"**Latency budget:** {LATENCY_BUDGET_MS:.0f} ms per message (CPU)"
    lat = [r[4] for r in log]
    over = sum(1 for x in lat if x > LATENCY_BUDGET_MS)
    return (f"**Latency budget:** {LATENCY_BUDGET_MS:.0f} ms per message (CPU)  \n"
            f"Messages: {len(lat)} · mean {sum(lat) / len(lat):.1f} ms · max {max(lat):.1f} ms · over budget: {over}")


def clear():
    return render_chat([]), pd.DataFrame(columns=LOG_COLUMNS), pd.DataFrame(columns=CMP_COLUMNS), [], [], stats([])


with gr.Blocks(title="Real-Time Chat Moderation") as demo:
    gr.Markdown("# 🎮 Real-Time Chat Moderation (Turkish)\n"
                "Each message is classified live as **clean** or **offensive**. Offensive messages are masked.")
    history_state, log_state = gr.State([]), gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.HTML(render_chat([]))
            with gr.Row():
                msg = gr.Textbox(placeholder="Write a chat message and press Enter...", show_label=False, scale=5)
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear chat")
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(MODEL_NAMES, value=DEFAULT_MODEL, label="Model")
            stats_md = gr.Markdown(stats([]))
            log_df = gr.Dataframe(pd.DataFrame(columns=LOG_COLUMNS), label="Response time log (newest first)",
                                  interactive=False, wrap=True)
            compare_chk = gr.Checkbox(label="Also compare all 5 models on each message", value=False)
            cmp_df = gr.Dataframe(pd.DataFrame(columns=CMP_COLUMNS), label="Model comparison (last message)",
                                  interactive=False, wrap=True)

    inputs = [msg, model_dd, compare_chk, history_state, log_state]
    outputs = [chat, log_df, cmp_df, msg, history_state, log_state, stats_md]
    send_btn.click(send, inputs, outputs)
    msg.submit(send, inputs, outputs)
    clear_btn.click(clear, None, [chat, log_df, cmp_df, history_state, log_state, stats_md])

if __name__ == "__main__":
    demo.launch(inbrowser=True)