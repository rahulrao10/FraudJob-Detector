import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.predict import Explainer, NNPredictor
from src.rules import check_rules
from src.text_utils import clean_text

app = Flask(__name__)
CORS(app)

predictor = NNPredictor()
explainer = Explainer()


def model_component(prob: float, thr: float) -> float:
    """Map the NN probability so that its tuned threshold sits at 0.5."""
    if prob < thr:
        return 0.5 * prob / thr
    return 0.5 + 0.5 * (prob - thr) / (1 - thr)


@app.get("/health")
def health():
    return jsonify(status="ok", threshold=predictor.threshold)


@app.post("/analyze")
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    raw = " ".join([data.get("title", ""), data.get("company", ""), data.get("text", "")])[:20000]
    if len(raw.strip()) < 40:
        return jsonify(error="Not enough text to analyse."), 400

    cleaned = clean_text(raw)
    prob = predictor.predict(cleaned)
    model_score = model_component(prob, predictor.threshold)
    rule_score, flags = check_rules(raw)
    final = 1 - (1 - model_score) * (1 - rule_score)  # noisy-OR: either signal can raise the risk

    level = "high" if final >= 0.7 else "medium" if final >= 0.4 else "low"
    return jsonify(
        score=round(final, 3), level=level,
        model_probability=round(prob, 3), rule_score=round(rule_score, 3),
        flags=flags, suspicious_terms=explainer.top_terms(cleaned),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
