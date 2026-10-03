"""Text cleaning, tokenizing and hand-made features.
Used by BOTH training and the Flask API so the two always agree."""
import html
import math
import re
from collections import Counter

import numpy as np

_TAG = re.compile(r"<[^>]+>")
_PH_URL = re.compile(r"#URL_[A-Za-z0-9]+#")
_PH_EMAIL = re.compile(r"#EMAIL_[A-Za-z0-9]+#")
_PH_PHONE = re.compile(r"#PHONE_[A-Za-z0-9]+#")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\d{10}(?!\d)")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")

FEATURE_NAMES = ["log_len", "exclaims", "upper_ratio", "has_email", "has_url", "has_phone"]
N_FEATS = len(FEATURE_NAMES)


def clean_text(text: str) -> str:
    """Strip HTML and map emails/urls/phones (real or dataset placeholders) to shared tokens."""
    text = html.unescape(str(text))
    text = _TAG.sub(" ", text)
    text = _PH_URL.sub(" urltoken ", text)
    text = _PH_EMAIL.sub(" emailtoken ", text)
    text = _PH_PHONE.sub(" phonetoken ", text)
    text = _EMAIL.sub(" emailtoken ", text)
    text = _URL.sub(" urltoken ", text)
    text = _PHONE.sub(" phonetoken ", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str):
    return _TOKEN.findall(text.lower())


def text_features(cleaned: str) -> np.ndarray:
    letters = [c for c in cleaned if c.isalpha()]
    upper_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
    low = cleaned.lower()
    return np.array([
        math.log1p(len(cleaned.split())),
        min(cleaned.count("!"), 20),
        upper_ratio,
        float("emailtoken" in low),
        float("urltoken" in low),
        float("phonetoken" in low),
    ], dtype=np.float32)


def build_vocab(texts, max_vocab: int):
    counts = Counter()
    for t in texts:
        counts.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int):
    ids = [vocab.get(tok, 1) for tok in tokenize(text)][:max_len]
    if not ids:
        ids = [1]
    return ids + [0] * (max_len - len(ids))
