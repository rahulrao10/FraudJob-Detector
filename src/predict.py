import json

import joblib
import numpy as np
import torch

from .config import ARTIFACTS
from .nn_model import FraudNet
from .text_utils import N_FEATS, encode, text_features


class NNPredictor:
    def __init__(self):
        meta = json.loads((ARTIFACTS / "nn_meta.json").read_text())
        self.vocab, self.max_len = meta["vocab"], meta["max_len"]
        self.mean, self.std = np.array(meta["mean"]), np.array(meta["std"])
        self.threshold = meta["threshold"]
        self.model = FraudNet(len(self.vocab), meta["emb_dim"], N_FEATS)
        self.model.load_state_dict(torch.load(ARTIFACTS / "fraud_nn.pt", map_location="cpu"))
        self.model.eval()

    def predict(self, cleaned_text: str) -> float:
        ids = torch.tensor([encode(cleaned_text, self.vocab, self.max_len)])
        feats = (text_features(cleaned_text) - self.mean) / self.std
        feats = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(self.model(ids, feats))[0])


class Explainer:
    """Uses the TF-IDF + Logistic Regression baseline to show which words pushed the score up."""

    def __init__(self):
        path = ARTIFACTS / "tfidf_lr.joblib"
        self.ok = path.exists()
        if self.ok:
            self.vec, self.clf = joblib.load(path)

    def top_terms(self, cleaned_text: str, k: int = 6):
        if not self.ok:
            return []
        X = self.vec.transform([cleaned_text]).tocsr()
        if X.nnz == 0:
            return []
        scores = X.data * self.clf.coef_[0][X.indices]
        terms = self.vec.get_feature_names_out()
        order = np.argsort(-scores)[:k]
        return [str(terms[X.indices[i]]) for i in order if scores[i] > 0]
