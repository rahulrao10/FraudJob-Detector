"""Deep learning model: embedding + simple neural network (PyTorch, CPU is fine)."""
import copy
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset

from .config import ARTIFACTS, EMB_DIM, MAX_LEN, MAX_VOCAB, SEED
from .data import split_dataset
from .metrics import evaluate, save_result
from .nn_model import FraudNet
from .text_utils import N_FEATS, build_vocab, encode, text_features

BATCH, EPOCHS, PATIENCE = 64, 25, 4


def prepare(df, vocab, mean=None, std=None):
    ids = np.array([encode(t, vocab, MAX_LEN) for t in df["text"]], dtype=np.int64)
    feats = np.stack([text_features(t) for t in df["text"]])
    if mean is None:
        mean, std = feats.mean(0), feats.std(0) + 1e-6
    feats = (feats - mean) / std
    return torch.tensor(ids), torch.tensor(feats, dtype=torch.float32), mean, std


def predict_probs(model, ids, feats):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), 512):
            out.append(torch.sigmoid(model(ids[i:i + 512], feats[i:i + 512])))
    return torch.cat(out).numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ARTIFACTS.mkdir(exist_ok=True)

    train, val, test = split_dataset()
    vocab = build_vocab(train["text"], MAX_VOCAB)
    Xtr, Ftr, mean, std = prepare(train, vocab)
    Xva, Fva, _, _ = prepare(val, vocab, mean, std)
    Xte, Fte, _, _ = prepare(test, vocab, mean, std)
    ytr = torch.tensor(train["fraudulent"].values, dtype=torch.float32)
    yva = val["fraudulent"].values

    n_pos = float(ytr.sum())
    pos_weight = torch.tensor(np.sqrt((len(ytr) - n_pos) / n_pos), dtype=torch.float32)
    print(f"vocab={len(vocab)} train={len(ytr)} fraud={int(n_pos)} pos_weight={pos_weight:.2f}")

    model = FraudNet(len(vocab), EMB_DIM, N_FEATS)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(Xtr, Ftr, ytr), batch_size=BATCH, shuffle=True)

    best_ap, best_state, bad = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, fb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb, fb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        ap = average_precision_score(yva, predict_probs(model, Xva, Fva))
        print(f"epoch {epoch:2d}  train_loss={total / len(ytr):.4f}  val_PR-AUC={ap:.4f}")
        if ap > best_ap:
            best_ap, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stopping")
                break

    model.load_state_dict(best_state)

    # pick the decision threshold on the VALIDATION set (never on test)
    p, r, thr = precision_recall_curve(yva, predict_probs(model, Xva, Fva))
    f1 = 2 * p * r / (p + r + 1e-9)
    threshold = float(thr[int(np.argmax(f1[:-1]))])
    print(f"chosen threshold = {threshold:.3f}")

    probs = predict_probs(model, Xte, Fte)
    save_result(evaluate("Embedding NN (DL)", test["fraudulent"].values, (probs >= threshold).astype(int), probs))

    torch.save(model.state_dict(), ARTIFACTS / "fraud_nn.pt")
    meta = {"vocab": vocab, "mean": mean.tolist(), "std": std.tolist(), "threshold": threshold,
            "max_len": MAX_LEN, "emb_dim": EMB_DIM}
    (ARTIFACTS / "nn_meta.json").write_text(json.dumps(meta))
    print("saved artifacts/fraud_nn.pt and artifacts/nn_meta.json")


if __name__ == "__main__":
    main()
