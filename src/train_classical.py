"""Baselines: TF-IDF + Naive Bayes / Linear SVM / Logistic Regression."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from .config import ARTIFACTS
from .data import split_dataset
from .metrics import evaluate, save_result


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    train, _, test = split_dataset()
    print(f"train={len(train)} test={len(test)} fraud in train={int(train.fraudulent.sum())}")

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(train["text"])
    Xte = vec.transform(test["text"])

    models = {
        "Naive Bayes": ComplementNB(),
        "Linear SVM": LinearSVC(class_weight="balanced"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    }
    for name, clf in models.items():
        clf.fit(Xtr, train["fraudulent"])
        pred = clf.predict(Xte)
        score = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xte)
        save_result(evaluate(name, test["fraudulent"], pred, score))
        if name == "Logistic Regression":  # also used to explain flagged words in the extension
            joblib.dump((vec, clf), ARTIFACTS / "tfidf_lr.joblib")


if __name__ == "__main__":
    main()
