import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, SEED
from .text_utils import clean_text

TEXT_COLS = ["title", "department", "company_profile", "description", "requirements", "benefits"]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Put the Kaggle CSV at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    cols = [c for c in TEXT_COLS if c in df.columns]
    df["text"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1).map(clean_text)
    df = df[df["text"].str.len() > 0].drop_duplicates(subset="text").reset_index(drop=True)
    return df[["text", "fraudulent"]]


def split_dataset():
    """Stratified 70/15/15 split, identical every run (fixed seed)."""
    df = load_dataset()
    train, temp = train_test_split(df, test_size=0.30, stratify=df["fraudulent"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["fraudulent"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
