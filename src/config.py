from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "fake_job_postings.csv"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

SEED = 42
MAX_VOCAB = 30000
MAX_LEN = 400
EMB_DIM = 100
