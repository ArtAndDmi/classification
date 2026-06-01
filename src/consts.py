from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "adult.csv"

TARGET = "income"

RANDOM_STATE = 42

TEST_SIZE = 0.2

CV_N_SPLITS = 5

N_JOBS = 2

CATEGORICAL_MISSING_VALUE = "?"
