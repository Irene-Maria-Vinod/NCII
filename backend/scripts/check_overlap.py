import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
tr = pd.read_csv(ROOT / "data" / "training_data.csv")["prompt"].str.strip().str.lower()
te = pd.read_csv(ROOT / "data" / "ambiguous_test.csv")["prompt"].str.strip().str.lower()
overlap = set(tr) & set(te)
print("Overlapping prompts:", len(overlap))
for p in overlap: print(" -", p)