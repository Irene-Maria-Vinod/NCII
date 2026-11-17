import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from app.classifier import TextClassifier
from app.policy import rule_decision, model_decision, decide

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "ambiguous_test.csv").dropna()
clf = TextClassifier()

y_true = df["label"].tolist()
systems = {"rules_only": [], "model_only": [], "combined": []}

for p in df["prompt"]:
    s = clf.score(p)
    systems["rules_only"].append(rule_decision(p)[0])
    systems["model_only"].append(model_decision(s))
    systems["combined"].append(decide(p, s)["decision"])

for name, decs in systems.items():
    flagged = [0 if d == "ALLOW" else 1 for d in decs]   # REVIEW/BLOCK = flagged
    print(f"\n== {name} ==")
    print("precision", round(precision_score(y_true, flagged, zero_division=0), 3))
    print("recall   ", round(recall_score(y_true, flagged, zero_division=0), 3))
    print("f1       ", round(f1_score(y_true, flagged, zero_division=0), 3))
    print("review rate", round(decs.count("REVIEW") / len(decs), 3))
    print(confusion_matrix(y_true, flagged))