import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "training_data.csv").drop_duplicates()
df = df.dropna()

X, y = df["prompt"], df["label"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
    ("clf", LogisticRegression(C=5, class_weight="balanced", max_iter=1000)),
])

# 5-fold cross-validation (more honest than a single split on small data)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
print(f"CV F1: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Hold-out report
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te)
print(classification_report(y_te, pred, digits=3))
print(confusion_matrix(y_te, pred))

# Final model on all data
pipe.fit(X, y)
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(pipe, ROOT / "models" / "text_clf.joblib")
print("Saved model.")