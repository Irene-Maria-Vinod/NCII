import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "text_clf.joblib"

class TextClassifier:
    def __init__(self):
        self.pipe = joblib.load(MODEL_PATH)

    def score(self, text: str) -> float:
        """Probability that the prompt is a harmful (NCII) request."""
        return float(self.pipe.predict_proba([text])[0][1])