from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.classifier import TextClassifier
from app.policy import decide
from app.audit import log_decision

app = FastAPI(title="NCII Pre-Generation Safety API", version="1.0")
clf = TextClassifier()
FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"


class CheckRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    consent_claimed: bool = False


class CheckResponse(BaseModel):
    request_id: str
    decision: str
    score: float
    reasons: list[str]
    policy_version: str


@app.post("/v1/safety/check", response_model=CheckResponse)
def check(req: CheckRequest):
    rid = str(uuid4())
    try:
        score = clf.score(req.prompt)
        result = decide(req.prompt, score, req.consent_claimed)
    except Exception:
        # Fail closed: if anything breaks, hold generation for review
        result = {"decision": "REVIEW", "score": 1.0,
                  "reasons": ["internal error: fail closed"], "policy_version": "v1.0"}
    log_decision(rid, req.prompt, result, req.consent_claimed)
    return {"request_id": rid, **result}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return FileResponse(FRONTEND)