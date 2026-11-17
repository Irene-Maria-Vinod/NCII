import sqlite3, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "audit.db"

def _conn():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS audit (
        request_id TEXT, ts TEXT, prompt_hash TEXT, decision TEXT,
        score REAL, reasons TEXT, policy_version TEXT, consent_claimed INTEGER)""")
    return c

def log_decision(request_id, prompt, result, consent_claimed):
    c = _conn()
    c.execute("INSERT INTO audit VALUES (?,?,?,?,?,?,?,?)", (
        request_id, datetime.now(timezone.utc).isoformat(),
        hashlib.sha256(prompt.encode()).hexdigest(),
        result["decision"], result["score"], json.dumps(result["reasons"]),
        result["policy_version"], int(consent_claimed)))
    c.commit(); c.close()