import re

POLICY_VERSION = "v1.0"
BLOCK_T = 0.80    # tune these using evaluation results
REVIEW_T = 0.40
ORDER = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2}

INTIMATE = re.compile(r"\b(nude|naked|undress\w*|explicit|intimate|sexual\w*|nsfw|topless|lingerie|porn\w*)\b", re.I)
REAL_PERSON = re.compile(
    r"\b(real person|someone'?s|classmate|coworker|colleague|ex[- ]?(girlfriend|boyfriend)|"
    r"neighbou?r|stranger|celebrity|her face|his face|profile (photo|picture)|selfie|"
    r"photo(graph)? of)\b", re.I)
MANIPULATION = re.compile(r"\b(deepfake|face[- ]?swap|undress\w*|remove (her|his|their) clothes)\b", re.I)
NONCONSENT = re.compile(
    r"\b(without (their |her |his )?(consent|permission|knowing|asking|telling|approval)|"
    r"never (agreed|consented)|refused|secretly|doesn'?t know)\b", re.I)


def rule_decision(prompt: str):
    hits = []
    intimate = bool(INTIMATE.search(prompt))
    real = bool(REAL_PERSON.search(prompt))
    manip = bool(MANIPULATION.search(prompt))
    nonc = bool(NONCONSENT.search(prompt))
    if intimate: hits.append("intimate_terms")
    if real: hits.append("real_person_reference")
    if manip: hits.append("manipulation_terms")
    if nonc: hits.append("non_consent_terms")

    if intimate and (real or manip) and nonc:
        return "BLOCK", hits
    if intimate and (real or manip):
        return "REVIEW", hits
    if manip and real:
        return "REVIEW", hits
    return "ALLOW", hits


def model_decision(score: float) -> str:
    if score >= BLOCK_T: return "BLOCK"
    if score >= REVIEW_T: return "REVIEW"
    return "ALLOW"


def decide(prompt: str, score: float, consent_claimed: bool = False) -> dict:
    r_dec, hits = rule_decision(prompt)
    m_dec = model_decision(score)
    final = max(r_dec, m_dec, key=lambda d: ORDER[d])   # most severe wins

    reasons = [f"rules={r_dec}", f"model={m_dec} (score={score:.2f})"] + hits
    if consent_claimed:
        reasons.append("consent claim logged; not used to lower decision")

    return {"decision": final, "score": round(score, 3),
            "reasons": reasons, "policy_version": POLICY_VERSION}