"""Rule layer for scam patterns common in India that the (US, older) dataset barely contains."""
import re

# (name, regex, weight, explanation)
RULES = [
    ("Asks for money upfront",
     r"(registration|security|training|joining|processing|refundable|application|kit|laptop|id\s*card|verification)\s*(fee|fees|deposit|charges?|amount)"
     r"|(pay|deposit|transfer)\s+(rs\.?|₹|inr)\s*\d|refundable\s+deposit",
     0.60, "Real employers do not charge candidates to apply, train or join."),
    ("Contact only via WhatsApp/Telegram",
     r"(whats\s*app|telegram|signal)",
     0.30, "Recruiting through chat apps instead of an official channel is a common scam pattern."),
    ("Free email address for recruiter",
     r"[\w.+-]+@(gmail|yahoo|hotmail|outlook|rediffmail|ymail)\.",
     0.25, "Companies usually hire from their own domain."),
    ("Unrealistic earnings promise",
     r"earn\s*(up\s*to\s*)?(rs\.?|₹|inr|\$)?\s*\d[\d,]*\s*(/|per|a)\s*(day|week|hour)"
     r"|(guaranteed|assured)\s+(job|placement|income|salary|earning)",
     0.35, "Very high or guaranteed pay for simple work is a red flag."),
    ("Asks for sensitive documents or credentials",
     r"(aadhaar|aadhar|pan\s*card|bank\s*account|cvv|\botp\b|password|passport\s*copy)",
     0.35, "Never share ID, bank details or OTPs before a verified offer."),
    ("No interview / direct joining",
     r"(no\s+interview|without\s+interview|direct\s+joining|instant\s+joining)",
     0.25, "Skipping the interview is unusual for real jobs."),
    ("Typical fake 'online job' tasks",
     r"(copy\s*[- ]?\s*paste|form\s+filling|ad\s+posting|like\s+(and|&)\s+share|typing\s+job|data\s+entry\s+from\s+home)",
     0.30, "Simple online tasks with big pay are a classic scam."),
    ("Pressure / urgency",
     r"(limited\s+(seats|slots|vacancies)|apply\s+(immediately|now|today)|hurry|last\s+few\s+seats)",
     0.12, "Urgency is used to stop you from checking."),
]
_COMPILED = [(n, re.compile(p, re.I), w, e) for n, p, w, e in RULES]


def check_rules(raw_text: str):
    """Return (rule_score 0..1, list of triggered flags)."""
    flags, keep = [], 1.0
    for name, rx, weight, why in _COMPILED:
        m = rx.search(raw_text)
        if m:
            flags.append({"flag": name, "why": why, "match": m.group(0)[:60]})
            keep *= (1 - weight)
    return 1 - keep, flags
