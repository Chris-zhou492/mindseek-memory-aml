"""Conservative, optional credential exclusion at the public Search boundary.

Whole contaminated hits are omitted: source quotes are never silently rewritten.
This is heuristic output filtering, not storage encryption or tenant authorization.
"""

import re
import unicodedata

VERSION = "search-credential-exclusion-v1"
LABEL = (
    r"(?:password|passwd|pwd|passphrase|api[ _-]?(?:key|credential)|"
    r"(?:access|auth|authentication|refresh|recovery)[ _-]?(?:token|key|code|secret)|"
    r"client[ _-]?secret|private[ _-]?key|密码|口令|恢复(?:令牌|码)|接口密钥|访问令牌|私钥)"
)
ASSIGNMENT = re.compile(
    rf"(?<![A-Za-z0-9_]){LABEL}\s*(?:[\"']\s*)?(?:is\b|equals\b|[:=]|为|是)\s*"
    r"(?:\"(?P<double>[^\"\r\n]{1,240})\"|'(?P<single>[^'\r\n]{1,240})'|(?P<bare>[^\s,;<>\"'，。；]{1,240}))",
    re.I,
)
FIELD = re.compile(rf"^{LABEL}$", re.I)
ABSENT = {"none", "null", "unknown", "redacted", "masked", "unset", "not", "recorded", "stored", "不详", "未设置", "无"}
TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b"),
    re.compile(r"\bBearer\s+([A-Za-z0-9._~+/-]{12,}=*)", re.I),
    re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:([^\s/@]+)@", re.I),
)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----")


def normalize(value):
    return unicodedata.normalize("NFKC", value)


def is_secret(value):
    value = normalize(value).strip().rstrip(".!。")
    return bool(value) and value.casefold() not in ABSENT and not re.fullmatch(r"[*•xX]+", value)


def collect(value):
    """Return discovered values and a sensitivity flag, without changing input."""
    secrets, flagged = set(), False
    if isinstance(value, str):
        text = normalize(value)
        flagged = bool(PRIVATE_KEY.search(text))
        for match in ASSIGNMENT.finditer(text):
            candidate = next(v for v in match.groups() if v is not None).rstrip(".!。")
            if is_secret(candidate):
                secrets.add(candidate)
                flagged = True
        for pattern in TOKEN_PATTERNS:
            for match in pattern.finditer(text):
                secrets.add(match.group(1) if match.lastindex else match.group())
                flagged = True
    elif isinstance(value, dict):
        # Structured profile/event fields can detach a value from its label.
        attribute = value.get("attribute")
        if isinstance(attribute, str) and FIELD.fullmatch(normalize(attribute)):
            candidate = value.get("value")
            if type(candidate) in {str, int} and is_secret(str(candidate)):
                secrets.add(normalize(str(candidate)))
                flagged = True
        for key, child in value.items():
            if (
                isinstance(key, str)
                and FIELD.fullmatch(normalize(key))
                and type(child) in {str, int}
                and is_secret(str(child))
            ):
                secrets.add(normalize(str(child)))
                flagged = True
            found_keys, sensitive_keys = collect(key)
            secrets.update(found_keys)
            flagged |= sensitive_keys
            found, sensitive = collect(child)
            secrets.update(found)
            flagged |= sensitive
    elif isinstance(value, (list, tuple)):
        for child in value:
            found, sensitive = collect(child)
            secrets.update(found)
            flagged |= sensitive
    return secrets, flagged


def contains(value, secrets):
    if isinstance(value, str):
        text = normalize(value).casefold()
        # Boundaries avoid a short secret such as "7" matching every source ID.
        return any(re.search(r"(?<!\w)" + re.escape(secret.casefold()) + r"(?!\w)", text) for secret in secrets)
    if isinstance(value, dict):
        return any(contains(key, secrets) or contains(child, secrets) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains(child, secrets) for child in value)
    return False


def filter_credentials(items):
    """Inspect all hit fields, then exclude labelled hits and detached repeats."""
    observations = [collect(item) for item in items]
    secrets = set().union(*(found for found, _ in observations))
    return [
        item
        for item, (_, flagged) in zip(items, observations, strict=True)
        if not flagged and not contains(item, secrets)
    ]
