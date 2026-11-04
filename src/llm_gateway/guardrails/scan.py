"""Prompt-injection and PII detection at the edge (stretch, PLAN 8.4).

HONEST SCOPE: PLAN names Llama Prompt Guard 2 (ONNX) and Presidio. Neither is installed here, so this module
is a regex stand-in behind the same interface: English and Spanish injection phrases, and a few PII shapes
(email, phone, card with Luhn check, Spanish DNI, US SSN, IBAN). It misses paraphrases and obfuscation that a
model would catch, and recall/false positives (G14) are NOT measured. Hence the default mode is `monitor`:
it counts and flags, it never blocks. `enforce` blocks injection only; PII is never blocked, because a
user may legitimately send their own email address."""

import re
from dataclasses import dataclass

from llm_gateway.api.schemas import ChatMessage
from llm_gateway.pricing.tokens import content_text

_GAP = r"[^.\n]{0,30}"
INJECTION = [
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"\b(ignore|disregard|forget|override)\b{_GAP}\b(previous|prior|above|earlier|all|any|your)\b"
        rf"{_GAP}\b(instructions?|prompts?|rules?|guidelines?)\b",
        rf"\b(reveal|show|print|repeat|leak)\b{_GAP}\b(system|hidden|initial)\s+(prompt|message|instructions?)\b",
        rf"\byou\s+are\s+now\b{_GAP}\b(dan|unrestricted|unfiltered|jailbroken|developer mode)\b",
        r"\b(developer|god)\s+mode\b",
        r"\bdo\s+anything\s+now\b",
        rf"\b(ignora|olvida|descarta)\w*\b{_GAP}\b(instrucciones|indicaciones|reglas)\b"
        rf"{_GAP}\b(anteriores|previas|de arriba|del sistema)\b",
        rf"\b(ignora|olvida)\b{_GAP}\b(todo|todas|tus)\b{_GAP}\b(instrucciones|reglas)\b",
        rf"\b(revela|muestra|imprime|repite)\b{_GAP}\b(prompt|instrucciones)\s+(del\s+)?(sistema|inicial)",
        r"\bmodo\s+(desarrollador|dios|sin\s+restricciones)\b",
        rf"\bahora\s+eres\b{_GAP}\b(sin\s+restricciones|sin\s+filtros|dan)\b",
    )
]

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d ().-]{8,14}\d(?!\w)")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_DNI = re.compile(r"\b\d{8}[A-HJ-NP-TV-Z]\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,4})?\b")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _has_card(text: str) -> bool:
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            return True
    return False


def pii_kinds(text: str) -> set[str]:
    found: set[str] = set()
    if _EMAIL.search(text):
        found.add("email")
    card = _has_card(text)
    if card:
        found.add("card")
    if _DNI.search(text):
        found.add("dni")
    if _SSN.search(text):
        found.add("ssn")
    if _IBAN.search(text):
        found.add("iban")
    if not card and _PHONE.search(text):
        found.add("phone")
    return found


@dataclass(frozen=True)
class Findings:
    injection: bool = False
    pii: frozenset[str] = frozenset()

    @property
    def flags(self) -> list[str]:
        return [*(["injection"] if self.injection else []), *(f"pii:{k}" for k in sorted(self.pii))]


def scan(messages: list[ChatMessage], untrusted: bool = False) -> Findings:
    """User messages always; every non-system message when the caller marked the content as untrusted
    (`X-Untrusted-Content: true`, for example web pages pasted in by a RAG pipeline)."""
    injection, pii = False, set()
    for m in messages:
        if m.role == "system" or (m.role != "user" and not untrusted):
            continue
        text = content_text(m.content)
        injection = injection or any(p.search(text) for p in INJECTION)
        pii |= pii_kinds(text)
    return Findings(injection, frozenset(pii))
