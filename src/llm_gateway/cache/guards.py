"""Lexical guards (PLAN §6.3 point 2, R08): two questions that differ in a number, amount, date, code or
negation are different questions, however close their embeddings are. Pure functions, no model."""

import re
from collections import Counter

from llm_gateway.cache.keys import normalize

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_CURRENCY = re.compile(r"[$€£¥]|\b(?:usd|eur|mxn|gbp|cop|ars|clp|pen|brl|dólares?|dolares?|euros?|pesos?)\b")
_CODE = re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[^\W\d_])[\w][\w-]{2,}\b")
_MONTHS = (
    "enero febrero marzo abril mayo junio julio agosto septiembre setiembre octubre noviembre diciembre "
    "january february march april may june july august september october november december "
    "lunes martes miércoles miercoles jueves viernes sábado sabado domingo "
    "monday tuesday wednesday thursday friday saturday sunday "
    "hoy mañana manana ayer today tomorrow yesterday"
).split()
_DATE_WORDS = re.compile(r"\b(?:" + "|".join(_MONTHS) + r")\b")
_NEGATION = re.compile(
    r"\b(?:no|not|never|nunca|jamás|jamas|sin|ni|tampoco|nadie|ningún|ningun|ninguna|ninguno|nada|nothing|"
    r"without|none|neither|nor|cannot|cant|dont|doesnt|didnt|wont|isnt|arent)\b"
)


def _numbers(text: str) -> frozenset[str]:
    """`1,000` and `1000` are the same amount; `1.5` stays distinct from `15`."""
    out = set()
    for raw in _NUMBER.findall(text):
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", raw):
            raw = raw.replace(",", "")
        out.add(raw)
    return frozenset(out)


def _prepare(text: str) -> str:
    text = normalize(text).lower()
    text = re.sub(r"\b(can)not\b", r"\1 not", text)
    text = re.sub(r"n['’]t\b", " not", text)  # don't, can't, isn't...
    return text


def features(text: str) -> dict[str, object]:
    t = _prepare(text)
    return {
        "numbers": _numbers(t),
        "currency": frozenset(_CURRENCY.findall(t)),
        "dates": frozenset(_DATE_WORDS.findall(t)),
        "codes": frozenset(_CODE.findall(t)),
        "negation": Counter(_NEGATION.findall(t)),  # "no ... no" is not "no"
    }


def mismatch(new: str, cached: str) -> str | None:
    """The first guard that fails, by name, or None when the pair may be served from cache."""
    a, b = features(new), features(cached)
    for name in ("numbers", "currency", "dates", "codes", "negation"):
        if a[name] != b[name]:
            return name
    return None
