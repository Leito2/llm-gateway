"""R16 / R22: prompts stay out of logs unless GW_LOG_PROMPTS=true, and then only redacted by Presidio.

Without Presidio installed the gateway refuses to start in that mode: there is no way to log raw prompts.

HONEST SCOPE. Presidio's default engine needs a large spaCy model (en_core_web_lg, ~400 MB) that is not
bundled. This module runs Presidio with a blank spaCy pipeline instead, so only its PATTERN recognizers
work: email, card (Luhn), phone, IBAN, US SSN, IP address, URL, plus a Spanish DNI recognizer added here.
Person names, places and other entities that need NER are NOT detected and appear in the log as written;
install a spaCy model and replace `BlankEngine` in `_engines` to get them."""

from functools import lru_cache

from llm_gateway.settings import Settings

ENTITIES = [
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "US_SSN",
    "IP_ADDRESS",
    "URL",
    "ES_DNI",
]


class PromptLoggingUnavailable(RuntimeError):
    pass


def check_prompt_logging(settings: Settings) -> None:
    if not settings.log_prompts:
        return
    try:
        import presidio_analyzer  # noqa: F401
        import presidio_anonymizer  # noqa: F401
    except ImportError as exc:
        raise PromptLoggingUnavailable(
            "GW_LOG_PROMPTS=true needs Presidio to redact personal data, and it is not installed "
            "(uv sync --extra pii). Refusing to start rather than log raw prompts."
        ) from exc


@lru_cache(maxsize=1)
def _engines():
    import spacy
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_analyzer.nlp_engine import SpacyNlpEngine
    from presidio_anonymizer import AnonymizerEngine

    class BlankEngine(SpacyNlpEngine):
        def __init__(self) -> None:
            super().__init__(models=[{"lang_code": "en", "model_name": "blank"}])

        def load(self) -> None:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            self.nlp = {"en": nlp}

    nlp_engine = BlankEngine()
    nlp_engine.load()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp_engine)
    registry.add_recognizer(
        PatternRecognizer(
            "ES_DNI",
            patterns=[Pattern("dni", r"\b\d{8}[A-HJ-NP-TV-Z]\b", 0.6)],
            supported_language="en",
        )
    )
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])
    return analyzer, AnonymizerEngine()


def redact(text: str, max_chars: int = 500) -> str:
    """Presidio-redacted, truncated text for the access log. Only called when log_prompts is on."""
    analyzer, anonymizer = _engines()
    text = text[:max_chars]
    results = analyzer.analyze(text=text, language="en", entities=ENTITIES)
    return anonymizer.anonymize(text=text, analyzer_results=results).text
