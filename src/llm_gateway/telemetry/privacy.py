"""R16 / R22: prompts stay out of logs unless GW_LOG_PROMPTS=true, and then only redacted by Presidio.

Without Presidio installed the gateway refuses to start in that mode: there is no way to log raw prompts."""

from llm_gateway.settings import Settings


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


def redact(text: str, max_chars: int = 500) -> str:
    """Presidio-redacted, truncated text for the access log. Only called when log_prompts is on."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    results = AnalyzerEngine().analyze(text=text[:max_chars], language="en")
    return AnonymizerEngine().anonymize(text=text[:max_chars], analyzer_results=results).text
