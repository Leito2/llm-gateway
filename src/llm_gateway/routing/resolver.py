"""Turn an alias policy into the chain of links that can actually run."""

from collections.abc import Mapping

from llm_gateway.routing.config import AliasPolicy, Target


def usable_chain(policy: AliasPolicy, providers: Mapping[str, object]) -> list[Target]:
    """Links whose provider is registered, minus degraded links when the alias forbids them."""
    return [t for t in policy.chain if t.provider in providers and (policy.allow_degraded or not t.degraded)]
