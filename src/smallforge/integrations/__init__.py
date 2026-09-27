"""Optional framework integrations; the benchmark core has no framework dependency."""

from .google_adk import AdkAgentBundle, AdkUnavailableError, build_adk_reference_agent

__all__ = ["AdkAgentBundle", "AdkUnavailableError", "build_adk_reference_agent"]
