"""Deterministic evidence and numerical-claim verification."""

from __future__ import annotations

import re

from smallforge.schemas import AnswerEnvelope, Evidence, ToolResult, VerificationDecision

NUMBER = re.compile(r"(?<![A-Za-z0-9_])[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?")
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
STOPWORDS = {
    "and",
    "are",
    "but",
    "for",
    "from",
    "has",
    "have",
    "into",
    "not",
    "that",
    "the",
    "their",
    "this",
    "was",
    "were",
    "with",
    "your",
    "you",
    "using",
    "result",
    "answer",
}
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


class EvidenceVerifier:
    """Checks properties that can be verified without trusting the generator."""

    def __init__(self, *, threshold: float = 0.75) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    def verify(
        self,
        answer: AnswerEnvelope,
        evidence: list[Evidence],
        tool_results: list[ToolResult],
    ) -> VerificationDecision:
        reasons: list[str] = []
        unsupported_claims: list[str] = []
        score = 1.0
        by_id = {item.source_id: item for item in evidence}
        unknown = [citation for citation in answer.citations if citation not in by_id]
        if unknown:
            reasons.append("Answer contains citation IDs that were not produced by tools.")
            unsupported_claims.extend(f"unknown citation: {citation}" for citation in unknown)
            score -= 0.6
        if evidence and not answer.citations:
            reasons.append("Evidence was available, but the answer supplied no citations.")
            score -= 0.45

        cited = [by_id[citation] for citation in answer.citations if citation in by_id]
        support_text = " ".join(item.content for item in cited)
        answer_numbers = _numeric_claims(answer.answer)
        support_numbers = _numeric_claims(support_text)
        unsupported_numbers = sorted(answer_numbers - support_numbers)
        if unsupported_numbers and evidence:
            reasons.append("One or more numerical claims do not appear in cited evidence.")
            unsupported_claims.extend(f"unsupported number: {item}" for item in unsupported_numbers)
            score -= min(0.55, 0.18 * len(unsupported_numbers))

        answer_terms = _terms(answer.answer)
        support_terms = _terms(support_text)
        if evidence and cited and len(answer_terms) >= 3 and not (answer_terms & support_terms):
            reasons.append("Answer has no substantive lexical overlap with its cited evidence.")
            unsupported_claims.append("answer/evidence lexical mismatch")
            score -= 0.4

        failed_tools = [result for result in tool_results if not result.success]
        if failed_tools:
            reasons.append(f"{len(failed_tools)} planned tool call(s) failed.")
            score -= min(0.3, 0.1 * len(failed_tools))
        if not evidence:
            reasons.append("No verifiable evidence was produced.")
            score -= 0.5

        score = min(1.0, max(0.0, score))
        passed = (
            score >= self.threshold and not unknown and not unsupported_numbers and bool(evidence)
        )
        if passed:
            reasons.append("Citations and numerical claims passed deterministic checks.")
        hint = None
        if not passed:
            hint = (
                "Use only successful tool outputs. Cite exact source_id values, remove unsupported "
                "claims, and recalculate any number absent from the cited evidence."
            )
        return VerificationDecision(
            passed=passed,
            score=round(score, 4),
            reasons=reasons,
            unsupported_claims=unsupported_claims,
            correction_hint=hint,
        )


def _normalize_number(value: str) -> str:
    return value.replace(",", "").lstrip("+")


def _numeric_claims(value: str) -> set[str]:
    claims = {_normalize_number(item) for item in NUMBER.findall(value)}
    claims.update(
        NUMBER_WORDS[token]
        for token in re.findall(r"[A-Za-z]+", value.lower())
        if token in NUMBER_WORDS
    )
    return claims


def _terms(value: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in WORD.finditer(value)
        if match.group(0).lower() not in STOPWORDS
    }
