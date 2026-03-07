from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.base import BaseLanguageModel

_ROOT_TAG = "PERSONA_PROMPT"
_ALLOWED_INPUT_KEYS = ("role", "traits", "facets", "critic_internal")


class PromptFormatError(ValueError):
    """Raised when the structured prompt cannot be serialized or parsed."""


@dataclass
class OptimizationResult:
    optimized_prompt: dict[str, Any]
    optimized_prompt_text: str
    base_prompt_text: str


def _coerce_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromptFormatError(f"Field '{field_name}' must be a non-empty string")
    return value.strip()


def _normalize_statements(
    value: Any,
    field_name: str,
) -> tuple[str, list[tuple[str, str]]]:
    if isinstance(value, Mapping):
        normalized = []
        for key, text in value.items():
            key_text = _coerce_text(str(key), f"{field_name}.key")
            text_value = _coerce_text(text, f"{field_name}.{key_text}")
            normalized.append((key_text, text_value))
        if len(normalized) != 5:
            raise PromptFormatError(f"Field '{field_name}' must contain exactly 5 statements")
        return "mapping", normalized

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        normalized = []
        for idx, text in enumerate(value, start=1):
            key_text = f"item_{idx}"
            text_value = _coerce_text(text, f"{field_name}[{idx}]")
            normalized.append((key_text, text_value))
        if len(normalized) != 5:
            raise PromptFormatError(f"Field '{field_name}' must contain exactly 5 statements")
        return "sequence", normalized

    raise PromptFormatError(
        f"Field '{field_name}' must be either a mapping[str, str] or a sequence[str]"
    )


def _render_statement_block(
    block_name: str,
    kind: str,
    items: list[tuple[str, str]],
) -> str:
    lines = [f"  <{block_name} format=\"{kind}\">"]
    for key, text in items:
        safe_key = html.escape(key, quote=True)
        safe_text = html.escape(text, quote=False)
        lines.append(f"    <ITEM key=\"{safe_key}\">{safe_text}</ITEM>")
    lines.append(f"  </{block_name}>")
    return "\n".join(lines)


def serialize_persona_prompt(base_prompt: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    for key in _ALLOWED_INPUT_KEYS:
        if key not in base_prompt:
            raise PromptFormatError(f"Missing required field '{key}'")

    role = _coerce_text(base_prompt["role"], "role")
    critic_internal = _coerce_text(base_prompt["critic_internal"], "critic_internal")
    traits_kind, traits_items = _normalize_statements(base_prompt["traits"], "traits")
    facets_kind, facets_items = _normalize_statements(base_prompt["facets"], "facets")

    lines = [f"<{_ROOT_TAG}>"]
    lines.append(f"  <ROLE>{html.escape(role, quote=False)}</ROLE>")
    lines.append(_render_statement_block("TRAITS", traits_kind, traits_items))
    lines.append(_render_statement_block("FACETS", facets_kind, facets_items))
    lines.append(f"  <CRITIC_INTERNAL>{html.escape(critic_internal, quote=False)}</CRITIC_INTERNAL>")
    lines.append(f"</{_ROOT_TAG}>")

    structure = {"traits": traits_kind, "facets": facets_kind}
    return "\n".join(lines), structure


def _extract_single_tag(text: str, tag: str) -> str:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", flags=re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise PromptFormatError(f"Missing or invalid tag <{tag}>")
    value = match.group(1).strip()
    if not value:
        raise PromptFormatError(f"Tag <{tag}> must be non-empty")
    return html.unescape(value)


def _extract_statement_block(text: str, tag: str) -> tuple[str, list[tuple[str, str]]]:
    block_pattern = re.compile(
        rf"<{tag}\s+format=\"(mapping|sequence)\">\s*(.*?)\s*</{tag}>",
        flags=re.DOTALL,
    )
    block_match = block_pattern.search(text)
    if not block_match:
        raise PromptFormatError(f"Missing or invalid tag <{tag} format=\"...\">")

    fmt = block_match.group(1).strip()
    body = block_match.group(2)
    item_pattern = re.compile(r"<ITEM\s+key=\"([^\"]+)\">\s*(.*?)\s*</ITEM>", flags=re.DOTALL)
    items = [
        (html.unescape(m.group(1).strip()), html.unescape(m.group(2).strip()))
        for m in item_pattern.finditer(body)
    ]
    if len(items) != 5:
        raise PromptFormatError(f"Tag <{tag}> must contain exactly 5 <ITEM> entries")
    if any(not key or not value for key, value in items):
        raise PromptFormatError(f"Tag <{tag}> contains empty key/value in <ITEM>")
    return fmt, items


def parse_persona_prompt(text: str, structure_hint: Mapping[str, str] | None = None) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise PromptFormatError("Optimized prompt text is empty")
    root_pattern = re.compile(rf"<{_ROOT_TAG}>\s*(.*?)\s*</{_ROOT_TAG}>", flags=re.DOTALL)
    root_matches = list(root_pattern.finditer(text))
    if not root_matches:
        raise PromptFormatError(f"Missing root tag <{_ROOT_TAG}>...</{_ROOT_TAG}>")

    # Some optimizers echo the input prompt before the final answer.
    # We use the last PERSONA_PROMPT block as the candidate output.
    root_body = root_matches[-1].group(1)
    role = _extract_single_tag(root_body, "ROLE")
    critic_internal = _extract_single_tag(root_body, "CRITIC_INTERNAL")

    traits_fmt, traits_items = _extract_statement_block(root_body, "TRAITS")
    facets_fmt, facets_items = _extract_statement_block(root_body, "FACETS")

    if structure_hint is not None:
        expected_traits = structure_hint.get("traits")
        expected_facets = structure_hint.get("facets")
        if expected_traits and traits_fmt != expected_traits:
            raise PromptFormatError(
                f"Expected TRAITS format '{expected_traits}', got '{traits_fmt}'"
            )
        if expected_facets and facets_fmt != expected_facets:
            raise PromptFormatError(
                f"Expected FACETS format '{expected_facets}', got '{facets_fmt}'"
            )

    traits: dict[str, str] | list[str]
    facets: dict[str, str] | list[str]

    if traits_fmt == "mapping":
        traits = {key: value for key, value in traits_items}
    else:
        traits = [value for _, value in traits_items]

    if facets_fmt == "mapping":
        facets = {key: value for key, value in facets_items}
    else:
        facets = [value for _, value in facets_items]

    return {
        "role": role,
        "traits": traits,
        "facets": facets,
        "critic_internal": critic_internal,
    }


def _default_problem_description() -> str:
    return (
        "Improve a persona system prompt for Big Five (OCEAN) questionnaire simulation. "
        "Keep the XML block structure and all keys unchanged. "
        "Rewrite only wording quality: clarity, psychological coherence, and consistency."
    )


def _build_hype_input(prompt_text: str) -> str:
    return (
       'You are a world expert in prompt engineering for psychometric personality simulation.\n'
        'Rewrite this persona prompt so that the model reproduces the Big Five traits and facets better and more consistently.\n'
        'You can and should:\n'
        '- Change the wording to make it more natural, behavioral, and accurate.'
        '- Improve the psychological connection between ROLE, TRAITS, FACETS, and CRITIC.'
        '- Make the text clearer and less formulaic.'
        'Be sure to keep:'
        '- The exact XML structure (<PERSONA_PROMPT>, <ROLE>, <TRAITS format=...>, <FACETS>, <CRITIC_INTERNAL>)\n'
        '- All ITEM keys (openness, facet_anger, etc.)\n'
        '- Exactly 5 elements in TRAITS and exactly 5 in FACETS\n'
        '- The semantic direction of each trait/facet (do not invert the meaning)\n\n'
        'Return ONLY one block <PERSONA_PROMPT>...</PERSONA_PROMPT> and nothing else.\n\n'
        f'{prompt_text}'
    )


def optimize_persona_prompt_hype(
    base_prompt: Mapping[str, Any],
    llm: BaseLanguageModel,
    problem_description: str | None = None,
) -> OptimizationResult:
    """
    Optimize structured persona prompt blocks via CoolPrompt HyPE.

    Required input keys:
      - role: str
      - traits: mapping[str, str] or sequence[str] (exactly 5 statements)
      - facets: mapping[str, str] or sequence[str] (exactly 5 statements)
      - critic_internal: str
    """
    if llm is None:
        raise PromptFormatError("Parameter 'llm' must be a valid LangChain model")

    base_prompt_text, structure = serialize_persona_prompt(base_prompt)
    hype_query = _build_hype_input(base_prompt_text)
    desc = (problem_description or _default_problem_description()).strip()

    try:
        from coolprompt.optimizer.hype.hype import hype_optimizer
    except Exception as exc:
        raise ImportError(
            "coolprompt is not available. Install it before using HyPE optimization."
        ) from exc

    optimized_text = hype_optimizer(model=llm, prompt=hype_query, problem_description=desc)
    optimized_prompt = parse_persona_prompt(optimized_text, structure_hint=structure)
    return OptimizationResult(
        optimized_prompt=optimized_prompt,
        optimized_prompt_text=optimized_text,
        base_prompt_text=base_prompt_text,
    )
