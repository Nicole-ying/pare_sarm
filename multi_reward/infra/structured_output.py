"""
Structured output: JSON schema enforcement and retry logic for LLM responses.

Used by all LLM agents to ensure output conforms to expected schemas.
"""

import json
import re
from typing import Callable, Optional


def enforce_schema(response: str, schema: dict, max_retries: int = 2,
                    retry_fn: Optional[Callable] = None,
                    retry_prompt_template: str = None) -> dict:
    """Parse LLM response and validate against schema.

    If parsing or validation fails, retry with a corrective prompt.

    Args:
        response: Raw LLM response text.
        schema: Dict with expected keys and their types. E.g.:
            {"diagnosis": str, "proposed_changes": list, "changed_count": int}
        max_retries: Maximum retry attempts.
        retry_fn: Function to call for retry; receives corrective prompt,
                  returns new response. If None, raises on failure.
        retry_prompt_template: Template with {errors} placeholder for retry.

    Returns:
        Parsed and validated dict.
    """
    errors = []
    data = _parse_json(response)

    if isinstance(data, dict) and "_parse_error" in data:
        errors.append(data["_parse_error"])
    else:
        errors = _validate_against_schema(data, schema)

    # Retry loop
    for attempt in range(max_retries):
        if not errors:
            return data

        if retry_fn is None:
            break

        corrective = (
            retry_prompt_template
            or "Your previous response had issues. Fix them.\nErrors: {errors}"
        )
        corrective = corrective.format(errors="\n- ".join(errors))

        new_response = retry_fn(corrective)
        data = _parse_json(new_response)

        if isinstance(data, dict) and "_parse_error" in data:
            errors = [data["_parse_error"]]
        else:
            errors = _validate_against_schema(data, schema)

    if errors:
        data["_schema_errors"] = errors

    return data


def _parse_json(text: str) -> dict:
    """Try multiple strategies to parse JSON from text."""
    # Direct
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Code block
    m = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Regex for JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {"_parse_error": f"Failed to parse JSON ({len(text)} chars)", "_raw": text[:2000]}


def _validate_against_schema(data: dict, schema: dict) -> list[str]:
    """Validate dict against expected key-type schema. Returns list of errors."""
    errors = []
    if not isinstance(data, dict):
        return ["Output is not a dict"]

    for key, expected_type in schema.items():
        if key not in data:
            errors.append(f"Missing required key: '{key}'")
        elif not isinstance(data[key], expected_type):
            errors.append(
                f"Key '{key}': expected {expected_type.__name__}, "
                f"got {type(data[key]).__name__}"
            )

    return errors
