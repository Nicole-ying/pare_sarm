"""
LLM client wrapper for multi_reward framework.

Adapted from eureka_llm/framework/llm_call.py.

Supports DeepSeek API via openai client. Handles structured output
parsing with retries.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional


def create_client(api_key: str = None, base_url: str = "https://api.deepseek.com",
                  timeout: float = 120.0):
    """Create an OpenAI client configured for DeepSeek API."""
    from openai import OpenAI
    import httpx

    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set.")
    return OpenAI(
        api_key=key,
        base_url=base_url,
        http_client=httpx.Client(
            verify=False, follow_redirects=True,
            timeout=httpx.Timeout(timeout, connect=30.0),
        ),
    )


def call_llm(prompt: str, api_key: str = None, model: str = "deepseek-reasoner",
             temperature: float = 0.6, timeout: float = 120.0) -> str:
    """Call DeepSeek API with retry. Returns response text."""
    client = create_client(api_key=api_key, timeout=timeout)
    print(f"  [LLM] Calling {model} (timeout={timeout}s, temp={temperature}) ...")
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = response.choices[0].message.content
            print(f"  [LLM] Response received ({len(content)} chars)")
            return content
        except Exception as e:
            if attempt < 2:
                import time
                wait = 2 ** attempt
                print(f"  [LLM] Attempt {attempt+1} failed: {e}. Retry in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM API call failed after 3 attempts: {e}") from e


def parse_json_response(response: str, schema_hint: dict = None,
                         max_retries: int = 2) -> dict:
    """Parse LLM response text into a JSON dict.

    Tries in order:
    1. Direct JSON parse
    2. Extract from ```json code block
    3. Extract from bare JSON-like object with regex
    4. Retry: return error dict if all parsing fails

    Args:
        response: Raw LLM response text.
        schema_hint: Optional dict of expected keys for validation.
        max_retries: Unused (retry happens at caller level).

    Returns:
        Parsed dict, or {"_parse_error": "...", "_raw": "..."} on failure.
    """
    # Attempt 1: direct JSON
    try:
        data = json.loads(response.strip())
        return data
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract from ```json block
    json_match = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Attempt 3: find JSON-like structure with regex
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # All attempts failed
    return {
        "_parse_error": f"Failed to parse JSON from response ({len(response)} chars)",
        "_raw": response[:2000],
    }


def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code block from LLM response.

    Args:
        response: Raw LLM response text.

    Returns:
        Python code string, or None if no code block found.
    """
    blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if not blocks:
        return None

    # Filter for blocks containing compute_reward
    relevant = [b for b in blocks if "def compute_reward" in b]
    if relevant:
        return "\n\n".join(relevant).rstrip() + "\n"
    return blocks[0].rstrip() + "\n"
