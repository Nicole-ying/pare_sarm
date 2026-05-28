"""LLM API calls via DeepSeek (OpenAI-compatible)."""

import re
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _get_client(api_key: str, timeout: float = 600.0):
    import httpx
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        http_client=httpx.Client(verify=False, follow_redirects=True, timeout=timeout),
    )


def call_llm(prompt: str, api_key: str, model: str = "deepseek-reasoner",
             temperature: float = 0.6, timeout: float = 600.0) -> str:
    """Call DeepSeek API and return cleaned response text."""
    client = _get_client(api_key, timeout)
    print(f"  [LLM] Calling {model} (temp={temperature}) ...")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    content = _clean_response(response.choices[0].message)
    print(f"  [LLM] Response ({len(content)} chars)")
    return content


def _clean_response(message) -> str:
    """Strip reasoning content and XML tags from LLM response."""
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning and content.startswith(reasoning):
        content = content[len(reasoning):].strip()
    content = _strip_reasoning_tags(content)
    return content


def _strip_reasoning_tags(text: str) -> str:
    """Remove XML-style reasoning/thinking tags."""
    for tag in ("think", "thinking", "reasoning", "analysis", "cot"):
        text = re.sub(rf"<{tag}>[\s\S]*?</{tag}>", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_reward_fn(response: str) -> str:
    """Extract Python code block(s) containing `def compute_reward`."""
    blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
    combined = []
    for block in blocks:
        if "def compute_reward" in block:
            combined.append(block.rstrip())
    if not combined:
        raise ValueError("No ```python block with compute_reward found.")
    return "\n\n".join(combined) + "\n"


def extract_code_block(response: str, func_name: str = "compute_reward") -> str:
    """Extract a specific Python function from an LLM response."""
    blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
    for block in blocks:
        if f"def {func_name}" in block:
            return block.rstrip() + "\n"
    raise ValueError(f"No ```python block with def {func_name} found.")


def compile_and_check(code: str) -> tuple[bool, str]:
    """Try to compile Python code. Returns (ok, error_message)."""
    try:
        compile(code, "<check>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def save_artifacts(output_dir: Path, prompt: str, response: str,
                   code: str = None) -> None:
    """Save prompt, response, and optional reward code to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "response.txt").write_text(response, encoding="utf-8")
    if code:
        header = f'"""LLM-generated reward function.\n"""\n\nimport math\nimport numpy as np\n\n'
        (output_dir / "reward_fn_source.py").write_text(header + code + "\n", encoding="utf-8")
