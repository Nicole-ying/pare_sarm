"""
llm_call.py — LLM API calls with optional function-calling tool support.
"""

import argparse
import json
import os
import re
import sys
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
    """Call DeepSeek API and return clean response text."""
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


def call_llm_with_tools(
    system_prompt: str,
    user_prompt: str,
    tool_schemas: list[dict],
    tool_executor,
    api_key: str,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    timeout: float = 600.0,
    max_rounds: int = 8,
) -> str:
    """Call LLM with function-calling tools (ReAct agent loop).

    The model can call tools, see results, and decide next steps.
    The tool_executor(tool_name, arguments_dict) -> str callback handles execution.

    Returns the final text response.
    """
    client = _get_client(api_key, timeout)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print(f"  [LLM-Tools] Starting with {len(tool_schemas)} tools (model={model})")

    for rnd in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
            temperature=temperature,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            content = msg.content or ""
            print(f"  [LLM-Tools] Done ({len(content)} chars)")
            return content

        print(f"  [LLM-Tools] Round {rnd + 1}: {len(msg.tool_calls)} tool call(s)")

        # Record assistant message with tool_calls
        tool_call_records = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_call_records.append({
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": tool_call_records,
        })

        # Execute each tool and feed results back
        for tc, record in zip(msg.tool_calls, tool_call_records):
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            print(f"    -> {tc.function.name}({str(args)[:100]})")
            result = tool_executor(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)[:4000],
            })

    print(f"  [LLM-Tools] Max rounds reached, forcing final answer")
    messages.append({"role": "user", "content": "Provide your final answer now."})
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _clean_response(message) -> str:
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning and content.startswith(reasoning):
        content = content[len(reasoning):].strip()
    content = _strip_reasoning_tags(content)
    return content


def _strip_reasoning_tags(text: str) -> str:
    """Remove XML-style reasoning/thinking tags that may leak into LLM responses."""
    for tag in ("think", "thinking", "reasoning", "analysis", "cot"):
        text = re.sub(rf"<{tag}>[\s\S]*?</{tag}>", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_reward_fn(response_text: str) -> str:
    blocks = re.findall(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    combined = []
    for block in blocks:
        if "def compute_reward" in block:
            combined.append(block.rstrip())
    if not combined:
        raise ValueError("No ```python block with compute_reward found.")
    return "\n\n".join(combined) + "\n"


def extract_analysis(response_text: str) -> str:
    cleaned = re.sub(r"```python\s*\n.*?```", "", response_text, flags=re.DOTALL)
    cleaned = re.sub(r"```\s*\n.*?```", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def save_artifacts(output_dir: Path, prompt: str, response: str,
                   analysis: str = None, code: str = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "response.md").write_text(response, encoding="utf-8")
    if analysis:
        (output_dir / "analysis.md").write_text(analysis, encoding="utf-8")
    if code:
        header = f'"""LLM-generated reward function.\nSource: {output_dir.name}\n"""\n\nimport math\nimport numpy as np\n\n'
        (output_dir / "reward_fn_source.py").write_text(header + code + "\n", encoding="utf-8")
        print(f"  Code saved -> {output_dir / 'reward_fn_source.py'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    prompt = Path(args.prompt).read_text(encoding="utf-8")
    print(f"Calling LLM ({args.model}) ...")
    response = call_llm(prompt, api_key, args.model, args.temperature)
    save_artifacts(Path(args.output_dir), prompt, response,
                   code=extract_reward_fn(response))
