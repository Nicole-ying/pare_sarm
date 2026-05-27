"""
env_perception_agent.py — Pre-training agent that reads source code and exploration
data to build a structured Task Manifest.

Runs once before Round 0. Output is saved to memory/TASK_MANIFEST.md.
"""

import json
import re
import sys
from pathlib import Path

_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
from llm_call import call_llm


def _load_file_store(env_dir: Path, task_description: str,
                     exploration_path: Path) -> dict:
    """Load all available files into a dict keyed by file_key."""
    store = {}
    store["task_description"] = task_description or None

    step_py = env_dir / "step.py"
    store["step_source"] = step_py.read_text("utf-8") if step_py.exists() else None

    env_py = env_dir / "env.py"
    store["env_source"] = env_py.read_text("utf-8") if env_py.exists() else None

    if exploration_path and exploration_path.exists():
        store["exploration"] = exploration_path.read_text("utf-8")

    return store


# Per-file truncation limits (chars). env.py is typically very large
# (793 lines, 29KB for LunarLander) so we give it more budget.
_file_truncate_limits = {
    "env_source": 20000,
}
_default_truncate_limit = 10000


def _detect_file_request(text: str, valid_keys: set) -> str | None:
    """Detect which file key the LLM wants to read, using lenient matching."""
    if not text:
        return None

    # 1) Exact function call: read_file("key") or read_file('key')
    for q in ['"', "'"]:
        m = re.search(r'read_file\s*\(\s*' + q + r'(\w+)' + q + r'\s*\)', text)
        if m and m.group(1) in valid_keys:
            return m.group(1)

    # 2) Natural language: "read the step_source" / "read step_source" / "let me look at env_source"
    m = re.search(r'(?:read|look at|examine|check|open|show)\s+(?:the\s+)?(`)?(\w+)(?(1)`|)', text, re.IGNORECASE)
    if m and m.group(2) in valid_keys:
        return m.group(2)

    # 3) Just the bare key name as a word (if it's a substantial response)
    words = set(re.findall(r'\b\w+\b', text.lower()))
    found = words & valid_keys
    if found and len(text) > 20:
        # Only trust bare key if it's clearly part of a reading intent
        if any(w in text.lower() for w in ["read", "look", "examine", "check", "start with", "first", "next"]):
            return found.pop()

    return None


def _extract_fallback_info(file_store: dict) -> dict:
    """Best-effort extraction from file store for the fallback manifest.

    Returns enough structured data for _build_generic_fallback_manifest()
    to produce a complete TaskManifest without any env-specific hardcoding.
    """
    info = {}

    env_source = file_store.get("env_source", "")
    if env_source:
        # Extract class name — prefer the class that extends gym.Env
        m = re.search(r'class\s+(\w+)\s*\(.*gym\.Env.*\):', env_source)
        if not m:
            m = re.search(r'class\s+(\w+)\s*\(.*Env.*\):', env_source)
        if not m:
            m = re.search(r'class\s+(\w+)\s*(?:\(.*?\))?:', env_source)
        info["class_name"] = m.group(1) if m else "Unknown"

        # Detect action space type and shape
        if "spaces.Discrete" in env_source:
            info["action_space_type"] = "Discrete"
            m = re.search(r'spaces\.Discrete\s*\(\s*(\d+)\s*\)', env_source)
            info["action_n"] = int(m.group(1)) if m else None
        elif "spaces.Box" in env_source:
            info["action_space_type"] = "Continuous Box"
            m = re.search(r'action_space\s*=\s*spaces\.Box\s*\([^,]+,\s*[^,]+,\s*\(([^)]+)\)', env_source)
            if m:
                info["action_shape"] = m.group(1).strip()
        else:
            info["action_space_type"] = "Unknown"

        # Extract observation space bounds
        m = re.search(r'observation_space\s*=\s*spaces\.Box\s*\(\s*low\s*=\s*np\.array\(\[(.*?)\]\)', env_source, re.DOTALL)
        if m:
            low_vals = [v.strip() for v in m.group(1).replace('\n', '').split(',')]
            info["obs_low"] = [float(v) if v.replace('.','').replace('-','').isdigit() else v for v in low_vals]
        m = re.search(r'high\s*=\s*np\.array\(\[(.*?)\]\)', env_source, re.DOTALL)
        if m:
            high_vals = [v.strip() for v in m.group(1).replace('\n', '').split(',')]
            info["obs_high"] = [float(v) if v.replace('.','').replace('-','').isdigit() else v for v in high_vals]

        # Count observation dimensions from Box definition
        m = re.search(r'spaces\.Box\s*\(\s*low[^)]+shape\s*=\s*\((\d+),?\s*\)', env_source)
        if not m:
            m = re.search(r'np\.array\(\[(.*?)\][^)]*\).*np\.array\(\[(.*?)\]', env_source, re.DOTALL)
            if m:
                info["obs_dim"] = len(m.group(1).split(','))
        else:
            info["obs_dim"] = int(m.group(1))

    step_source = file_store.get("step_source", "")
    if step_source:
        m = re.search(r'reward,\s*components\s*=\s*self\.compute_reward\(([^)]+)\)', step_source)
        if not m:
            m = re.search(r'self\.compute_reward\(([^)]+)\)', step_source)
        info["reward_signature"] = m.group(1) if m else "state, action, terminated"

    exploration = file_store.get("exploration", "")
    if exploration:
        try:
            exp = json.loads(exploration)
            info["obs_dim_stats"] = exp.get("obs_dim_stats", [])
            info["termination_summary"] = exp.get("termination_summary", {})
            info["episode_length_stats"] = exp.get("episode_length_stats", {})
            info["zero_action"] = exp.get("zero_action", {})
            info["max_episode_steps"] = exp.get("max_episode_steps", 1000)
        except json.JSONDecodeError:
            info["obs_dim_stats"] = []
            info["termination_summary"] = {}

    return info


def _build_generic_fallback_manifest(info: dict, task_description: str) -> str:
    """Build a generic Task Manifest from auto-discovered data.

    No env-specific hardcoding. Every field is populated from parsed source
    code or exploration data. When meaning cannot be inferred, dimensions
    are labeled as 'unknown' — the LLM will interpret them from context.
    """
    parts = ["# Task Manifest\n"]

    # Task Goal
    goal = task_description or "See environment source code and exploration data."
    parts.append("## Task Goal")
    parts.append(goal)
    parts.append("")

    # Success / Failure conditions
    term = info.get("termination_summary", {})
    if term:
        parts.append("## Termination Conditions (from exploration)")
        for reason, detail in term.items():
            if isinstance(detail, dict):
                parts.append(f"- `{reason}`: {detail.get('count', '?')} / {detail.get('fraction', '?')} episodes")
            else:
                parts.append(f"- `{reason}`: {detail}")
    else:
        parts.append("## Termination Conditions")
        parts.append("- (unknown — see step source code)")
    parts.append("")

    # Observation dimensions from exploration + space bounds
    parts.append("## Key Observation Dimensions")
    parts.append("| Dim | Observed Range (random policy) | Space Bounds |")
    parts.append("|-----|-------------------------------|-------------|")

    obs_stats = info.get("obs_dim_stats", [])
    if obs_stats:
        for s in obs_stats:
            dim = s.get("dim", "?")
            s_min = s.get("sample_min", "?")
            s_max = s.get("sample_max", "?")
            sp_low = s.get("space_low", "?")
            sp_high = s.get("space_high", "?")
            parts.append(f"| {dim} | [{s_min}, {s_max}] | [{sp_low}, {sp_high}] |")
    else:
        parts.append("| — | — | — |")
    parts.append("")
    parts.append("Note: Dimension meanings are not auto-labeled. Infer them from the env source code and observation space bounds. Look for patterns: [-π, π] suggests angles, [0, 1] suggests contact/normalized values, large ranges suggest velocities/positions.")
    parts.append("")

    # Critical dimensions hint
    parts.append("## Critical Dimensions for Reward Design")
    parts.append("Identify 2-4 observation dimensions most critical for task success by reading the environment source code. For each, explain what physical quantity it measures and what type of reward gradient it requires.")
    parts.append("")

    # Action space
    act_type = info.get("action_space_type", "Unknown")
    parts.append("## Action Space")
    parts.append(f"Type: {act_type}")
    if info.get("action_n"):
        parts.append(f"Discrete actions: {info['action_n']} (0..{info['action_n'] - 1})")
    if info.get("action_shape"):
        parts.append(f"Continuous dimensions: {info['action_shape']}")
    parts.append("")

    # Compute reward signature (from step.py — authoritative)
    sig = info.get("reward_signature", "state, action, terminated")
    parts.append("## Compute Reward Signature")
    parts.append(f"`compute_reward({sig})`")
    parts.append("This is extracted from step.py — it is the exact argument list the environment passes to compute_reward.")

    return "\n".join(parts)


def _extract_manifest(response: str) -> str | None:
    """Extract Task Manifest markdown from FINAL ANSWER or # Task Manifest header.

    Uses multiple strategies in order of reliability:
    1. Find FINAL ANSWER marker → locate # Task Manifest → extract from there
    2. Find FINAL ANSWER marker → extract content starting with #
    3. Find # Task Manifest anywhere (case-insensitive)
    """
    text = response.strip()
    if not text:
        return None

    # Strategy 1: FINAL ANSWER marker → locate # Task Manifest in the text after it
    idx = text.upper().rfind("FINAL ANSWER")
    if idx != -1:
        after = text[idx + len("FINAL ANSWER"):].strip()
        # Look for # Task Manifest header in the after-text
        manifest_match = re.search(r'(#\s*Task\s+Manifest[\s\S]*)', after, re.IGNORECASE)
        if manifest_match:
            return manifest_match.group(1).strip()
        # Fallback: if after-text starts with a # header, use it
        if after.startswith("#"):
            return after

    # Strategy 2: Find # Task Manifest anywhere (case-insensitive)
    manifest_match = re.search(r'(#\s*Task\s+Manifest[\s\S]*)', text, re.IGNORECASE)
    if manifest_match:
        return manifest_match.group(1).strip()

    return None


def run_env_perception_agent(env_dir: Path, task_description: str,
                              exploration_path: Path, api_key: str,
                              model: str = "deepseek-reasoner",
                              temperature: float = 0.3,
                              memory_system=None) -> str:
    """Run env perception: autonomous ReAct loop.

    The agent decides which files to read and in what order, building up
    understanding step by step, then produces a structured Task Manifest.

    Args:
        env_dir: Environment directory (contains step.py, env.py).
        task_description: Task goal text (from env_descriptions/*.md).
        exploration_path: Path to exploration JSON.
        api_key: LLM API key.
        model: LLM model name.
        temperature: Sampling temperature.
        memory_system: MemorySystem instance (for saving manifest).

    Returns:
        Task Manifest markdown string.
    """
    template_path = (Path(__file__).resolve().parent.parent.parent /
                     "templates" / "env_perception_prompt.txt")
    system_msg = template_path.read_text("utf-8") if template_path.exists() else ""

    file_store = _load_file_store(env_dir, task_description, exploration_path)
    valid_keys = {k for k, v in file_store.items() if v is not None}
    required_files = {k for k in ["step_source", "env_source"] if k in valid_keys}
    if "exploration" in valid_keys:
        required_files.add("exploration")

    conversation_history = []
    read_files = set()

    max_steps = 12
    final_response = None
    idle_rounds = 0

    for step in range(max_steps):
        parts = [system_msg]

        if step == 0:
            parts.append(
                f"\n\nAvailable files: {sorted(valid_keys)}\n\n"
                "Which file would you like to read first? "
                "Say the name (e.g. 'task_description') or use read_file(\"key\")."
            )
        else:
            for msg in conversation_history:
                parts.append(f"\n\n{msg['role'].title()}: {msg['content']}")

            parts.append("\n\nAssistant:")

        full_prompt = "".join(parts)

        try:
            response = call_llm(full_prompt, api_key, model, temperature)
        except Exception as e:
            print(f"  [EnvPerception] LLM call failed at step {step}: {e}")
            break

        conversation_history.append({"role": "assistant", "content": response})

        # Check for FINAL ANSWER
        if "FINAL ANSWER" in response.upper():
            # Verify FINAL ANSWER is genuine: deepseek-reasoner sometimes
            # outputs FINAL ANSWER and then continues asking for more files.
            idx = response.upper().rfind("FINAL ANSWER")
            after_final = response[idx + len("FINAL ANSWER"):].strip()
            if not _detect_file_request(after_final, valid_keys):
                # Enforce required files before allowing final answer
                missing_required = required_files - read_files
                if missing_required:
                    print(f"  [EnvPerception] FINAL ANSWER without reading: {sorted(missing_required)}")
                    prompt = (
                        f"You need to read these files first: {sorted(missing_required)}.\n"
                        f"Call read_file(\"{next(iter(missing_required))}\") before giving FINAL ANSWER."
                    )
                    conversation_history.append({"role": "user", "content": prompt})
                    continue
                final_response = response
                print(f"  [EnvPerception] Final answer received (step {step + 1})")
                break
            else:
                print(f"  [EnvPerception] FINAL ANSWER but followed by file request — continuing (step {step + 1})")

        # Auto-detect Task Manifest output — the LLM often writes
        # the manifest without the FINAL ANSWER keyword, especially
        # deepseek-reasoner which drops trailing instructions.
        if response.strip().startswith("# Task Manifest"):
            # Also verify it doesn't contain a file request
            if not _detect_file_request(response, valid_keys):
                # Enforce required files before allowing auto-detect
                missing_required = required_files - read_files
                if missing_required:
                    print(f"  [EnvPerception] Task Manifest without reading: {sorted(missing_required)}")
                    prompt = (
                        f"You need to read these files first: {sorted(missing_required)}.\n"
                        f"Call read_file(\"{next(iter(missing_required))}\") before outputting the manifest."
                    )
                    conversation_history.append({"role": "user", "content": prompt})
                    continue
                final_response = response
                print(f"  [EnvPerception] Task Manifest detected without FINAL ANSWER (step {step + 1})")
                break

        # Detect file request
        file_key = _detect_file_request(response, valid_keys)

        if file_key and file_key not in read_files:
            idle_rounds = 0
            read_files.add(file_key)
            content = file_store.get(file_key)
            print(f"  [EnvPerception] Reading: {file_key} (step {step + 1})")

            if content:
                limit = _file_truncate_limits.get(file_key, _default_truncate_limit)
                truncated = content[:limit]
                user_msg = f"### File: {file_key}\n\n```\n{truncated}\n```\n\nWhat do you learn from this file? Then tell me which file to read next, or output FINAL ANSWER if you have enough information."
                if len(content) > limit:
                    user_msg += f"\n\n(File truncated to {limit} chars. Original: {len(content)} chars.)"
            else:
                user_msg = f"### File: {file_key}\n\n(File not found or empty.)"
            conversation_history.append({"role": "user", "content": user_msg})
        else:
            idle_rounds += 1

            if file_key and file_key in read_files:
                prompt = f"You already read {file_key}. Pick another from: {sorted(valid_keys - read_files)}"
            elif read_files:
                prompt = f"Read another file from {sorted(valid_keys - read_files)} or output FINAL ANSWER."
            else:
                prompt = f"Say a file name from {sorted(valid_keys)} to start reading."

            if idle_rounds >= 3 and not read_files:
                # Force-feed the first file after 3 idle rounds
                first_key = next(iter(valid_keys))
                read_files.add(first_key)
                content = file_store.get(first_key)
                print(f"  [EnvPerception] Force-reading: {first_key} (step {step + 1}, {idle_rounds} idle)")
                limit = _file_truncate_limits.get(first_key, _default_truncate_limit)
                prompt = f"### File: {first_key}\n\n```\n{content[:limit]}\n```\n\nAnalyze this file. Then tell me which file to read next or output FINAL ANSWER."
            elif idle_rounds >= 2 and read_files:
                prompt = f"You haven't made a file request in a while. Read another file from {sorted(valid_keys - read_files)} or output FINAL ANSWER with the complete Task Manifest."

            conversation_history.append({"role": "user", "content": prompt})

    # Save conversation for debugging
    if memory_system and memory_system.run_dir:
        try:
            conv_path = memory_system.run_dir / "env_perception_conversation.json"
            conv_path.parent.mkdir(parents=True, exist_ok=True)
            conv_path.write_text(
                json.dumps(conversation_history, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    # Extract Task Manifest
    last_response = final_response or (conversation_history[-1]["content"] if conversation_history else "")
    manifest = _extract_manifest(last_response)

    if not manifest:
        for msg in reversed(conversation_history):
            if msg["role"] == "assistant":
                manifest = _extract_manifest(msg["content"])
                if manifest:
                    break

    if not manifest:
        print(f"  [EnvPerception] No valid Task Manifest extracted. Building generic fallback.")
        info = _extract_fallback_info(file_store)
        manifest = _build_generic_fallback_manifest(info, task_description)
        print(f"  [EnvPerception] Generic fallback used. {len(conversation_history)} turns in history.")

    if memory_system:
        memory_system.save_task_manifest(manifest)
        print(f"  [EnvPerception] Task Manifest saved → {memory_system.task_manifest_path}")

    print(f"  [EnvPerception] Files read: {sorted(read_files)}")
    return manifest
