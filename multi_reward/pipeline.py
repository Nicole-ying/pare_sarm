#!/usr/bin/env python3
"""
pipeline.py — Multi-agent reward design orchestrator (MMCP).

Modes: round0 | iterate | continue | full
"""
import argparse, json, os, re, shutil, subprocess, sys, threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import perf_counter

_wd = Path(__file__).resolve().parent
if str(_wd) not in sys.path: sys.path.insert(0, str(_wd))

from infra.llm_client import call_llm, extract_code_from_response
from infra.config_loader import load_yaml
from infra.file_utils import ensure_dir, save_json, load_json, load_text, save_text, experiment_dir_name
from infra.logging_setup import setup_logging
from evidence.evidence_analyzer import EvidenceAnalyzer
from env_interpreter.env_interpreter import EnvInterpreter
from diagnostician.diagnostician import run_diagnostician
from moderator.moderator import run_moderator_phase1, run_moderator_phase2
from implementor.implementor import Implementor, apply_edits
from code_validator.code_validator import validate_code
from memory.memory_store import MemoryStore

BEIJING = timezone(timedelta(hours=8))

def _run_subprocess(cmd: list) -> subprocess.CompletedProcess:
    """Run subprocess with stdout/stderr inherited (no pipe, no buffer blocking)."""
    return subprocess.run(cmd)

def _dump_config(cfg):
    try: import yaml; return yaml.safe_dump(cfg, sort_keys=False)
    except ImportError: return json.dumps(cfg, ensure_ascii=False, indent=2)

def _extract_sig(src): m = re.search(r'self\.compute_reward\(([^)]+)\)', src); return m.group(1).strip() if m else "action"

def _strip_module_prefix(code: str) -> str:
    """Remove ALL module-level imports, docstrings, blank prefix.
    Properly handles multi-line docstrings. Returns clean function body."""
    lines = code.splitlines(); i = 0; in_docstring = False
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("def compute_reward"):
            return "\n".join(lines[i:])
        if s.startswith('"""') or s.startswith("'''"):
            in_docstring = not in_docstring
            if s.count('"""') >= 2 or s.count("'''") >= 2:
                in_docstring = False  # single-line docstring
            i += 1; continue
        if in_docstring:
            i += 1; continue
        if s.startswith("import ") or s.startswith("from ") or s.startswith("#") or not s:
            i += 1; continue
        # Non-prefix, non-function content found before compute_reward — skip it
        i += 1
    return "\n".join(lines[i:])

def _auto_fix_code(code: str) -> str:
    lines = code.splitlines(); seen_imports = set(); seen_header = False; fixed = []
    for line in lines:
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            if s in seen_imports: continue
            seen_imports.add(s)
        if '"""LLM-generated' in s:
            if seen_header:
                if s.count('"""') >= 2 or s.endswith('"""'): continue
            else: seen_header = True
        fixed.append(line)
    return "\n".join(fixed)

# ═══════════════════ Round 0 ═══════════════════
def run_round0(env_dir, exploration_path, config, exp_dir, api_key, model, temperature, dry_run=False):
    r0 = ensure_dir(exp_dir / "round0"); memory = MemoryStore(exp_dir)
    step_src = load_text(env_dir / "step.py"); expl_data = load_text(exploration_path)
    if not dry_run:
        tu = EnvInterpreter(env_dir, exploration_path, api_key, model, 0.3).interpret()
        save_json(memory.task_manifest_path.with_suffix(".json"), tu)
        save_text(memory.task_manifest_path, _tu_md(tu))
    else: tu = {}
    if not dry_run:
        prompt = _build_r0_prompt(tu, step_src, expl_data, config)
        save_text(r0 / "initial_prompt.txt", prompt)
        resp = call_llm(prompt, api_key, model, temperature)
        save_text(r0 / "initial_response.md", resp)
        code = extract_code_from_response(resp)
        if not code: raise RuntimeError("No code block in initial response")
        hdr = f'"""LLM-generated (Round 0).\n{datetime.now(BEIJING).isoformat()}\n"""\n\nimport math\nimport numpy as np\n\n'
        save_text(r0 / "reward_fn_source.py", hdr + _strip_module_prefix(code) + "\n")
    public = {k:v for k,v in config.items() if k!="llm_api_key"}
    save_text(exp_dir / "config.yaml", _dump_config(public)); save_text(r0 / "config.yaml", _dump_config(public))
    return {"r0": r0, "tu": tu}

def _build_r0_prompt(tu, step_src, expl, cfg):
    sig = _extract_sig(step_src)
    return f"""Design an initial reward function for an RL environment.

## Environment Step Code (how physics & termination work)
```python
{step_src[:5000]}
```
## Exploration Data
```json
{expl[:3000]}
```
## Task Understanding
```json
{json.dumps(tu, indent=2, ensure_ascii=False)}
```

## Your Task
Write ONLY a `def compute_reward(self, {sig}):` function.

Requirements:
- Signature: `def compute_reward(self, {sig}):` — must match EXACTLY
- Returns: `(float, dict)` — total reward and dict of component_name→value
- Design 3-4 reward components MAX. Simple rewards work better. More complexity can be added in later iterations if needed.
- Each component must serve a CLEAR, DISTINCT purpose. Do not create overlapping components.
- Ensure at least ONE component provides positive signal (not all negative).
- Prefer simple formulas over complex conditional logic.
- Do NOT include ANY import statements. numpy (as np) and math are pre-injected in scope.
- Do NOT store simulator objects as self attributes
- Every component must appear in the returned dict
- Initialize every variable BEFORE any conditional that might skip its definition

Output ONLY the ```python code block. No explanations."""

def _tu_md(tu):
    ti = tu.get("task_identity", {})
    pc = tu.get("physical_constraints", {})
    di = tu.get("design_implications", {})
    traps = tu.get("reward_trap_warnings", [])
    return f"""# Task Manifest
## Task: {ti.get('primary_objective','')}
## Success: {ti.get('success_condition','')}
## Failure: {ti.get('failure_conditions',[])}
## Physics: gravity={pc.get('gravity_strength','?')}, balance={pc.get('balance_critical',False)}
## Implications: {json.dumps(di)}
## Traps: {json.dumps(traps)}"""

# ═══════════════════ Round N ═══════════════════
def run_iteration(exp_dir, env_dir, round_num, exploration_path, config, api_key, model, temperature, dry_run=False, skip_train=False):
    prev = exp_dir / f"round{round_num-1}"; out = ensure_dir(exp_dir / f"round{round_num}")
    memory = MemoryStore(exp_dir)
    tu = load_json(memory.task_manifest_path.with_suffix(".json")) or {}
    step_src = load_text(env_dir / "step.py")
    dc = config.get("diagnostician", {}); max_react = dc.get("max_react_steps", 5)

    # Step 1: Evidence
    board = EvidenceAnalyzer(prev, env_dir, exp_dir).analyze() if not dry_run else {"meta":{"round":round_num}}

    # Step 2: Diagnosticians (parallel)
    res = {"a": None, "b": None}
    if not dry_run:
        def _a(): res["a"] = run_diagnostician("A", prev, board, tu, memory, api_key, model, dc.get("temperature_a",0.6), max_react)
        def _b(): res["b"] = run_diagnostician("B", prev, board, tu, memory, api_key, model, dc.get("temperature_b",0.4), max_react)
        ta=threading.Thread(target=_a); tb=threading.Thread(target=_b); ta.start(); tb.start(); ta.join(); tb.join()
        da, db = res["a"] or {}, res["b"] or {}
    else: da, db = {}, {}

    # Step 3-5: Moderator → convergence
    agenda = run_moderator_phase1(da, db, board, api_key, model) if not dry_run else {}
    save_json(out / "debate_agenda.json", agenda) if not dry_run else None

    # Pass current + historical reward code + task understanding to Moderator
    prev_rw = load_text(prev / "reward_fn_source.py")
    mem_ctx = memory.get_recent_lessons(n=3) if round_num > 1 else ""
    # Also append actual reward code history
    for r in range(max(0, round_num - 3), round_num):
        rw_hist = memory.get_reward(r)
        if rw_hist:
            mem_ctx += f"\n[Round {r} reward code]\n```python\n{rw_hist[:1500]}\n```\n"
    conv = run_moderator_phase2(agenda, da, db, board, prev_rw, tu, mem_ctx, False, api_key, model) if not dry_run else {"decision":"converge","final_diagnosis":da}
    save_json(out / "convergence_decision.json", conv)
    final_diag = conv.get("final_diagnosis", da)
    save_json(out / "final_diagnosis.json", final_diag)

    # Step 6: Apply diagnosis changes (fragments first, full_code fallback)
    if not dry_run and not skip_train:
        expected_sig = _extract_sig(step_src)

        # Try fragment edits first
        plan = {"edits": final_diag.get("proposed_changes", [])}
        code = apply_edits(prev_rw, plan)
        applied = plan.get("applied_count", 0)
        total = plan.get("total_count", 1)

        # Fallback: use full_code from Moderator if fragments failed
        full_code = conv.get("full_code", "") or final_diag.get("full_code", "")
        if applied < total and full_code:
            print(f"  Only {applied}/{total} fragments applied. Using Moderator's full_code.")
            code = full_code

        v = validate_code(code, expected_sig, prev_rw, final_diag)
        if v["passed"]:
            print(f"  Validation PASSED ({applied}/{total} fragments)")
        else:
            print(f"  Validation: {len(v['errors'])} errors")

        hard_errors = [e for e in v.get('errors', [])
                       if e['category'] in ('syntax', 'signature', 'return_tuple')]
        if hard_errors:
            code = _auto_fix_code(code)
            v = validate_code(code, expected_sig, prev_rw, final_diag)
            hard_errors = [e for e in v.get('errors', [])
                           if e['category'] in ('syntax', 'signature', 'return_tuple')]
            if hard_errors:
                print(f"  HARD errors remain, using previous reward.")
                code = prev_rw

        hdr = f'"""LLM-generated (Round {round_num}).\n"""\n\nimport math\nimport numpy as np\n\n'
        save_text(out / "reward_fn_source.py", hdr + _strip_module_prefix(code) + "\n")
    else: code = prev_rw

    # Step 7: Train
    if not dry_run and not skip_train:
        rcfg = {"total_timesteps":config.get("total_timesteps",2_000_000),"n_envs":config.get("n_envs",8),
                "normalize":config.get("normalize",True),"seed":config.get("seed"),"device":config.get("device","cpu"),
                "ppo":config.get("ppo",{}),"evaluation":config.get("evaluation",{"freq":200000,"episodes":10}),
                "checkpoint":config.get("checkpoint",{"freq":200000})}
        save_text(out / "config.yaml", _dump_config(rcfg))
        eid = f"{env_dir.name}-round{round_num}"
        cmd = [sys.executable, str(_wd/"train.py"), "--env-dir", str(env_dir), "--env-id", eid,
               "--config", str(out/"config.yaml"), "--run-dir", str(out), "--reward-source", str(out/"reward_fn_source.py")]
        t0=perf_counter(); r=_run_subprocess(cmd)
        if r.returncode!=0: print(f"  Training FAILED!"); return {"round":round_num,"trained":False}
        print(f"  Done ({perf_counter()-t0:.0f}s)")
        EvidenceAnalyzer(out, env_dir, exp_dir).analyze()
        memory.add_round_lesson(round_num, str(final_diag.get("diagnosis",""))[:200], f"see board r{round_num}", f"R{round_num} done")
        # Save reward to memory for future agents to reference
        memory.save_reward(round_num, load_text(out / "reward_fn_source.py"))
    return {"round":round_num, "trained": not (dry_run or skip_train)}

# ═══════════════════ CLI ═══════════════════
def main():
    p = argparse.ArgumentParser(description="multi_reward pipeline")
    p.add_argument("--mode", required=True, choices=["round0","iterate","continue","full"])
    p.add_argument("--experiment-dir"); p.add_argument("--env-dir"); p.add_argument("--exploration")
    p.add_argument("--config"); p.add_argument("--round", type=int, default=1)
    p.add_argument("--model", default="deepseek-reasoner"); p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    cfg = load_yaml(Path(args.config)) if args.config else {}
    api_key = cfg.get("llm_api_key", api_key)
    if not api_key and not args.dry_run: print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)

    if args.mode == "round0":
        env_dir=Path(args.env_dir).resolve(); exp_path=Path(args.exploration).resolve()
        exp_dir=_wd.parent/"runs"/experiment_dir_name(env_dir.name, cfg.get("total_timesteps",2000000))
        setup_logging(exp_dir)
        run_round0(env_dir, exp_path, cfg, exp_dir, api_key, args.model, args.temperature, args.dry_run)
        print(f"\nExperiment: {exp_dir}")

    elif args.mode == "iterate":
        exp_dir=Path(args.experiment_dir).resolve(); setup_logging(exp_dir)
        if (exp_dir/"config.yaml").exists(): cfg = load_yaml(exp_dir/"config.yaml")
        env_dir=_find_env(exp_dir, cfg.get("env_id"))
        run_iteration(exp_dir, env_dir, args.round, _find_expl(exp_dir), cfg, api_key, args.model, args.temperature, args.dry_run, args.skip_train)

    elif args.mode == "continue":
        exp_dir=Path(args.experiment_dir).resolve(); setup_logging(exp_dir)
        cfg = load_yaml(exp_dir/"config.yaml") if (exp_dir/"config.yaml").exists() else {}
        total=cfg.get("rounds",5); memory=MemoryStore(exp_dir); start=max(memory.get_available_rounds())+1 if memory.get_available_rounds() else 1
        env_dir=_find_env(exp_dir, cfg.get("env_id"))
        for r in range(start, total+1):
            if not run_iteration(exp_dir, env_dir, r, _find_expl(exp_dir), cfg, api_key, args.model, args.temperature).get("trained"): break

    elif args.mode == "full":
        env_dir=Path(args.env_dir).resolve(); exp_path=Path(args.exploration).resolve(); cfg=load_yaml(Path(args.config))
        total=cfg.get("rounds",5); exp_dir=_wd.parent/"runs"/experiment_dir_name(env_dir.name, cfg.get("total_timesteps",2000000))
        setup_logging(exp_dir)
        run_round0(env_dir, exp_path, cfg, exp_dir, api_key, args.model, args.temperature, args.dry_run)
        if args.dry_run: return
        train=_wd/"train.py"; r0=exp_dir/"round0"; eid=f"{env_dir.name}-round0"
        save_text(r0/"config.yaml", _dump_config(cfg))
        r=_run_subprocess([sys.executable,str(train),"--env-dir",str(env_dir),"--env-id",eid,"--config",str(r0/"config.yaml"),"--run-dir",str(r0),"--reward-source",str(r0/"reward_fn_source.py")])
        if r.returncode!=0: print("Round 0 train FAILED"); sys.exit(1)
        EvidenceAnalyzer(r0, env_dir, exp_dir).analyze()
        cfg["env_id"]=eid
        for rn in range(1, total+1):
            if not run_iteration(exp_dir, env_dir, rn, exp_path, cfg, api_key, args.model, args.temperature).get("trained"): break
        save_text(exp_dir/"STATUS", f"COMPLETED ({total} rounds)\n"); print(f"\nDone: {exp_dir}")

def _find_env(exp_dir, hint=None):
    import re
    bases = [_wd.parent/"eureka_llm"/"envs", _wd.parent/"envs"]
    if hint:
        env_name = re.sub(r"-round\d+$", "", hint)
        for base in bases:
            if not base.exists(): continue
            for d in base.iterdir():
                if d.is_dir() and (d/"step.py").exists() and d.name.lower()==env_name.lower(): return d
    exp_name = exp_dir.name
    parts = exp_name.rsplit("_", 2)
    env_name = parts[0] if parts else exp_name
    for base in bases:
        if not base.exists(): continue
        for d in base.iterdir():
            if d.is_dir() and (d/"step.py").exists() and d.name.lower()==env_name.lower(): return d
    for base in bases:
        if not base.exists(): continue
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d/"step.py").exists(): return d
    raise FileNotFoundError(f"Cannot find env dir. Searched: {bases}")

def _find_expl(exp_dir):
    for base in [_wd.parent/"eureka_llm"/"explorations", _wd.parent/"explorations"]:
        if base.exists():
            for f in base.glob("*.json"): return f
    return None

if __name__=="__main__": main()
