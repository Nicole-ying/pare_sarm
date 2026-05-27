"""Test Meta-Analyzer on LunarLander rounds 0-4, producing analysis for round 5."""

import sys, json, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory.memory_system import MemorySystem
from agents.meta_analyzer_agent import run_meta_analyzer_agent

EXP_DIR = Path(__file__).resolve().parent.parent / "runs" / "lunarlander-v2_2605141802_1000000"
ROUND_NUM = 5  # analyzing rounds 0-4, proposing for round 5

def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    prev_round_dir = EXP_DIR / "round4"
    memory_system = MemorySystem(EXP_DIR)

    # Build a minimal skill manager for late stage
    class DummySkillManager:
        active_docs = ""
        def activate(self, name):
            pass

    skill_mgr = DummySkillManager()

    print(f"{'='*60}")
    print(f"  Meta-Analyzer Test: LunarLander Rounds 0-4 → Proposal for Round 5")
    print(f"  Experiment: {EXP_DIR.name}")
    print(f"{'='*60}")

    result = run_meta_analyzer_agent(
        run_dir=prev_round_dir,
        round_num=ROUND_NUM,
        memory_system=memory_system,
        api_key=api_key,
        model="deepseek-reasoner",
        temperature=0.4,
        skill_manager=skill_mgr,
    )

    proposal = result.get("proposal", {})
    print(f"\n{'='*60}")
    print(f"  META-ANALYZER RESULT")
    print(f"{'='*60}")
    print(f"Diagnosis: {proposal.get('diagnosis', 'N/A')}")
    print(f"Changes: {proposal.get('changed_count', 0)}")
    for c in proposal.get("proposed_changes", []):
        print(f"\n  Component: {c.get('component', '?')}")
        print(f"  Reason: {c.get('reason', '?')[:200]}")
    print()

    # Save for review
    out_path = EXP_DIR / "round5" / "meta_analyzer_test_output.json"
    out_path.write_text(
        json.dumps({"proposal": proposal}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Full output → {out_path}")

if __name__ == "__main__":
    main()
