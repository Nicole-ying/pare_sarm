"""Show raw Implementor output."""
import json, sys
sys.path.insert(0, '.')
from infra.llm_client import call_llm
from infra.file_utils import load_text

EXP = r'C:\Users\Administrator\eure_llm\runs\lunarlander-v2_260518092851_1000000'
r0 = open(f'{EXP}/round0/reward_fn_source.py', encoding='utf-8').read()
diag = json.load(open(f'{EXP}/round1/final_diagnosis.json', encoding='utf-8'))
changes = diag.get('final_diagnosis', diag).get('proposed_changes', [])

system = load_text('implementor/prompts/implementor_system.txt')
# Show only the relevant code section around the target
code_ctx = ""
for c in changes:
    cur = c.get('current_code', '')
    for i, line in enumerate(r0.splitlines()):
        if cur[:30] in line:
            lo = max(0, i - 1)
            hi = min(len(r0.splitlines()), i + 3)
            code_ctx = "\n".join(r0.splitlines()[lo:hi])
            break

prompt = f"""{system}

## Changes
```json
{json.dumps(changes, indent=2)}
```

## Current Code (around target)
```python
{code_ctx}
```

Output JSON edits array."""

api_key = 'YOUR_DEEPSEEK_API_KEY'
resp = call_llm(prompt, api_key, 'deepseek-chat', 0.15)

print("=" * 60)
print("RAW LLM RESPONSE:")
print("=" * 60)
print(resp)
print()

# Parse and show
try:
    if '```json' in resp:
        js = resp.split('```json')[1].split('```')[0]
    elif '```' in resp:
        js = resp.split('```')[1].split('```')[0]
    else:
        js = resp
    result = json.loads(js)
    print("PARSED EDITS:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Parse failed: {e}")
