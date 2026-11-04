"""Print the step trace of questions that failed, to see if MAX_STEPS binds.

A forced finish leaves a step {"type": "final", "note": "forced after max steps"};
a voluntary finish is a plain final step. Steps are loop *turns*, not tool calls,
so a turn with three parallel calls costs one step.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"D:\prac_prog\fun\Prismforce\Policy_agent\policy-rag-agent_V3\policy-rag-agent")
sys.path.insert(0, str(ROOT))

PACK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Benchmark" / "hr_rag_policy_pack_and_benchmark"
IDS = (sys.argv[2] if len(sys.argv) > 2 else "MH-01,MH-12").split(",")
DATA = str(sys.argv[3]) if len(sys.argv) > 3 else str(ROOT / "data_bench")
os.environ["DATA_DIR"] = DATA

from app import config as cfg_mod            # noqa: E402
cfg_mod._CACHE = None
from app.admin import AdminAPI               # noqa: E402

rows = {r["id"]: r for r in (json.loads(line) for line in
        (sorted(PACK.glob("*.jsonl"))[0]).read_text(encoding="utf-8").splitlines())}

api = AdminAPI()
print(f"AGENT_MAX_STEPS={api.cfg.agent_max_steps}  data={DATA}\n")
for qid in IDS:
    row = rows[qid]
    res = api.ask(row["question"])
    steps = res.steps or []
    tools = [s.get("tool") for s in steps if s.get("type") == "tool_call"]
    forced = [s for s in steps if str(s.get("note", "")).startswith("forced")]
    print(f"=== {qid}: {len(steps)} steps, {len(tools)} tool calls, "
          f"FORCED={bool(forced)}  ({res.elapsed_s:.1f}s)")
    print("    tools:", ", ".join(t or "?" for t in tools))
    print("    last step:", json.dumps(steps[-1])[:170] if steps else "(none)")
    print("    answer:", (res.answer or "")[:300].replace("\n", " "))
    print()
