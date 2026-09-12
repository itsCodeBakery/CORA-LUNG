from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]

state = json.loads(
    (ROOT / "PROJECT_STATE.json").read_text(encoding="utf-8")
)

commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    text=True,
    capture_output=True
).stdout.strip()

print("=" * 90)
print("CORA-LUNG PROJECT RESUME")
print("=" * 90)
print("Repository      :", ROOT)
print("Commit          :", commit)
print("Completed block :", state["last_completed_block"])
print("Stage           :", state["current_stage"])
print("Current gate    :", state["current_gate"])
print("Next action     :", state["next_action"])
print("=" * 90)
