# ==========================================================================================
# COVA-3D — BLOCK 09E-N2-TEST-DIAG
# Diagnose N2 Regression-Test Failure
#
# NO repository modification
# NO commit
# NO optimizer step
# NO dense masks
# NO final-CV access
#
# Run in the SAME Kaggle session where 09E-N2-LOCK failed.
# ==========================================================================================

from pathlib import Path
import subprocess
import sys

REPO = Path("/kaggle/working/CORA-LUNG")

EXPECTED_HEAD = "b51e3667b543"


def heading(text):
    print("\n" + "=" * 132)
    print(text)
    print("=" * 132)


def run(cmd):
    return subprocess.run(
        cmd,
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


heading("COVA-3D 09E-N2-TEST-DIAG — VERIFY WORKING TREE")

head = run(
    ["git", "rev-parse", "HEAD"]
).stdout.strip()

print("Git HEAD                              :", head[:12])

if not head.startswith(EXPECTED_HEAD):
    raise RuntimeError(
        f"Unexpected Git HEAD.\nExpected {EXPECTED_HEAD}\nObserved {head}"
    )

status = run(
    ["git", "status", "--short"]
).stdout

print("\nWORKING TREE")
print("------------")
print(status if status.strip() else "CLEAN — unexpected after failed N2 block")


heading("STEP 1/5 — SHOW N2 LOSS IMPLEMENTATION DIFF")

diff_loss = run(
    [
        "git",
        "diff",
        "--",
        "src/cora_lung/losses/partial.py",
    ]
)

print(diff_loss.stdout)

if diff_loss.stderr.strip():
    print("STDERR:")
    print(diff_loss.stderr)


heading("STEP 2/5 — SHOW N2 TEST FILE")

test_path = (
    REPO
    / "tests"
    / "test_cova3d_numerical_amendment_n2.py"
)

if not test_path.exists():
    raise RuntimeError(
        "N2 regression-test file does not exist in the working tree."
    )

print(
    test_path.read_text(
        encoding="utf-8"
    )
)


heading("STEP 3/5 — RUN ONLY N2 TESTS WITH FULL TRACEBACK")

env = dict(**__import__("os").environ)

src = str(REPO / "src")

env["PYTHONPATH"] = (
    src
    + (
        __import__("os").pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH")
        else ""
    )
)

n2_test = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_cova3d_numerical_amendment_n2.py",
        "-vv",
        "-x",
        "--tb=long",
        "-s",
        "-p",
        "no:cacheprovider",
    ],
    cwd=str(REPO),
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

print(n2_test.stdout)

if n2_test.stderr.strip():
    print("\nPYTEST STDERR")
    print("-------------")
    print(n2_test.stderr)

print("\nN2-only pytest return code            :", n2_test.returncode)


heading("STEP 4/5 — RUN PARTIAL-LOSS TESTS SEPARATELY")

loss_test = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_partial_losses.py",
        "-vv",
        "-x",
        "--tb=long",
        "-s",
        "-p",
        "no:cacheprovider",
    ],
    cwd=str(REPO),
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

print(loss_test.stdout)

if loss_test.stderr.strip():
    print("\nPYTEST STDERR")
    print("-------------")
    print(loss_test.stderr)

print("\nPartial-loss pytest return code       :", loss_test.returncode)


heading("STEP 5/5 — IMPORT / DIRECT NUMERICAL SMOKE TEST")

try:
    import importlib

    for name in [
        "cora_lung.losses.partial",
    ]:
        sys.modules.pop(name, None)

    importlib.invalidate_caches()

    from cora_lung.losses.partial import (
        partial_dice,
        partial_dice_fp32,
        partial_segmentation_loss_n2,
    )

    import torch

    print("Import partial_dice_fp32              : PASS")
    print("Import partial_segmentation_loss_n2   : PASS")

    logits = torch.tensor(
        [-16.0, -15.5, 2.0, -1.0],
        dtype=torch.float16,
        requires_grad=True,
    )

    target = torch.tensor(
        [0, 0, 1, -1],
        dtype=torch.int8,
    )

    observed = partial_dice_fp32(
        logits,
        target,
    )

    reference = partial_dice(
        logits.float(),
        target,
    )

    print("Observed FP32 Dice                    :", float(observed.detach()))
    print("Reference FP32 Dice                   :", float(reference.detach()))
    print("Observed dtype                        :", observed.dtype)
    print(
        "Absolute difference                   :",
        abs(
            float(observed.detach())
            - float(reference.detach())
        ),
    )

    observed.backward()

    print(
        "Gradient finite                       :",
        bool(torch.isfinite(logits.grad).all()),
    )

    print("Gradient                              :", logits.grad)

except Exception as exc:

    print("DIRECT SMOKE TEST                     : FAILED")
    print(type(exc).__name__ + ": " + str(exc))

    import traceback
    traceback.print_exc()


heading("FINAL N2 TEST-DIAG REPORT")

print("Repository modified by this diagnostic : NO")
print("Git commit created                     : NO")
print("optimizer.step() calls                 : 0")
print("Dense masks accessed                   : 0")
print("Final outer-CV access                  : 0")
print("N2-only pytest return code             :", n2_test.returncode)
print("Partial-loss pytest return code        :", loss_test.returncode)

print()
print("NEXT:")
print("Send me the COMPLETE output from this cell.")
print("Do NOT rerun 09E-N2-LOCK yet.")
print(
    "I will correct only the failing implementation/test detail and preserve "
    "the already established N2 numerical diagnosis."
)

print("=" * 132)
