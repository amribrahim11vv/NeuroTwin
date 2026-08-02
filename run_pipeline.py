"""
run_pipeline.py
================
Tribe V2  -  Full Pipeline Runner.

Executes all phases in order using the kidney_ai conda environment.

Phase 1  -  Foundation:
  1. id_alignment_verification.py
  2. data_ingestion_pipeline.py
  3. reward_prior_training.py
  4. reward_prior_validation_report.py
  5. dataset_summary_report.py

Phase 2  -  LSDT + Offline RL:
  6. lsdt_backbone_pretrain.py
  7. lora_patient_adaptation.py
  8. cql_offline_rl.py

Phase 3  -  Clinical Output:
  9. clinical_output_layer.py
  10. shap_explainer.py
  11. hitl_gate.py --auto

Usage:
    python run_pipeline.py            # Run all phases
    python run_pipeline.py --phase 1  # Run Phase 1 only
    python run_pipeline.py --phase 2  # Run Phase 2 only
    python run_pipeline.py --phase 3  # Run Phase 3 only
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
PYTHON   = sys.executable   # Use same Python that launched this script

PHASES = {
    1: [
        "src/id_alignment_verification.py",
        "src/data_ingestion_pipeline.py",
        "src/reward_prior_training.py",
        "src/reward_prior_validation_report.py",
        "src/dataset_summary_report.py",
    ],
    2: [
        "src/lsdt_backbone_pretrain.py",
        "src/lora_patient_adaptation.py",
        "src/cql_offline_rl.py",
    ],
    3: [
        "src/clinical_output_layer.py",
        "src/shap_explainer.py",
        "src/hitl_gate.py --auto",
    ],
}


def run_script(script_cmd: str) -> bool:
    parts = script_cmd.split()
    script = parts[0]
    args   = parts[1:]

    print(f"\n{'='*70}")
    print(f">  Running: {script}")
    print(f"{'='*70}")

    script_path = BASE_DIR / script
    if not script_path.exists():
        print(f"[SKIP] {script} not found  -  skipping")
        return True

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(script_path)] + args,
        cwd=str(BASE_DIR),
        capture_output=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAILED] {script} exited with code {result.returncode} "
              f"after {elapsed:.1f}s")
        return False

    print(f"\n[OK] {script} completed in {elapsed:.1f}s")
    return True


def main():
    args = sys.argv[1:]
    
    if "--phase" in args:
        phase_idx = args.index("--phase")
        phase_num = int(args[phase_idx + 1])
        phases_to_run = {phase_num: PHASES[phase_num]}
    else:
        phases_to_run = PHASES

    print("=" * 70)
    print("Tribe V2  -  Full Pipeline Runner")
    print(f"Python: {PYTHON}")
    print(f"Phases to run: {list(phases_to_run.keys())}")
    print("=" * 70)

    t_start = time.time()
    failed = []

    for phase_num, scripts in phases_to_run.items():
        print(f"\n\n{'#'*70}")
        print(f"# PHASE {phase_num}")
        print(f"{'#'*70}")

        for script_cmd in scripts:
            ok = run_script(script_cmd)
            if not ok:
                failed.append(f"Phase {phase_num}: {script_cmd}")
                print(f"\n[ABORT] Phase {phase_num} failed at: {script_cmd}")
                print("        Fix the error above and re-run.")
                break
        else:
            continue
        break  # Stop at first phase failure

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE  -  {total_time:.1f}s total")
    if failed:
        print(f"FAILURES: {failed}")
        sys.exit(1)
    else:
        print("All phases completed successfully [OK]")
    print("=" * 70)


if __name__ == "__main__":
    main()
