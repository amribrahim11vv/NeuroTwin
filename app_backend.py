# -*- coding: utf-8 -*-
"""
app_backend.py
==============
NeuroTwin — High-Performance FastAPI Backend Engine.
Serves static dashboard files, REST API endpoints, Pydantic validations,
and PyTorch inference integrations for PTSD Clinical AI Decision Support.

Run via:
    python app_backend.py
or:
    uvicorn app_backend:app --reload --port 8000
"""

import re
import json
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import torch

# Paths
BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR / "outputs"
CLINICAL_DIR = OUTPUTS_DIR / "clinical_outputs"
REPORTS_DIR = OUTPUTS_DIR / "clinical_reports"
AUDIT_LOG = OUTPUTS_DIR / "hitl_audit_log.jsonl"
MODELS_DIR = BASE_DIR / "models"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# Models State
MODELS_CACHE: Dict[str, Any] = {}

# Pydantic Schemas
class PatientSummary(BaseModel):
    participant_id: str
    group: str
    n_sessions: int
    pcl5_delta: Optional[float] = None
    pre_pcl5_total: Optional[float] = None
    post_pcl5_total: Optional[float] = None

class ShapFeature(BaseModel):
    rank: int
    feature: str
    importance: float

class ShapResponse(BaseModel):
    features: List[ShapFeature]

class HITLSubmission(BaseModel):
    participant_id: str
    rl_recommendation: Optional[str] = ""
    q_values: Optional[List[float]] = [0.0, 0.0, 0.0]
    decision: str = Field(..., description="ACCEPT, OVERRIDE, or DEFER")
    final_action: Optional[str] = None
    clinician_note: Optional[str] = ""

# Load PyTorch Models helper
def load_pytorch_checkpoints():
    cql_path = MODELS_DIR / "cql_q_network.pt"
    lsdt_path = MODELS_DIR / "lsdt_backbone.pt"
    
    if cql_path.exists():
        try:
            ckpt = torch.load(cql_path, weights_only=True)
            MODELS_CACHE["cql_q_net"] = ckpt
            print(f"[AI ENGINE] Loaded CQL Q-Network Checkpoint successfully.")
        except Exception as e:
            print(f"[AI ENGINE WARN] Could not load CQL Q-Network: {e}")
            
    if lsdt_path.exists():
        try:
            ckpt = torch.load(lsdt_path, weights_only=True)
            MODELS_CACHE["lsdt_backbone"] = ckpt
            print(f"[AI ENGINE] Loaded β-VAE LSDT Backbone Checkpoint successfully.")
        except Exception as e:
            print(f"[AI ENGINE WARN] Could not load LSDT Backbone: {e}")


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="NeuroTwin — Clinical AI & Brain Twin Backend Engine",
        description="FastAPI Production Engine for PTSD Digital Brain Twin, CQL Offline RL Policy, & Human-in-the-Loop Decision Audits.",
        version="2.0.0"
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        load_pytorch_checkpoints()
        print("\n" + "="*60)
        print(" 🧠 NeuroTwin FastAPI Backend Online")
        print(" 🌐 Swagger Documentation: http://127.0.0.1:8000/docs")
        print(" 🌐 Interactive Dashboard: http://127.0.0.1:8000")
        print("="*60 + "\n")

    # API Endpoints
    @app.get("/api/patients", response_model=List[PatientSummary], tags=["Cohort Management"])
    def get_patients():
        patients = []
        if CLINICAL_DIR.exists():
            for f in CLINICAL_DIR.glob("report_*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    patients.append({
                        "participant_id": data["participant_id"],
                        "group": data["group"],
                        "n_sessions": data["n_sessions"],
                        "pcl5_delta": data["model_predictions"]["pcl5_delta"],
                        "pre_pcl5_total": data["clinical_baselines"]["pre_pcl5_total"],
                        "post_pcl5_total": data["clinical_baselines"]["post_pcl5_total"]
                    })
                except Exception as e:
                    print(f"Error reading {f}: {e}")

        try:
            patients.sort(key=lambda x: int(x["participant_id"][1:]) if x["participant_id"][1:].isdigit() else x["participant_id"])
        except Exception:
            patients.sort(key=lambda x: x["participant_id"])

        return patients

    @app.get("/api/patient/{patient_id}", tags=["Patient Explorer"])
    def get_patient_detail(patient_id: str):
        pid = re.sub(r'[^a-zA-Z0-9_-]', '', patient_id)
        file_path = CLINICAL_DIR / f"report_{pid}.json"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Patient {pid} not found")

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading report: {str(e)}")

    @app.get("/api/cohort", tags=["Cohort Management"])
    def get_cohort_summary():
        file_path = CLINICAL_DIR / "cohort_summary.json"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Cohort summary not found")

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading summary: {str(e)}")

    @app.get("/api/shap", response_model=ShapResponse, tags=["Explainable AI"])
    def get_shap_summary():
        summary_path = OUTPUTS_DIR / "shap_summary.txt"
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail="SHAP summary not found")

        try:
            features = []
            with open(summary_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            parsing = False
            for line in lines:
                if "Rank" in line and "Feature" in line:
                    parsing = True
                    continue
                if parsing:
                    line = line.strip()
                    if not line or line.startswith("-") or line.startswith("="):
                        continue
                    parts = [p.strip() for p in line.split() if p.strip()]
                    if len(parts) >= 3:
                        features.append({
                            "rank": int(parts[0]),
                            "feature": parts[1],
                            "importance": float(parts[2])
                        })
            return {"features": features}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing SHAP summary: {str(e)}")

    @app.post("/api/hitl/submit", tags=["HITL Clinical Auditor"])
    def submit_hitl(data: HITLSubmission):
        try:
            pid = re.sub(r'[^a-zA-Z0-9_-]', '', data.participant_id)
            patient_file = CLINICAL_DIR / f"report_{pid}.json"
            pre_pcl5, post_pcl5, delta, group = None, None, None, "UNKNOWN"
            
            if patient_file.exists():
                with open(patient_file, "r", encoding="utf-8") as fh:
                    p_data = json.load(fh)
                    pre_pcl5 = p_data["clinical_baselines"]["pre_pcl5_total"]
                    post_pcl5 = p_data["clinical_baselines"]["post_pcl5_total"]
                    delta = p_data["model_predictions"]["pcl5_delta"]
                    group = p_data["group"]

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "participant_id": pid,
                "group": group,
                "rl_recommendation": data.rl_recommendation or "",
                "q_values": data.q_values or [0.0, 0.0, 0.0],
                "decision": data.decision,
                "final_action": data.final_action,
                "clinician_note": data.clinician_note or "",
                "pre_pcl5": pre_pcl5,
                "post_pcl5": post_pcl5,
                "pcl5_delta": delta,
            }

            with open(AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = REPORTS_DIR / "hitl_decisions.csv"

            rows = []
            if AUDIT_LOG.exists():
                with open(AUDIT_LOG, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rows.append(json.loads(line.strip()))
                        except Exception:
                            pass

            if rows:
                keys = list(rows[0].keys())
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for row in rows:
                        row_flat = {k: (str(v) if isinstance(v, list) else v) for k, v in row.items()}
                        writer.writerow(row_flat)

            return {"status": "success", "message": "Decision logged successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error logging decision: {str(e)}")

    # Serve SHAP plot image
    @app.get("/shap_bar_plot.png", tags=["Assets"])
    def get_shap_plot():
        img_path = OUTPUTS_DIR / "shap_bar_plot.png"
        if img_path.exists():
            return FileResponse(img_path)
        raise HTTPException(status_code=404, detail="Plot image not found")

    # Static UI Files Mounting
    @app.get("/")
    def read_root():
        return FileResponse(DASHBOARD_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="static")

else:
    print("[ERROR] FastAPI and uvicorn are required to run app_backend.py. Install them via: pip install fastapi uvicorn")

if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        uvicorn.run("app_backend:app", host="127.0.0.1", port=8000, reload=True)
