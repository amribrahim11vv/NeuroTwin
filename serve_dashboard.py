# -*- coding: utf-8 -*-
"""
serve_dashboard.py
==================
Tribe V2 — Lightweight Custom Web Server for Clinical Dashboard.
Serves static dashboard files and provides REST API endpoints.
Zero external dependencies (uses standard library http.server).
"""

import http.server
import socketserver
import json
import urllib.parse
import re
from pathlib import Path
import csv
from datetime import datetime, timezone

PORT = 8000
HOST = "127.0.0.1"
BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR / "outputs"
CLINICAL_DIR = OUTPUTS_DIR / "clinical_outputs"
REPORTS_DIR = OUTPUTS_DIR / "clinical_reports"
AUDIT_LOG = OUTPUTS_DIR / "hitl_audit_log.jsonl"


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        parsed_url = urllib.parse.urlparse(path)
        rel_path = urllib.parse.unquote(parsed_url.path).lstrip('/')
        if not rel_path or rel_path == "/":
            rel_path = "index.html"
        if rel_path == "shap_bar_plot.png":
            target = (OUTPUTS_DIR / "shap_bar_plot.png").resolve()
        else:
            target = (BASE_DIR / "dashboard" / rel_path).resolve()

        # Prevent path traversal attacks escaping BASE_DIR
        try:
            if not target.is_relative_to(BASE_DIR.resolve()):
                return str(BASE_DIR / "dashboard" / "index.html")
        except AttributeError:
            if not str(target).startswith(str(BASE_DIR.resolve())):
                return str(BASE_DIR / "dashboard" / "index.html")
        return str(target)

    def end_headers(self):
        # Prevent caching for development/updates
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_GET(self):
        # Parse URL path
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # Route API requests
        if path == "/api/patients":
            self.get_patients()
        elif path == "/api/cohort":
            self.get_cohort()
        elif path == "/api/shap":
            self.get_shap()
        elif path.startswith("/api/patient/"):
            raw_pid = path.split("/")[-1]
            pid = re.sub(r'[^a-zA-Z0-9_-]', '', raw_pid)
            self.get_patient_detail(pid)
        else:
            # Fallback to serving static files
            if path == "/":
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/hitl/submit":
            self.submit_hitl()
        else:
            self.send_json_error(404, "Endpoint not found")

    # -- API Handlers -------------------------------------------------

    def get_patients(self):
        """Fetch list of all patients and brief summary of their records."""
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

        # Sort by participant ID (e.g., P01, P02...)
        try:
            patients.sort(key=lambda x: int(x["participant_id"][1:]) if x["participant_id"][1:].isdigit() else x["participant_id"])
        except Exception:
            patients.sort(key=lambda x: x["participant_id"])

        self.send_json(patients)

    def get_patient_detail(self, pid):
        """Fetch detailed digital twin profile for a single patient."""
        file_path = CLINICAL_DIR / f"report_{pid}.json"
        if not file_path.exists():
            self.send_error(404, f"Patient {pid} not found")
            return

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.send_json(data)
        except Exception as e:
            self.send_error(500, f"Error reading patient detail: {str(e)}")

    def get_cohort(self):
        """Fetch cohort-level statistics summary."""
        file_path = CLINICAL_DIR / "cohort_summary.json"
        if not file_path.exists():
            self.send_error(404, "Cohort summary not found")
            return

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.send_json(data)
        except Exception as e:
            self.send_error(500, f"Error reading cohort summary: {str(e)}")

    def get_shap(self):
        """Parse and return SHAP summary data."""
        summary_path = OUTPUTS_DIR / "shap_summary.txt"
        if not summary_path.exists():
            self.send_error(404, "SHAP summary not found")
            return

        try:
            features = []
            with open(summary_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            # Parse lines (skip headers)
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
            self.send_json({"features": features})
        except Exception as e:
            self.send_error(500, f"Error parsing SHAP summary: {str(e)}")

    def submit_hitl(self):
        """Log HITL gate decisions and export updated CSV."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length <= 0 or content_length > 1024 * 1024:
                self.send_json_error(400, "Invalid or excessive Content-Length")
                return

            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            pid = data.get("participant_id")
            decision = data.get("decision")
            override_action = data.get("final_action")
            note = str(data.get("clinician_note", ""))

            if not pid or not decision:
                self.send_json_error(400, "Missing participant_id or decision")
                return

            # Sanitize PID
            pid = re.sub(r'[^a-zA-Z0-9_-]', '', str(pid))

            # Read existing patient details to append correct context
            patient_file = CLINICAL_DIR / f"report_{pid}.json"
            pre_pcl5, post_pcl5, delta, group = None, None, None, "UNKNOWN"
            if patient_file.exists():
                with open(patient_file, "r", encoding="utf-8") as fh:
                    p_data = json.load(fh)
                    pre_pcl5 = p_data["clinical_baselines"]["pre_pcl5_total"]
                    post_pcl5 = p_data["clinical_baselines"]["post_pcl5_total"]
                    delta = p_data["model_predictions"]["pcl5_delta"]
                    group = p_data["group"]

            # Construct audit entry
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "participant_id": pid,
                "group": group,
                "rl_recommendation": data.get("rl_recommendation", ""),
                "q_values": data.get("q_values", [0.0, 0.0, 0.0]),
                "decision": decision,
                "final_action": override_action,
                "clinician_note": note,
                "pre_pcl5": pre_pcl5,
                "post_pcl5": post_pcl5,
                "pcl5_delta": delta,
            }

            # Append to JSONL audit log
            with open(AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            # Update decisions CSV
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

            self.send_json({"status": "success", "message": "Decision logged successfully"})
        except json.JSONDecodeError:
            self.send_json_error(400, "Invalid JSON payload")
        except Exception as e:
            self.send_json_error(500, f"Error logging decision: {str(e)}")

    # -- Helper Methods -----------------------------------------------

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def send_json_error(self, code, message):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message, "status": code}).encode("utf-8"))


def run():
    # Allow port reuse to avoid 'address already in use' errors on quick restarts
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((HOST, PORT), DashboardHandler) as httpd:
        print(f"\n========================================================")
        print(f" Tribe V2 - Clinical Decision-Support Dashboard")
        print(f" Active and running at: http://{HOST}:{PORT}")
        print(f" Press Ctrl+C to terminate the server.")
        print(f"========================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    run()
