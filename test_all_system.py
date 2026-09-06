"""
Comprehensive Test & Verification Suite for CardioRisk AI
Validates Backend API, All 6 Deep Learning/ML Modalities, Database, Reports, FHIR & Frontend Pages
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from backend.app import app

client = TestClient(app)

results = []

def test_case(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results.append((name, True, "OK"))
                print(f"  [PASS] {name}")
            except Exception as e:
                results.append((name, False, str(e)))
                print(f"  [FAIL] {name}: {e}")
        return wrapper
    return decorator

print("=" * 70)
print(" CARDIORISK AI - SYSTEM COMPREHENSIVE TEST SUITE")
print("=" * 70)

# 1. Health & Status
@test_case("API Health Check (/health)")
def test_health():
    res = client.get("/health")
    assert res.status_code == 200, f"Status: {res.status_code}"
    data = res.json()
    assert data.get("status") == "healthy", f"Unexpected status: {data}"
    assert "models_loaded" in data, "Missing models_loaded in health response"

# 2. Authentication
@test_case("Doctor Login Authentication (/login)")
def test_doctor_login():
    res = client.post("/login", json={"username": "doctor", "password": "doctor123"})
    assert res.status_code == 200, f"Status: {res.status_code}, text: {res.text}"
    data = res.json()
    assert data.get("role") == "doctor", f"Unexpected role: {data}"

@test_case("Patient Login Authentication (/login)")
def test_patient_login():
    res = client.post("/login", json={"username": "patient", "password": "patient123"})
    assert res.status_code == 200, f"Status: {res.status_code}, text: {res.text}"
    data = res.json()
    assert data.get("role") == "patient", f"Unexpected role: {data}"

@test_case("Invalid Login Rejection (/login)")
def test_invalid_login():
    res = client.post("/login", json={"username": "wrong", "password": "bad"})
    assert res.status_code == 401, f"Expected 401, got {res.status_code}"

# 3. Statistics & Database
@test_case("Get Statistics (/get-statistics)")
def test_statistics():
    res = client.get("/get-statistics")
    assert res.status_code == 200, f"Status: {res.status_code}"
    data = res.json()
    assert "total" in data and "high_risk" in data, f"Unexpected statistics: {data}"

@test_case("Get All Patients Registry (/patients/distinct & /get-assessments)")
def test_patients():
    res1 = client.get("/patients/distinct")
    assert res1.status_code == 200, f"Status: {res1.status_code}"
    res2 = client.get("/get-assessments")
    assert res2.status_code == 200, f"Status: {res2.status_code}"

# 4. Tabular Risk Prediction
@test_case("Tabular Assessment (/predict)")
def test_tabular_predict():
    payload = {
        "age": 55,
        "gender": 1,
        "height_cm": 170.0,
        "weight_kg": 78.0,
        "systolic_bp": 140,
        "diastolic_bp": 90,
        "cholesterol": 2,
        "glucose": 1,
        "smoking": 0,
        "alcohol": 0,
        "physical_activity": 1
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200, f"Status: {res.status_code}, text: {res.text}"
    data = res.json()
    assert "risk_score" in data or "fused_risk_score" in data, f"Missing risk score in response: {data}"
    assert "risk_category" in data, f"Missing risk_category in response: {data}"

# 5. Full Multimodal Fusion Prediction
@test_case("Full Multimodal Ingestion & Dynamic Fusion (/predict)")
def test_full_multimodal_predict():
    payload = {
        "age": 62,
        "gender": 1,
        "height_cm": 168.0,
        "weight_kg": 84.0,
        "systolic_bp": 155,
        "diastolic_bp": 95,
        "cholesterol": 3,
        "glucose": 2,
        "smoking": 1,
        "alcohol": 1,
        "physical_activity": 0,
        "ecg": {
            "heart_rate_bpm": 84.0,
            "pr_interval_ms": 165.0,
            "qrs_duration_ms": 92.0,
            "qtc_interval_ms": 435.0,
            "st_elevation_mv": 0.05
        },
        "hrv": {
            "heart_rate_bpm": 82.0,
            "sdnn_ms": 28.5,
            "rmssd_ms": 22.1,
            "pnn50_percent": 4.2
        },
        "heart_sound": {
            "heart_rate_bpm": 82.0,
            "s1_intensity": 1.2,
            "s2_intensity": 1.0,
            "systolic_murmur_detected": False
        },
        "scg": {
            "heart_rate_bpm": 83.0,
            "mean_interval_sec": 0.72,
            "interval_std_sec": 0.04,
            "signal_energy": 12.4
        },
        "retinal": {
            "vessel_density": 0.14,
            "mean_vessel_width_px": 2.8,
            "tortuosity_index": 0.35
        }
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200, f"Status: {res.status_code}, text: {res.text}"
    data = res.json()
    assert "fused_risk_score" in data, f"Missing score: {data}"
    assert len(data.get("modalities_used", [])) >= 5, f"Expected multimodal fusion: {data.get('modalities_used')}"

# 6. Meta-Learner Late Fusion Endpoint
@test_case("Meta-Learner 6-Modality Late-Fusion (/predict_risk)")
def test_meta_learner_predict():
    payload = {
        "tabular_vector": [0.65],
        "p_ecg": 0.72,
        "p_heart_sound": 0.58,
        "p_ppg": 0.61,
        "p_scg": 0.55,
        "p_retinal": 0.68
    }
    res = client.post("/predict_risk", json=payload)
    assert res.status_code == 200, f"Status: {res.status_code}, text: {res.text}"
    data = res.json()
    assert "unified_risk_score" in data, f"Missing unified_risk_score: {data}"
    assert "clinical_diagnosis" in data, f"Missing clinical_diagnosis: {data}"

# 7. AI Clinical Dialogue Endpoint
@test_case("Clinical AI Chatbot Dialogue (/api/chat)")
def test_chat():
    res = client.post("/api/chat", json={"message": "What is normal blood pressure?"})
    assert res.status_code == 200, f"Status: {res.status_code}"
    data = res.json()
    assert "reply" in data, f"Missing reply in chat: {data}"

# 8. Counterfactual & XAI
@test_case("Counterfactual Recommendations (/counterfactual)")
def test_counterfactual():
    from scripts.counterfactual import generate_counterfactuals
    features = {
        "age": 58,
        "gender": 1,
        "height_cm": 172.0,
        "weight_kg": 88.0,
        "systolic_bp": 150,
        "diastolic_bp": 95,
        "cholesterol": 2,
        "glucose": 1,
        "smoking": 1,
        "alcohol": 0,
        "physical_activity": 0
    }
    cf = generate_counterfactuals(features)
    assert "current_risk" in cf, f"Missing current_risk: {cf}"
    assert "changes" in cf, f"Missing changes: {cf}"

# 9. FHIR Export Endpoint
@test_case("FHIR Clinical Bundle Export (/fhir/export-bundle)")
def test_fhir():
    patient_info = {"id": "TEST_AUTO_001", "name": "John Doe", "age": 55, "gender": 1}
    assessment = {"fused_risk_score": 42.5, "risk_category": "Moderate Risk", "timestamp": "2026-09-01T12:00:00"}
    res = client.post("/fhir/export-bundle", json={"patient_data": patient_info, "result": assessment})
    assert res.status_code == 200, f"Status: {res.status_code}, text: {res.text}"
    bundle = res.json()
    assert bundle.get("resourceType") == "Bundle", f"Invalid FHIR Bundle: {bundle}"
    assert len(bundle.get("entry", [])) >= 2, f"Expected entries in FHIR bundle: {bundle}"

# 10. Clinical PDF Report Generation Endpoint
@test_case("Clinical PDF Report Generator Endpoint (/generate-pdf-report)")
def test_pdf():
    payload = {
        "patient_data": {
            "id": 101,
            "name": "Jane Smith",
            "age": 60,
            "gender": 2,
            "systolic_bp": 145,
            "diastolic_bp": 92,
            "cholesterol": 2,
            "glucose": 1
        },
        "result": {
            "fused_risk_score": 48.2,
            "risk_category": "Moderate Risk",
            "modalities_used": ["tabular", "ecg", "ppg"]
        }
    }
    res = client.post("/generate-pdf-report", json=payload)
    assert res.status_code == 200, f"Status: {res.status_code}"
    assert res.headers.get("content-type") == "application/pdf", f"Unexpected content-type: {res.headers}"
    assert len(res.content) > 500, "PDF content too small"

# 11. Frontend Interactive Web Tools
@test_case("Interactive HTML Tools Verification")
def test_html_tools():
    tools = [
        "landing.html",
        "ai_chatbot.html",
        "chatbot.html",
        "ppg_capture.html",
        "scg_capture.html",
        "heart_sound_capture.html",
        "retinal_capture.html",
        "live_telemetry.html",
        "voice_input.html"
    ]
    for t in tools:
        path = os.path.join(project_root, "frontend", t)
        assert os.path.exists(path), f"Tool {t} missing at {path}"
        assert os.path.getsize(path) > 500, f"Tool {t} is empty"

# 12. Streamlit App Syntax & Compilation Check
@test_case("Streamlit App Compilation (frontend/app.py)")
def test_streamlit_compilation():
    app_path = os.path.join(project_root, "frontend", "app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()
    compile(code, app_path, "exec")

# Run all tests
test_health()
test_doctor_login()
test_patient_login()
test_invalid_login()
test_statistics()
test_patients()
test_tabular_predict()
test_full_multimodal_predict()
test_meta_learner_predict()
test_chat()
test_counterfactual()
test_fhir()
test_pdf()
test_html_tools()
test_streamlit_compilation()

print("=" * 70)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"RESULTS: {passed}/{total} Test Cases Passed")
print("=" * 70)

if passed < total:
    sys.exit(1)
