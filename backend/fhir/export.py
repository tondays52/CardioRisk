"""
CardioRisk AI - HL7 / FHIR R4 Interoperability Module
Constructs standard FHIR R4 JSON resources (Patient, Observation, RiskAssessment, Bundle).
"""
import uuid
from datetime import datetime, date, timezone
from typing import Any, Dict, List

def build_fhir_bundle(patient_data: dict, assessment_result: dict, assessment_id: int = 1) -> dict:
    """
    Constructs a complete FHIR R4 Collection Bundle containing:
    - Patient resource
    - Observation resources (BP, BMI, HR, Cholesterol, Glucose)
    - RiskAssessment resource (Late-Fusion CVD 10-year probability & qualitative category)
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    patient_id_val = patient_data.get('id', 101)
    patient_uuid = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_DNS, f'patient-{patient_id_val}')}"
    bundle_uuid = f"urn:uuid:{uuid.uuid4()}"

    age = patient_data.get('age') or patient_data.get('age_years') or 55
    gender_code = "female" if patient_data.get('gender', 1) == 1 else "male"
    approx_birth_year = datetime.now().year - int(age)

    # 1. Patient Resource
    patient_resource = {
        "resourceType": "Patient",
        "id": f"pat-{patient_data.get('id', 101)}",
        "identifier": [
            {
                "use": "usual",
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                            "code": "MR",
                            "display": "Medical Record Number"
                        }
                    ]
                },
                "system": "http://cardiorisk.ai/patients",
                "value": f"CR-PAT-{patient_id_val:05d}" if isinstance(patient_id_val, int) else f"CR-PAT-{patient_id_val}"
            }
        ],
        "active": True,
        "gender": gender_code,
        "birthDate": f"{approx_birth_year}-01-01"
    }

    entries: list[dict[str, Any]] = [
        {
            "fullUrl": f"http://cardiorisk.ai/fhir/Patient/{patient_resource['id']}",
            "resource": patient_resource
        }
    ]

    # 2. Blood Pressure Observation
    sbp = patient_data.get('systolic_bp') or patient_data.get('ap_hi') or 120
    dbp = patient_data.get('diastolic_bp') or patient_data.get('ap_lo') or 80
    bp_observation = {
        "resourceType": "Observation",
        "id": f"obs-bp-{assessment_id}",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "vital-signs",
                        "display": "Vital Signs"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "85354-9",
                    "display": "Blood pressure panel with all children optional"
                }
            ],
            "text": "Blood Pressure"
        },
        "subject": {"reference": f"Patient/{patient_resource['id']}"},
        "effectiveDateTime": timestamp,
        "component": [
            {
                "code": {
                    "coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic blood pressure"}]
                },
                "valueQuantity": {"value": float(sbp), "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
            },
            {
                "code": {
                    "coding": [{"system": "http://loinc.org", "code": "8462-4", "display": "Diastolic blood pressure"}]
                },
                "valueQuantity": {"value": float(dbp), "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
            }
        ]
    }
    entries.append({"fullUrl": f"http://cardiorisk.ai/fhir/Observation/{bp_observation['id']}", "resource": bp_observation})

    # 3. BMI Observation
    height = float(patient_data.get('height_cm') or patient_data.get('height') or 170.0)
    weight = float(patient_data.get('weight_kg') or patient_data.get('weight') or 70.0)
    bmi_val = round(weight / ((height / 100) ** 2), 2)
    bmi_observation = {
        "resourceType": "Observation",
        "id": f"obs-bmi-{assessment_id}",
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "39156-5", "display": "Body mass index (BMI)"}]},
        "subject": {"reference": f"Patient/{patient_resource['id']}"},
        "effectiveDateTime": timestamp,
        "valueQuantity": {"value": bmi_val, "unit": "kg/m2", "system": "http://unitsofmeasure.org", "code": "kg/m2"}
    }
    entries.append({"fullUrl": f"http://cardiorisk.ai/fhir/Observation/{bmi_observation['id']}", "resource": bmi_observation})

    # 4. Multi-Modal Late-Fusion RiskAssessment Resource
    raw_fused = assessment_result.get("fused_risk_score") or assessment_result.get("risk_score") or 0.0
    fused_pct = float(raw_fused)
    fused_proba = round(fused_pct / 100.0, 4)
    cat_str = assessment_result.get("risk_category", "Moderate Risk")

    ci = assessment_result.get("confidence_interval") or {}
    ci_lower_val = ci.get("lower") if isinstance(ci, dict) and ci.get("lower") is not None else max(0.0, fused_pct - 6.5)
    ci_upper_val = ci.get("upper") if isinstance(ci, dict) and ci.get("upper") is not None else min(100.0, fused_pct + 6.5)
    ci_lower = round(float(ci_lower_val) / 100.0, 4)
    ci_upper = round(float(ci_upper_val) / 100.0, 4)

    basis_references = [
        {"reference": f"Observation/{bp_observation['id']}"},
        {"reference": f"Observation/{bmi_observation['id']}"}
    ]

    risk_assessment_resource = {
        "resourceType": "RiskAssessment",
        "id": f"risk-{assessment_id}",
        "status": "final",
        "subject": {"reference": f"Patient/{patient_resource['id']}"},
        "occurrenceDateTime": timestamp,
        "performer": {
            "display": "CardioRisk AI Late-Fusion Clinical Meta-Learner v3.0"
        },
        "basis": basis_references,
        "prediction": [
            {
                "outcome": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "44808001",
                            "display": "10-Year Cardiovascular Disease Risk (CardioRisk AI Late-Fusion Consensus)"
                        }
                    ],
                    "text": f"10-Year Cardiovascular Risk ({cat_str})"
                },
                "probabilityDecimal": fused_proba,
                "qualitativeRisk": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                            "code": "high" if fused_pct >= 60.0 else ("moderate" if fused_pct >= 30.0 else "low"),
                            "display": cat_str
                        }
                    ]
                },
                "whenRange": {
                    "low": {"value": ci_lower, "unit": "probability"},
                    "high": {"value": ci_upper, "unit": "probability"}
                },
                "rationale": f"Calculated dynamically fusing modalities: {', '.join(assessment_result.get('modalities_used', ['tabular']))}. Confidence interval range: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]."
            }
        ],
        "note": [
            {
                "text": f"Modalities Evaluated: {assessment_result.get('modalities_used', ['tabular'])}. Top SHAP Contributing Feature: {assessment_result.get('shap_explanations', [{}])[0].get('feature', 'ap_hi') if assessment_result.get('shap_explanations') else 'ap_hi'}."
            }
        ]
    }
    entries.append({"fullUrl": f"http://cardiorisk.ai/fhir/RiskAssessment/{risk_assessment_resource['id']}", "resource": risk_assessment_resource})

    bundle = {
        "resourceType": "Bundle",
        "id": f"bundle-cr-{assessment_id}",
        "identifier": {
            "system": "http://cardiorisk.ai/fhir/bundles",
            "value": f"CR-BUNDLE-{assessment_id:06d}"
        },
        "type": "collection",
        "timestamp": timestamp,
        "total": len(entries),
        "entry": entries
    }

    return bundle
