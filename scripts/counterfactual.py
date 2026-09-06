"""
CardioRisk AI - Counterfactual Explanations
Finds minimal changes to modifiable features to reduce risk.
"""
import joblib
import pandas as pd
import numpy as np

from typing import Dict, Any, List

FEATURE_ORDER = [
    'age', 'gender', 'height_cm', 'weight_kg', 'bmi',
    'systolic_bp', 'diastolic_bp', 'pulse_pressure',
    'cholesterol', 'glucose', 'smoking', 'alcohol', 'physical_activity'
]

MODIFIABLE_FEATURES = ['systolic_bp', 'smoking', 'alcohol', 'physical_activity', 'weight_kg']

def load_model():
    return joblib.load("models/tabular/best_model.joblib")

def _normalize_features(features: dict) -> dict:
    """Normalize aliases and compute derived metrics (BMI, pulse pressure)."""
    norm: Dict[str, Any] = {}
    
    age = features.get('age')
    if age is None:
        age = features.get('age_years', 50)
    norm['age'] = int(age if age is not None else 50)
    
    gender = features.get('gender')
    norm['gender'] = int(gender if gender is not None else 1)
    
    height = features.get('height_cm')
    if height is None:
        height = features.get('height', 170.0)
    height_val = float(height if height is not None else 170.0)
    
    weight = features.get('weight_kg')
    if weight is None:
        weight = features.get('weight', 70.0)
    weight_val = float(weight if weight is not None else 70.0)
    
    norm['height_cm'] = height_val
    norm['weight_kg'] = weight_val
    
    bmi_val = features.get('bmi')
    if bmi_val is not None:
        norm['bmi'] = float(bmi_val)
    else:
        h_m = height_val / 100.0 if height_val > 3.0 else height_val
        norm['bmi'] = round(weight_val / (h_m ** 2), 2) if h_m > 0 else 24.2
        
    sbp = features.get('systolic_bp')
    if sbp is None:
        sbp = features.get('ap_hi', 120)
    sbp_val = int(sbp if sbp is not None else 120)
    
    dbp = features.get('diastolic_bp')
    if dbp is None:
        dbp = features.get('ap_lo', 80)
    dbp_val = int(dbp if dbp is not None else 80)
    
    norm['systolic_bp'] = sbp_val
    norm['diastolic_bp'] = dbp_val
    norm['pulse_pressure'] = sbp_val - dbp_val
    
    chol = features.get('cholesterol')
    norm['cholesterol'] = int(chol if chol is not None else 1)
    
    gluc = features.get('glucose')
    if gluc is None:
        gluc = features.get('gluc', 1)
    norm['glucose'] = int(gluc if gluc is not None else 1)
    
    smoke = features.get('smoking')
    if smoke is None:
        smoke = features.get('smoke', 0)
    norm['smoking'] = int(smoke if smoke is not None else 0)
    
    alco = features.get('alcohol')
    if alco is None:
        alco = features.get('alco', 0)
    norm['alcohol'] = int(alco if alco is not None else 0)
    
    active = features.get('physical_activity')
    if active is None:
        active = features.get('active', 1)
    norm['physical_activity'] = int(active if active is not None else 1)
    
    return norm

def predict_risk(model, features: dict) -> float:
    """Predict risk for a given feature set."""
    norm_f = _normalize_features(features)
    X = pd.DataFrame([norm_f])[FEATURE_ORDER]
    return float(model.predict_proba(X)[0, 1])

def generate_counterfactuals(features: dict, target_risk: float = 0.20, max_iterations: int = 100) -> Dict[str, Any]:
    """
    Find minimal changes to modifiable features to reach target risk.
    """
    model = load_model()
    norm_features = _normalize_features(features)
    current_risk = predict_risk(model, norm_features)
    
    if current_risk <= target_risk:
        return {
            'current_risk': round(current_risk * 100, 2),
            'new_risk': round(current_risk * 100, 2),
            'target_risk': round(target_risk * 100, 2),
            'total_risk_reduction': 0.0,
            'changes': [],
            'message': 'Already at or below target risk.'
        }
    
    changes = []
    modified_features = norm_features.copy()
    
    # Try each modifiable feature
    if norm_features.get('smoking', 0) == 1:
        modified_features['smoking'] = 0
        new_risk = predict_risk(model, modified_features)
        if new_risk < current_risk:
            changes.append({
                'feature': 'Smoking',
                'from': 'Yes',
                'to': 'No',
                'risk_reduction': round((current_risk - new_risk) * 100, 2)
            })
            current_risk = new_risk
    
    # Reduce systolic BP in steps of 5
    current_sbp = int(norm_features.get('systolic_bp', 120))
    target_sbp = 120
    if current_sbp > target_sbp:
        for new_sbp in range(current_sbp - 5, target_sbp - 1, -5):
            modified_features['systolic_bp'] = new_sbp
            modified_features['pulse_pressure'] = new_sbp - int(norm_features.get('diastolic_bp', 80))
            new_risk = predict_risk(model, modified_features)
            if new_risk < current_risk:
                changes.append({
                    'feature': 'Systolic BP',
                    'from': f"{current_sbp} mmHg",
                    'to': f"{new_sbp} mmHg",
                    'risk_reduction': round((current_risk - new_risk) * 100, 2)
                })
                current_sbp = new_sbp
                current_risk = new_risk
                if current_risk <= target_risk:
                    break
    
    # Add physical activity
    if norm_features.get('physical_activity', 0) == 0:
        modified_features['physical_activity'] = 1
        new_risk = predict_risk(model, modified_features)
        if new_risk < current_risk:
            changes.append({
                'feature': 'Physical Activity',
                'from': 'No',
                'to': 'Yes',
                'risk_reduction': round((current_risk - new_risk) * 100, 2)
            })
            current_risk = new_risk
    
    initial_risk = predict_risk(model, norm_features)
    return {
        'current_risk': round(initial_risk * 100, 2),
        'new_risk': round(current_risk * 100, 2),
        'target_risk': round(target_risk * 100, 2),
        'total_risk_reduction': round((initial_risk - current_risk) * 100, 2),
        'changes': changes,
        'message': 'Changes identified.'
    }

if __name__ == "__main__":
    sample_features = {
        'age': 45, 'gender': 2, 'height_cm': 175, 'weight_kg': 80,
        'bmi': 26.12, 'systolic_bp': 145, 'diastolic_bp': 92,
        'pulse_pressure': 53, 'cholesterol': 2, 'glucose': 1,
        'smoking': 1, 'alcohol': 0, 'physical_activity': 0
    }
    
    result = generate_counterfactuals(sample_features)
    print(f"Current Risk: {result['current_risk']}%")
    print(f"New Risk: {result['new_risk']}%")
    print(f"Total Reduction: {result['total_risk_reduction']}%")
    print(f"\nChanges:")
    for change in result['changes']:
        print(f"  {change['feature']}: {change['from']} → {change['to']} (reduces risk by {change['risk_reduction']}%)")