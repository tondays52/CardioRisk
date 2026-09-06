"""
Save test patient data to database for dashboard
Generates diverse patient profiles with different risk levels.
"""
import requests
import json
import time
from typing import Any, Dict, List

# Test patients with diverse profiles
test_cases: List[Dict[str, Any]] = [
    {
        'patient': 'Patient 1 (Young, Healthy, Low Risk)',
        'data': {
            'age': 25,
            'gender': 2,
            'height_cm': 175,
            'weight_kg': 70,
            'systolic_bp': 110,
            'diastolic_bp': 70,
            'cholesterol': 1,
            'glucose': 1,
            'smoking': 0,
            'alcohol': 0,
            'physical_activity': 1,
            'ecg': {'heart_rate_bpm': 68, 'mean_qrs_duration_ms': 90},
            'heart_sound': {'heart_rate_bpm': 68, 'snr_estimate': 6.0, 'mean_spectral_centroid': 200}
        }
    },
    {
        'patient': 'Patient 2 (Middle Age, Moderate Risk)',
        'data': {
            'age': 55,
            'gender': 1,
            'height_cm': 168,
            'weight_kg': 75,
            'systolic_bp': 135,
            'diastolic_bp': 85,
            'cholesterol': 2,
            'glucose': 2,
            'smoking': 0,
            'alcohol': 1,
            'physical_activity': 0,
            'ecg': {'heart_rate_bpm': 78, 'mean_qrs_duration_ms': 100},
            'heart_sound': {'heart_rate_bpm': 78, 'snr_estimate': 4.0, 'mean_spectral_centroid': 350}
        }
    },
    {
        'patient': 'Patient 3 (Elderly, High Risk)',
        'data': {
            'age': 70,
            'gender': 1,
            'height_cm': 165,
            'weight_kg': 85,
            'systolic_bp': 160,
            'diastolic_bp': 95,
            'cholesterol': 3,
            'glucose': 3,
            'smoking': 1,
            'alcohol': 1,
            'physical_activity': 0,
            'ecg': {'heart_rate_bpm': 85, 'mean_qrs_duration_ms': 130},
            'heart_sound': {'heart_rate_bpm': 85, 'snr_estimate': 2.0, 'mean_spectral_centroid': 450}
        }
    },
    {
        'patient': 'Patient 4 (Young Athlete, Very Low Risk)',
        'data': {
            'age': 22,
            'gender': 2,
            'height_cm': 170,
            'weight_kg': 60,
            'systolic_bp': 105,
            'diastolic_bp': 65,
            'cholesterol': 1,
            'glucose': 1,
            'smoking': 0,
            'alcohol': 0,
            'physical_activity': 1,
            'ecg': {'heart_rate_bpm': 55, 'mean_qrs_duration_ms': 85},
            'heart_sound': {'heart_rate_bpm': 55, 'snr_estimate': 7.0, 'mean_spectral_centroid': 180}
        }
    },
    {
        'patient': 'Patient 5 (Obese, Sedentary, High Risk)',
        'data': {
            'age': 45,
            'gender': 2,
            'height_cm': 160,
            'weight_kg': 95,
            'systolic_bp': 140,
            'diastolic_bp': 90,
            'cholesterol': 2,
            'glucose': 2,
            'smoking': 1,
            'alcohol': 0,
            'physical_activity': 0,
            'ecg': {'heart_rate_bpm': 82, 'mean_qrs_duration_ms': 115},
            'heart_sound': {'heart_rate_bpm': 82, 'snr_estimate': 3.0, 'mean_spectral_centroid': 380}
        }
    },
    {
        'patient': 'Patient 6 (Elderly, Controlled Risk)',
        'data': {
            'age': 65,
            'gender': 1,
            'height_cm': 172,
            'weight_kg': 72,
            'systolic_bp': 125,
            'diastolic_bp': 78,
            'cholesterol': 1,
            'glucose': 1,
            'smoking': 0,
            'alcohol': 0,
            'physical_activity': 1,
            'ecg': {'heart_rate_bpm': 72, 'mean_qrs_duration_ms': 100},
            'heart_sound': {'heart_rate_bpm': 72, 'snr_estimate': 5.5, 'mean_spectral_centroid': 280}
        }
    },
    {
        'patient': 'Patient 7 (Young Smoker, Moderate Risk)',
        'data': {
            'age': 30,
            'gender': 1,
            'height_cm': 178,
            'weight_kg': 78,
            'systolic_bp': 125,
            'diastolic_bp': 80,
            'cholesterol': 2,
            'glucose': 1,
            'smoking': 1,
            'alcohol': 1,
            'physical_activity': 1,
            'ecg': {'heart_rate_bpm': 76, 'mean_qrs_duration_ms': 105},
            'heart_sound': {'heart_rate_bpm': 76, 'snr_estimate': 4.5, 'mean_spectral_centroid': 320}
        }
    },
    {
        'patient': 'Patient 8 (Healthy Senior)',
        'data': {
            'age': 68,
            'gender': 2,
            'height_cm': 162,
            'weight_kg': 65,
            'systolic_bp': 118,
            'diastolic_bp': 72,
            'cholesterol': 1,
            'glucose': 1,
            'smoking': 0,
            'alcohol': 0,
            'physical_activity': 1,
            'ecg': {'heart_rate_bpm': 70, 'mean_qrs_duration_ms': 95},
            'heart_sound': {'heart_rate_bpm': 70, 'snr_estimate': 6.0, 'mean_spectral_centroid': 230}
        }
    }
]

def clear_database():
    """Clear existing patients from database."""
    try:
        import sqlite3
        conn = sqlite3.connect('data/cardiorisk.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM assessments')
        cursor.execute('DELETE FROM patients')
        cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("assessments", "patients")')
        conn.commit()
        conn.close()
        print('✅ Database cleared!')
        return True
    except Exception as e:
        print(f'⚠️ Could not clear database: {e}')
        return False

def save_patients():
    print('=' * 60)
    print('Saving diverse patient profiles to database...')
    print('=' * 60)

    # Check if backend is running
    try:
        r = requests.get('http://localhost:8000/health', timeout=2)
        if r.status_code != 200:
            print('❌ Backend not healthy!')
            return
    except:
        print('❌ Backend not running! Start it first: python backend/api/main.py')
        return

    # Clear existing data
    clear_database()

    success_count = 0
    error_count = 0

    for i, test in enumerate(test_cases):
        try:
            print(f"\n📊 {test['patient']}...")
            
            # Get prediction
            r = requests.post('http://localhost:8000/predict', json=test['data'], timeout=10)
            result = r.json()
            
            # Print results with both scores
            risk_score = result.get('risk_score', 0)
            fused_score = result.get('fused_risk_score', 0)
            category = result.get('risk_category', 'Unknown')
            modalities = result.get('modalities_used', [])
            weights = result.get('ensemble_weights', {})
            
            print(f"  📈 Tabular Risk: {risk_score}%")
            print(f"  🔄 Fused Risk:   {fused_score}%")
            print(f"  📊 Category:     {category}")
            print(f"  🧠 Modalities:   {modalities}")
            print(f"  ⚖️ Weights:      {weights}")
            
            # Check if scores are different
            if abs(risk_score - fused_score) > 0.5:
                print(f"  ✅ Ensemble is working! Difference: {abs(risk_score - fused_score):.2f}%")
            else:
                print(f"  ⚠️ Scores are similar (diff: {abs(risk_score - fused_score):.2f}%)")
            
            # Save to database
            save_data = {
                'patient_data': {
                    'age': test['data']['age'],
                    'gender': test['data']['gender'],
                    'height_cm': test['data']['height_cm'],
                    'weight_kg': test['data']['weight_kg']
                },
                'result': result
            }
            
            save_r = requests.post('http://localhost:8000/save-assessment', json=save_data, timeout=10)
            if save_r.status_code == 200:
                success_count += 1
                print(f"  ✅ Saved successfully!")
            else:
                error_count += 1
                print(f"  ❌ Failed to save: {save_r.status_code}")
            
            time.sleep(0.5)
            
        except Exception as e:
            error_count += 1
            print(f"  ❌ Error: {e}")

    print('\n' + '=' * 60)
    print(f'✅ Successfully saved: {success_count} patients')
    print(f'❌ Errors: {error_count} patients')
    print('=' * 60)
    print('\n👨‍⚕️ Check the Doctor Dashboard to view all patients!')

if __name__ == "__main__":
    save_patients()