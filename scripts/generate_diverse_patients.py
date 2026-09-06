"""
Generate diverse test patients for dashboard
"""
import requests
import time
import sqlite3

def clear_patients():
    conn = sqlite3.connect('data/cardiorisk.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM assessments')
    cursor.execute('DELETE FROM patients')
    cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("assessments", "patients")')
    conn.commit()
    conn.close()
    print('✅ Database cleared!')

patients = [
    {
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
        'physical_activity': 1
    },
    {
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
        'physical_activity': 0
    },
    {
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
        'physical_activity': 0
    },
    {
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
        'physical_activity': 1
    },
    {
        'age': 45,
        'gender': 2,
        'height_cm': 160,
        'weight_kg': 90,
        'systolic_bp': 140,
        'diastolic_bp': 90,
        'cholesterol': 2,
        'glucose': 2,
        'smoking': 1,
        'alcohol': 0,
        'physical_activity': 0
    },
    {
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
        'physical_activity': 1
    }
]

print('=' * 60)
print('Generating Diverse Test Patients')
print('=' * 60)

clear_patients()

for i, patient in enumerate(patients):
    try:
        r = requests.post('http://localhost:8000/predict', json=patient, timeout=10)
        result = r.json()
        
        risk = result.get('fused_risk_score', 0)
        category = result.get('risk_category', 'Unknown')
        
        print(f"Patient {i+1}: Age={patient['age']}, Risk={risk}%, Category={category}")
        
        save_data = {
            'patient_data': {
                'age': patient['age'],
                'gender': patient['gender'],
                'height_cm': patient['height_cm'],
                'weight_kg': patient['weight_kg']
            },
            'result': result
        }
        save_r = requests.post('http://localhost:8000/save-assessment', json=save_data, timeout=10)
        print(f"  ✅ Saved")
        time.sleep(0.5)
    except Exception as e:
        print(f"  ❌ Error: {e}")

print('=' * 60)
print('✅ Done! Check the Doctor Dashboard.')