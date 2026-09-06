"""
CardioRisk AI - Database Module
SQLite-based patient record storage.
"""
import sqlite3
import json
from datetime import datetime
import os
from typing import Any, Optional, Dict, List

_backend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_backend_dir)
DB_PATH = os.path.join(_project_root, "data", "cardiorisk.db")

def init_db():
    """Initialize the database with required tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender INTEGER,
            height_cm REAL,
            weight_kg REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Assessments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            risk_score REAL,
            fused_risk_score REAL,
            risk_category TEXT,
            confidence_lower REAL,
            confidence_upper REAL,
            modalities_used TEXT,
            raw_data TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("[SUCCESS] Database initialized.")

def save_assessment(patient_data: dict, result: dict):
    """
    Save an assessment to the database.
    Creates a new patient or uses existing one.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Extract patient data
    age = patient_data.get('age', 0)
    gender = patient_data.get('gender', 0)
    height_cm = patient_data.get('height_cm', 0)
    weight_kg = patient_data.get('weight_kg', 0)
    
    # If age is 0, try to get it from features
    if age == 0 or age is None:
        features = result.get('features_used', {})
        age = features.get('age', features.get('age_years', 0))
        gender = features.get('gender', 0)
        height_cm = features.get('height', features.get('height_cm', 0))
        weight_kg = features.get('weight', features.get('weight_kg', 0))
    
    # Insert patient
    cursor.execute('''
        INSERT INTO patients (age, gender, height_cm, weight_kg)
        VALUES (?, ?, ?, ?)
    ''', (age, gender, height_cm, weight_kg))
    patient_id = cursor.lastrowid
    
    # Extract result data
    risk_score = result.get('risk_score', 0)
    fused_risk_score = result.get('fused_risk_score', 0)
    risk_category = result.get('risk_category', 'Unknown')
    confidence_interval = result.get('confidence_interval', {})
    confidence_lower = confidence_interval.get('lower', 0)
    confidence_upper = confidence_interval.get('upper', 0)
    modalities_used = result.get('modalities_used', [])
    modalities_json = json.dumps(modalities_used)
    result_json = json.dumps(result)
    
    # Insert assessment
    cursor.execute('''
        INSERT INTO assessments (
            patient_id, risk_score, fused_risk_score, risk_category,
            confidence_lower, confidence_upper, modalities_used, raw_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        patient_id,
        risk_score,
        fused_risk_score,
        risk_category,
        confidence_lower,
        confidence_upper,
        modalities_json,
        result_json
    ))
    
    conn.commit()
    conn.close()
    print(f"[SUCCESS] Saved patient {patient_id}: age={age}, gender={gender}")
    return patient_id

def get_all_assessments(limit=100):
    """Retrieve all assessments with patient data from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            p.id as patient_id,
            p.age,
            p.gender,
            a.assessment_date as date,
            a.risk_score,
            a.fused_risk_score,
            a.risk_category,
            a.confidence_lower,
            a.confidence_upper,
            a.modalities_used
        FROM assessments a
        JOIN patients p ON a.patient_id = p.id
        ORDER BY a.assessment_date DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    assessments = []
    for row in rows:
        assessments.append({
            'patient_id': row[0],
            'age': row[1],
            'gender': row[2],
            'date': row[3],
            'risk_score': row[4],
            'fused_risk_score': row[5],
            'risk_category': row[6],
            'confidence_lower': row[7],
            'confidence_upper': row[8],
            'modalities_used': json.loads(row[9]) if row[9] else []
        })
    
    return assessments

def get_assessment_by_id(assessment_id: int):
    """Retrieve complete assessment payload including raw data and patient vitals."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            a.id, a.patient_id, a.assessment_date, a.risk_score,
            a.fused_risk_score, a.risk_category, a.confidence_lower,
            a.confidence_upper, a.modalities_used, a.raw_data,
            p.age, p.gender, p.height_cm, p.weight_kg, p.name
        FROM assessments a
        JOIN patients p ON a.patient_id = p.id
        WHERE a.id = ?
    ''', (assessment_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    raw_res = json.loads(row[9]) if row[9] else {}
    return {
        'id': row[0],
        'patient_id': row[1],
        'date': row[2],
        'risk_score': row[3],
        'fused_risk_score': row[4],
        'risk_category': row[5],
        'confidence_lower': row[6],
        'confidence_upper': row[7],
        'modalities_used': json.loads(row[8]) if row[8] else [],
        'assessment_result': raw_res,
        'patient_data': {
            'id': row[1],
            'age': row[10],
            'gender': row[11],
            'height_cm': row[12],
            'weight_kg': row[13],
            'name': row[14] or f"Patient #{row[1]}"
        }
    }

def get_patient_trajectory(patient_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve chronological longitudinal trajectory for a specific patient."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            a.id, a.assessment_date, a.risk_score, a.fused_risk_score,
            a.risk_category, a.confidence_lower, a.confidence_upper,
            a.modalities_used, a.raw_data
        FROM assessments a
        WHERE a.patient_id = ?
        ORDER BY a.assessment_date ASC
    ''', (patient_id,))
    rows = cursor.fetchall()
    
    cursor.execute('SELECT id, name, age, gender, height_cm, weight_kg FROM patients WHERE id = ?', (patient_id,))
    pat_row = cursor.fetchone()
    conn.close()
    
    if not pat_row:
        return None
        
    patient_info = {
        'id': pat_row[0],
        'name': pat_row[1] or f"Patient #{pat_row[0]}",
        'age': pat_row[2],
        'gender': pat_row[3],
        'height_cm': pat_row[4],
        'weight_kg': pat_row[5]
    }
    
    visits = []
    for r in rows:
        raw_res = json.loads(r[8]) if r[8] else {}
        features = raw_res.get('features_used', {})
        visits.append({
            'assessment_id': r[0],
            'date': r[1],
            'tabular_risk': r[2],
            'fused_risk': r[3],
            'risk_category': r[4],
            'confidence_interval': {'lower': r[5], 'upper': r[6]},
            'modalities': json.loads(r[7]) if r[7] else [],
            'systolic_bp': features.get('ap_hi', 120),
            'diastolic_bp': features.get('ap_lo', 80),
            'bmi': features.get('bmi', round(pat_row[5]/((pat_row[4]/100)**2), 1) if pat_row[4] else 24.0),
            'cholesterol': features.get('cholesterol', 1),
            'glucose': features.get('gluc', 1)
        })
        
    return {
        'patient': patient_info,
        'total_visits': len(visits),
        'visits': visits
    }

def get_distinct_patients():
    """Retrieve distinct patients with their summary assessment stats."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            p.id, p.name, p.age, p.gender, p.height_cm, p.weight_kg,
            COUNT(a.id) as visit_count,
            MAX(a.assessment_date) as last_visit,
            AVG(a.fused_risk_score) as avg_risk,
            (SELECT fused_risk_score FROM assessments WHERE patient_id = p.id ORDER BY assessment_date DESC LIMIT 1) as latest_risk,
            (SELECT risk_category FROM assessments WHERE patient_id = p.id ORDER BY assessment_date DESC LIMIT 1) as latest_category
        FROM patients p
        LEFT JOIN assessments a ON p.id = a.patient_id
        GROUP BY p.id
        ORDER BY p.id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    patients = []
    for r in rows:
        patients.append({
            'patient_id': r[0],
            'name': r[1] or f"Patient #{r[0]}",
            'age': r[2],
            'gender': r[3],
            'height_cm': r[4],
            'weight_kg': r[5],
            'visit_count': r[6],
            'last_visit': r[7],
            'avg_risk': round(r[8], 1) if r[8] is not None else 0.0,
            'latest_risk': round(r[9], 1) if r[9] is not None else 0.0,
            'latest_category': r[10] or 'N/A'
        })
    return patients

def get_statistics():
    """Retrieve summary statistics for assessments."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM assessments')
    total = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM assessments WHERE risk_category LIKE '%High%'")
    high_risk = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM assessments WHERE risk_category LIKE '%Mod%'")
    moderate_risk = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM assessments WHERE risk_category LIKE '%Low%'")
    low_risk = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        'total': total,
        'high_risk': high_risk,
        'moderate_risk': moderate_risk,
        'low_risk': low_risk
    }

def clear_database():
    """Clear all data from the database (for testing)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM assessments')
    cursor.execute('DELETE FROM patients')
    cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("assessments", "patients")')
    conn.commit()
    conn.close()
    print("[SUCCESS] Database cleared!")

if __name__ == "__main__":
    init_db()
    stats = get_statistics()
    print(f"Total assessments: {stats['total']}")
    print(f"High risk: {stats['high_risk']}")
    print(f"Moderate risk: {stats['moderate_risk']}")
    print(f"Low risk: {stats['low_risk']}")