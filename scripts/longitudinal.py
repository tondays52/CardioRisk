"""
CardioRisk AI - Longitudinal Risk Tracking
Tracks patient risk over multiple visits.
"""
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

DB_PATH = "data/cardiorisk.db"

def get_patient_risk_history(patient_id=None, limit=50):
    """
    Get risk history for a patient or all patients.
    """
    conn = sqlite3.connect(DB_PATH)
    
    if patient_id:
        query = '''
            SELECT p.id as patient_id, p.age, p.gender,
                   a.assessment_date, a.risk_score, a.fused_risk_score,
                   a.risk_category, a.confidence_lower, a.confidence_upper
            FROM assessments a
            JOIN patients p ON a.patient_id = p.id
            WHERE p.id = ?
            ORDER BY a.assessment_date
        '''
        df = pd.read_sql_query(query, conn, params=[patient_id])
    else:
        query = '''
            SELECT p.id as patient_id, p.age, p.gender,
                   a.assessment_date, a.risk_score, a.fused_risk_score,
                   a.risk_category, a.confidence_lower, a.confidence_upper
            FROM assessments a
            JOIN patients p ON a.patient_id = p.id
            ORDER BY a.assessment_date
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=[limit])
    
    conn.close()
    return df

def get_all_patients():
    """Get list of unique patients with visit counts."""
    conn = sqlite3.connect(DB_PATH)
    query = '''
        SELECT p.id, p.age, p.gender, COUNT(a.id) as visit_count
        FROM patients p
        LEFT JOIN assessments a ON p.id = a.patient_id
        GROUP BY p.id
        ORDER BY p.id
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_risk_trend_chart(history_df):
    """Create risk trend chart from patient history."""
    if history_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history_df['assessment_date'],
        y=history_df['fused_risk_score'],
        mode='lines+markers',
        name='Fused Risk',
        line=dict(color='#E11D48', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['assessment_date'].tolist() + history_df['assessment_date'].tolist()[::-1],
        y=history_df['confidence_upper'].tolist() + history_df['confidence_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(225, 29, 72, 0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% CI'
    ))
    
    fig.add_hrect(y0=0, y1=10, fillcolor="#10B981", opacity=0.1, line_width=0)
    fig.add_hrect(y0=10, y1=20, fillcolor="#F59E0B", opacity=0.1, line_width=0)
    fig.add_hrect(y0=20, y1=100, fillcolor="#EF4444", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title="Risk Score Trend Over Time",
        xaxis_title="Assessment Date",
        yaxis_title="Risk Score (%)",
        yaxis_range=[0, 100],
        height=400,
        hovermode='x unified'
    )
    
    return fig

def analyze_trend(history_df):
    """Analyze risk trend."""
    if len(history_df) < 2:
        return {
            'trend': 'Insufficient data',
            'change': 0,
            'message': 'Need at least 2 assessments to analyze trend.'
        }
    
    first_risk = history_df['fused_risk_score'].iloc[0]
    last_risk = history_df['fused_risk_score'].iloc[-1]
    change = round(last_risk - first_risk, 2)
    
    if change < -5:
        trend = 'Improving'
        message = f'Risk decreased by {abs(change)}% since first assessment.'
    elif change > 5:
        trend = 'Worsening'
        message = f'Risk increased by {change}% since first assessment.'
    else:
        trend = 'Stable'
        message = f'Risk relatively stable (change: {change}%).'
    
    return {
        'trend': trend,
        'change': change,
        'message': message,
        'first_risk': first_risk,
        'last_risk': last_risk,
        'num_visits': len(history_df)
    }

if __name__ == "__main__":
    print("Longitudinal Risk Tracking Module")
    print("=" * 40)
    
    patients = get_all_patients()
    print(f"\nTotal patients: {len(patients)}")
    
    if not patients.empty:
        print("\nPatients:")
        print(patients.to_string(index=False))
        
        for patient_id in patients['id'].tolist():
            history = get_patient_risk_history(patient_id)
            print(f"\nRisk history for Patient {patient_id}:")
            if history.empty:
                print("  No assessments found.")
            else:
                print(history[['assessment_date', 'fused_risk_score', 'risk_category']].to_string(index=False))
                trend = analyze_trend(history)
                print(f"  Trend: {trend['trend']}")
                print(f"  Message: {trend['message']}")
    else:
        print("No patient data available. Save some assessments first.")