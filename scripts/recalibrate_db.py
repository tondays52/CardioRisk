"""
Recalibrates existing patient assessment categories in cardiorisk.db
ensuring standard clinical tiers:
- Low Risk (Green #10B981): < 30%
- Moderate Risk (Orange #F59E0B): 30% - 60%
- High Risk (Red #EF4444): >= 60%
"""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cardiorisk.db")
db_path = os.path.abspath(db_path)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Recalibrate categories based on fused_risk_score
cursor.execute('''
    UPDATE assessments
    SET risk_category = CASE
        WHEN fused_risk_score < 30.0 THEN 'Low Risk'
        WHEN fused_risk_score < 60.0 THEN 'Moderate Risk'
        ELSE 'High Risk'
    END
''')
conn.commit()

cursor.execute('SELECT id, patient_id, fused_risk_score, risk_category FROM assessments')
rows = cursor.fetchall()
print("Updated Assessment Records:")
for r in rows:
    print(f"  Patient {r[1]}: Fused Risk = {r[2]}% -> {r[3]}")

conn.close()
print("\n[SUCCESS] Database recalibration complete.")
