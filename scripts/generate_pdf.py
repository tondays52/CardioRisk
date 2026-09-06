"""
CardioRisk AI - PDF Report Generator
Generates professional PDF reports from prediction results.
"""
import os
from typing import Any, cast
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def create_gauge_chart(risk_score, output_path="models/gauge.png"):
    """Create a gauge chart for the PDF."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    
    # Color zones
    colors = ['#10B981', '#F59E0B', '#EF4444']
    zones = [(0, 0.1), (0.1, 0.2), (0.2, 1.0)]
    
    for (start, end), color in zip(zones, colors):
        zone_theta = np.linspace(np.pi * start, np.pi * end, 50)
        ax.fill_between(zone_theta, 0.85, 1.0, color=color, alpha=0.6)
    
    # Needle
    needle_angle = np.pi * (1 - risk_score / 100)
    ax.plot([needle_angle, needle_angle], [0, 0.85], color='black', linewidth=3)
    ax.plot([needle_angle], [0.85], 'ko', markersize=10)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

def generate_pdf_report(result: dict, output_path=None):
    """
    Generate a professional PDF report from prediction results.
    Saves to reports/ folder by default with timestamp.
    """
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Generate filename with timestamp if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/CardioRisk_Report_{timestamp}.pdf"
    
    # Create gauge chart
    risk_score = result.get('fused_risk_score', result.get('risk_score', 0))
    create_gauge_chart(risk_score)
    
    # PDF setup
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#E11D48'),
        alignment=TA_CENTER,
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=HexColor('#64748B'),
        alignment=TA_CENTER,
        spaceAfter=20
    )
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=HexColor('#1E293B'),
        spaceBefore=20,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        textColor=HexColor('#1E293B')
    )
    
    # Title
    story.append(Paragraph("CardioRisk AI", title_style))
    story.append(Paragraph("Cardiovascular Risk Assessment Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=HexColor('#E11D48')))
    story.append(Spacer(1, 20))
    
    # Date
    story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y %H:%M')}", body_style))
    story.append(Spacer(1, 20))
    
    # Risk Score Section
    story.append(Paragraph("Risk Assessment Summary", section_style))
    
    risk_category = result.get('risk_category', 'Unknown')
    if 'Low' in risk_category:
        risk_color = '#10B981'
    elif 'Moderate' in risk_category:
        risk_color = '#F59E0B'
    else:
        risk_color = '#EF4444'
    
    risk_data = [
        ['Risk Category', risk_category],
        ['Fused Risk Score', f"{risk_score}%"],
        ['Tabular Risk Score', f"{result.get('risk_score', 0)}%"],
    ]
    
    ci = result.get('confidence_interval', {})
    if ci:
        risk_data.append(['95% Confidence Interval', f"[{ci.get('lower', 0)}% - {ci.get('upper', 0)}%]"])
    
    risk_data.append(['Modalities Used', ', '.join(result.get('modalities_used', []))])
    
    risk_table = Table(risk_data, colWidths=[2.5*inch, 3.5*inch])
    risk_table.setStyle(TableStyle(cast(Any, [
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#1E293B')),
        ('TEXTCOLOR', (1, 1), (1, 1), HexColor(risk_color)),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ])))
    story.append(risk_table)
    story.append(Spacer(1, 20))
    
    # Gauge image
    if os.path.exists("models/gauge.png"):
        img = Image("models/gauge.png", width=4*inch, height=2*inch)
        story.append(img)
    story.append(Spacer(1, 20))
    
    # Features Section
    features = result.get('features_used', {})
    if features:
        story.append(Paragraph("Patient Features", section_style))
        
        feature_data = [['Feature', 'Value']]
        for key, value in features.items():
            feature_data.append([key.replace('_', ' ').title(), str(value)])
        
        feature_table = Table(feature_data, colWidths=[3*inch, 3*inch])
        feature_table.setStyle(TableStyle(cast(Any, [
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#E11D48')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#E2E8F0')),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#F8FAFC')]),
        ])))
        story.append(feature_table)
        story.append(Spacer(1, 20))
    
    # Modality Analyses
    analyses = [
        ('HRV Analysis', result.get('hrv_analysis')),
        ('ECG Analysis', result.get('ecg_analysis')),
        ('Heart Sound Analysis', result.get('heart_sound_analysis')),
        ('SCG Analysis', result.get('scg_analysis')),
    ]
    
    for name, analysis in analyses:
        if analysis:
            story.append(Paragraph(name, section_style))
            for key, value in analysis.items():
                if isinstance(value, str) and len(value) < 100:
                    story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", body_style))
            story.append(Spacer(1, 10))
    
    # Recommendations
    story.append(Paragraph("Recommendations", section_style))
    recommendations = []
    
    bp = features.get('systolic_bp', 120)
    if bp > 140:
        recommendations.append("Your blood pressure is elevated. Consider reducing sodium intake and consulting a physician.")
    
    if features.get('smoking', 0) == 1:
        recommendations.append("Quitting smoking can significantly reduce your cardiovascular risk.")
    
    bmi = features.get('bmi', 22)
    if bmi > 30:
        recommendations.append("Your BMI indicates obesity. Weight loss through diet and exercise is recommended.")
    
    if features.get('physical_activity', 0) == 0:
        recommendations.append("Regular physical activity (150 min/week) can improve heart health.")
    
    if not recommendations:
        recommendations.append("Your risk factors are within acceptable ranges. Maintain a healthy lifestyle.")
    
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", body_style))
    
    story.append(Spacer(1, 30))
    
    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#CBD5E1')))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is for academic and educational purposes only. "
        "It is not intended to replace professional medical diagnosis or treatment. "
        "Please consult a qualified healthcare provider for medical advice.",
        ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, textColor=HexColor('#64748B'))
    ))
    
    # Build PDF
    doc.build(story)
    print(f"PDF report saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Test with sample result
    sample_result = {
        'risk_score': 80.99,
        'fused_risk_score': 51.35,
        'risk_category': 'High Risk',
        'confidence_interval': {'lower': 12.44, 'upper': 90.26},
        'modalities_used': ['tabular', 'hrv', 'ecg', 'heart_sound', 'scg'],
        'features_used': {
            'age': 45, 'gender': 2, 'height_cm': 175, 'weight_kg': 80,
            'bmi': 26.12, 'systolic_bp': 145, 'diastolic_bp': 92,
            'pulse_pressure': 53, 'cholesterol': 2, 'glucose': 1,
            'smoking': 1, 'alcohol': 0, 'physical_activity': 0
        },
        'hrv_analysis': {'heart_rate_bpm': 72.1, 'sdnn_ms': 35.5, 'autonomic_assessment': 'Moderate HRV'},
        'ecg_analysis': {'heart_rate_bpm': 72.0, 'ecg_assessment': 'ECG appears normal'},
        'heart_sound_analysis': {'heart_rate_bpm': 72.0, 'heart_sound_assessment': 'Heart sound appears normal'},
        'scg_analysis': {'heart_rate_bpm': 72.0, 'scg_assessment': 'SCG appears normal'}
    }
    
    generate_pdf_report(sample_result)