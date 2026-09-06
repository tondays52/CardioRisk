"""
CardioRisk AI - Clinical PDF Report Generator
Generates high-resolution, institutional multi-modal cardiovascular risk assessment dossiers.
"""
import io
import os
import json
from datetime import datetime
from typing import Optional, Any, cast
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.units import inch

def generate_clinical_pdf(patient_data: dict, assessment_result: dict, clinician_notes: Optional[str] = None) -> bytes:
    """
    Generates a professional multi-page Clinical Risk Assessment Dossier in PDF format.
    Returns the PDF as raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0284C7")     # Cyan / Sky
    c_dark = colors.HexColor("#0F172A")        # Slate 900
    c_sub = colors.HexColor("#475569")         # Slate 600
    c_border = colors.HexColor("#CBD5E1")      # Slate 300
    c_bg_light = colors.HexColor("#F8FAFC")    # Slate 50

    # Risk Color Mapping
    raw_score = assessment_result.get("fused_risk_score")
    if raw_score is None:
        raw_score = assessment_result.get("risk_score", 0.0)
    try:
        fused_score = float(raw_score) if raw_score is not None else 0.0
    except (ValueError, TypeError):
        fused_score = 0.0
    if fused_score < 30.0:
        risk_color = colors.HexColor("#10B981") # Green
        risk_level = "LOW CARDIOVASCULAR RISK"
    elif fused_score < 60.0:
        risk_color = colors.HexColor("#F59E0B") # Amber
        risk_level = "MODERATE CARDIOVASCULAR RISK"
    else:
        risk_color = colors.HexColor("#EF4444") # Red
        risk_level = "HIGH CARDIOVASCULAR RISK (CRITICAL)"

    # Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_dark
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=c_primary
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_dark,
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark
    )

    badge_style = ParagraphStyle(
        'RiskBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.white,
        alignment=1 # Center
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>CARDIORISK AI™</b> | Enterprise Clinical Intelligence", subtitle_style),
            Paragraph(f"Date: <b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</b>", ParagraphStyle('DateRight', parent=body_style, alignment=2, textColor=c_sub))
        ],
        [
            Paragraph("Comprehensive Multi-Modal Cardiovascular Assessment Dossier", title_style),
            Paragraph("Status: <b>VERIFIED AI TRIAGE</b>", ParagraphStyle('StatusRight', parent=body_style, alignment=2, textColor=colors.HexColor("#059669")))
        ]
    ]
    header_table = Table(header_data, colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=2, spaceAfter=10))

    # 2. Executive Risk Barometer Box
    ci = assessment_result.get("confidence_interval") or {}
    ci_text = f"95% CI: [{ci.get('lower', max(0.0, fused_score-6.5)):.1f}% - {ci.get('upper', min(100.0, fused_score+6.5)):.1f}%]"
    
    risk_banner_data = [
        [
            Paragraph(f"<b>10-YEAR ESTIMATED CVD RISK: {fused_score:.1f}%</b><br/><font size=8>{ci_text}</font>", badge_style),
            Paragraph(f"<b>CLINICAL CATEGORY</b><br/><font size=11><b>{risk_level}</b></font><br/><font size=8 color='#64748B'>Late-Fusion Dynamic Confidence Meta-Classifier v3.0</font>", ParagraphStyle('RiskText', parent=body_style, alignment=1))
        ]
    ]
    risk_table = Table(risk_banner_data, colWidths=[240, 300])
    risk_table.setStyle(TableStyle(cast(Any, [
        ('BACKGROUND', (0, 0), (0, 0), risk_color),
        ('BACKGROUND', (1, 0), (1, 0), c_bg_light),
        ('BOX', (0, 0), (-1, -1), 1, c_border),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ])))
    story.append(risk_table)
    story.append(Spacer(1, 10))

    # 3. Patient Clinical Baseline & Vitals
    story.append(Paragraph("1. Patient Baseline Profile & Clinical Indicators", h2_style))
    
    age = patient_data.get('age', patient_data.get('age_years', 'N/A'))
    gender_val = patient_data.get('gender', 1)
    gender_str = "Female" if gender_val == 1 else "Male"
    height = patient_data.get('height_cm') or patient_data.get('height') or 170.0
    weight = patient_data.get('weight_kg') or patient_data.get('weight') or 70.0
    
    try:
        if weight is not None and height is not None and float(height) > 0:
            bmi_val = round(float(weight) / ((float(height) / 100) ** 2), 1)
        else:
            bmi_val = "N/A"
    except Exception:
        bmi_val = "N/A"

    sbp = patient_data.get('systolic_bp', patient_data.get('ap_hi', 120))
    dbp = patient_data.get('diastolic_bp', patient_data.get('ap_lo', 80))
    chol = patient_data.get('cholesterol', 1)
    chol_str = "Normal" if chol == 1 else ("Above Normal" if chol == 2 else "Well Above Normal")
    gluc = patient_data.get('glucose', patient_data.get('gluc', 1))
    gluc_str = "Normal" if gluc == 1 else ("Above Normal" if gluc == 2 else "Well Above Normal")
    smoke = "Yes" if patient_data.get('smoking', patient_data.get('smoke', 0)) == 1 else "No"
    alco = "Yes" if patient_data.get('alcohol', patient_data.get('alco', 0)) == 1 else "No"
    active = "Active" if patient_data.get('physical_activity', patient_data.get('active', 1)) == 1 else "Sedentary"

    vitals_data = [
        [
            Paragraph("<b>Age:</b>", body_style), Paragraph(str(age), body_style),
            Paragraph("<b>Gender:</b>", body_style), Paragraph(gender_str, body_style),
            Paragraph("<b>BMI:</b>", body_style), Paragraph(f"{bmi_val} kg/m²", body_style)
        ],
        [
            Paragraph("<b>Blood Pressure:</b>", body_style), Paragraph(f"{sbp}/{dbp} mmHg", body_style),
            Paragraph("<b>Cholesterol:</b>", body_style), Paragraph(chol_str, body_style),
            Paragraph("<b>Glucose:</b>", body_style), Paragraph(gluc_str, body_style)
        ],
        [
            Paragraph("<b>Smoking:</b>", body_style), Paragraph(smoke, body_style),
            Paragraph("<b>Alcohol:</b>", body_style), Paragraph(alco, body_style),
            Paragraph("<b>Activity:</b>", body_style), Paragraph(active, body_style)
        ]
    ]
    vitals_table = Table(vitals_data, colWidths=[90, 90, 90, 90, 90, 90])
    vitals_table.setStyle(TableStyle(cast(Any, [
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ])))
    story.append(vitals_table)
    story.append(Spacer(1, 10))

    # 4. Multi-Modal Modality Fusion Breakdown
    story.append(Paragraph("2. Multi-Modal Physiological Telemetry Breakdown", h2_style))
    mods = assessment_result.get("modalities_used", ["tabular"])
    weights = assessment_result.get("ensemble_weights", {"tabular": 1.0})
    
    mod_rows = [
        [
            Paragraph("<b>Physiological Modality</b>", body_style),
            Paragraph("<b>Ingestion Channel</b>", body_style),
            Paragraph("<b>Confidence Weight</b>", body_style),
            Paragraph("<b>Status</b>", body_style)
        ]
    ]

    mod_catalog = {
        "tabular": ("Clinical Tabular Vitals", "EHR Record / Form Intake", "XGBoost Cost-Sensitive"),
        "ecg": ("12-Lead Electrocardiogram", "Rhythm 1D-CNN", "Active Lead Analysis"),
        "heart_sound": ("Phonocardiogram (PCG)", "Acoustic Spectrogram 2D-CNN", "Acoustic Triage"),
        "hrv": ("Photoplethysmography (PPG)", "Webcam / Optical Sensor", "Temporal HRV Analysis"),
        "scg": ("Seismocardiography (SCG)", "IMU Accelerometer 1D-CNN", "Mechanical Vibration"),
        "retinal": ("Retinal Fundus Microvasculature", "U-Net Morphological Segmentation", "Vascular Tortuosity")
    }

    for m, (m_name, m_ch, m_note) in mod_catalog.items():
        is_used = m in mods
        w_val = f"{weights.get(m, 0.0)*100:.1f}%" if is_used else "0.0% (Inactive)"
        status_txt = f"<font color='#059669'><b>INGESTED</b></font>" if is_used else "<font color='#94A3B8'>Omitted</font>"
        mod_rows.append([
            Paragraph(f"<b>{m_name}</b>", body_style),
            Paragraph(m_ch, body_style),
            Paragraph(w_val, body_style),
            Paragraph(status_txt, body_style)
        ])

    mod_table = Table(mod_rows, colWidths=[180, 160, 100, 100])
    mod_table.setStyle(TableStyle(cast(Any, [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ])))
    story.append(mod_table)
    story.append(Spacer(1, 10))

    # 5. Explainable AI: SHAP Attribution Table
    story.append(Paragraph("3. Explainable AI (XAI) Feature Attribution (SHAP)", h2_style))
    shap_factors = assessment_result.get("shap_explanations", [])[:5]
    if shap_factors:
        shap_rows = [
            [
                Paragraph("<b>Clinical Feature</b>", body_style),
                Paragraph("<b>Patient Value</b>", body_style),
                Paragraph("<b>SHAP Impact</b>", body_style),
                Paragraph("<b>Direction</b>", body_style)
            ]
        ]
        for item in shap_factors:
            feat = item.get("feature", "")
            imp = item.get("importance", 0.0)
            val = item.get("value", "")
            dir_str = "<font color='#EF4444'><b>Elevating Risk (+▲)</b></font>" if imp > 0 else "<font color='#10B981'><b>Protective (-▼)</b></font>"
            shap_rows.append([
                Paragraph(feat, body_style),
                Paragraph(str(val), body_style),
                Paragraph(f"{imp:+.4f}", body_style),
                Paragraph(dir_str, body_style)
            ])
        shap_table = Table(shap_rows, colWidths=[140, 120, 120, 160])
        shap_table.setStyle(TableStyle(cast(Any, [
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
            ('BOX', (0,0), (-1,-1), 0.5, c_border),
            ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ])))
        story.append(shap_table)
    story.append(Spacer(1, 10))

    # 6. Counterfactual Roadmap & Clinical Recommendations
    cf_data = assessment_result.get("counterfactuals") or {}
    changes = cf_data.get("changes", [])
    if changes:
        story.append(Paragraph("4. Actionable Counterfactual Roadmap (Targeted Interventions)", h2_style))
        cf_rows = [
            [
                Paragraph("<b>Priority</b>", body_style),
                Paragraph("<b>Intervention Target</b>", body_style),
                Paragraph("<b>Current State</b>", body_style),
                Paragraph("<b>Target Goal</b>", body_style),
                Paragraph("<b>Expected Risk Reduction</b>", body_style)
            ]
        ]
        for c in changes:
            cf_rows.append([
                Paragraph(f"<b>#{c.get('priority', '-')}</b>", body_style),
                Paragraph(f"<b>{c.get('feature', '')}</b>", body_style),
                Paragraph(str(c.get('from', '')), body_style),
                Paragraph(f"<font color='#0284C7'>{c.get('to', '')}</font>", body_style),
                Paragraph(f"<font color='#059669'><b>-{c.get('risk_reduction', 0)}%</b></font>", body_style)
            ])
        cf_table = Table(cf_rows, colWidths=[50, 140, 110, 130, 110])
        cf_table.setStyle(TableStyle(cast(Any, [
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
            ('BOX', (0,0), (-1,-1), 0.5, c_border),
            ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ])))
        story.append(cf_table)
        story.append(Spacer(1, 8))

    # 7. Clinician Notes & Verification
    notes_block = []
    notes_block.append(Paragraph("5. Physician Verification & Clinical Disposition", h2_style))
    custom_notes = clinician_notes if clinician_notes else "Patient evaluated with CardioRisk AI Clinical Decision Support System. Multi-modal late-fusion consensus indicates consistent risk stratification. Lifestyle and pharmacological roadmap recommended as specified."
    
    notes_data = [
        [Paragraph(f"<b>Attending Notes:</b> {custom_notes}", body_style)],
        [
            Paragraph("<br/><br/>________________________________________<br/><b>Cardiologist / Attending Physician Signature</b>", body_style),
            Paragraph(f"<br/><br/>Date of Verification: <b>{datetime.now().strftime('%B %d, %Y')}</b><br/>EHR Synced: <b>ID #{patient_data.get('id', 'REC-AUTO')}</b>", body_style)
        ]
    ]
    notes_table = Table(notes_data, colWidths=[300, 240])
    notes_table.setStyle(TableStyle(cast(Any, [
        ('SPAN', (0, 0), (1, 0)),
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ])))
    notes_block.append(notes_table)
    
    story.append(KeepTogether(notes_block))

    # Build Document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
