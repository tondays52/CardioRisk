import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# Initialize Widescreen 16:9 Presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# High-Contrast Clinical Color Palette
BG_DARK      = RGBColor(11, 19, 43)        # Slate Navy (#0B132B)
CARD_BG      = RGBColor(28, 37, 65)        # Card Navy (#1C2541)
ACCENT_CYAN  = RGBColor(0, 168, 232)       # Electric Cyan (#00A8E8)
ACCENT_CORAL = RGBColor(230, 57, 70)       # Alert Coral (#E63946)
ACCENT_GREEN = RGBColor(46, 196, 182)      # Diagnostic Mint (#2EC4B6)
TEXT_WHITE   = RGBColor(248, 249, 250)     # Primary White
TEXT_MUTED   = RGBColor(156, 163, 175)     # Subtitle Slate Gray
BORDER_COL   = RGBColor(58, 80, 107)

def apply_background(slide):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_DARK
    bg.line.color.rgb = BG_DARK

def add_header(slide, title_text, category="NORTH SOUTH UNIVERSITY • CSE 499A SENIOR CAPSTONE DEFENSE"):
    # Category / Super-title
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.name = "Arial"
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_CYAN

    # Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

def create_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER_COL):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1)
    return shape

# ==========================================================
# SLIDE 1: Title Slide
# ==========================================================
s1 = prs.slides.add_slide(blank_layout)
apply_background(s1)

tb_main = s1.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.33), Inches(2.2))
tf1 = tb_main.text_frame
tf1.word_wrap = True

p0 = tf1.paragraphs[0]
p0.text = "NORTH SOUTH UNIVERSITY • DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING"
p0.font.size = Pt(11)
p0.font.bold = True
p0.font.color.rgb = ACCENT_CYAN
p0.space_after = Pt(10)

p1 = tf1.add_paragraph()
p1.text = "CardioRisk AI"
p1.font.size = Pt(44)
p1.font.bold = True
p1.font.color.rgb = TEXT_WHITE

p2 = tf1.add_paragraph()
p2.text = "A Multimodal Deep Learning System for Accessible Cardiovascular Risk Assessment"
p2.font.size = Pt(18)
p2.font.color.rgb = TEXT_MUTED

create_card(s1, Inches(1.0), Inches(4.2), Inches(11.33), Inches(2.4))
tb_meta = s1.shapes.add_textbox(Inches(1.3), Inches(4.4), Inches(10.7), Inches(2.0))
tf_m = tb_meta.text_frame

p_meta_title = tf_m.paragraphs[0]
p_meta_title.text = "CSE 499A: SENIOR DESIGN CAPSTONE DEFENSE"
p_meta_title.font.bold = True
p_meta_title.font.size = Pt(14)
p_meta_title.font.color.rgb = ACCENT_GREEN
p_meta_title.space_after = Pt(8)

p_cand = tf_m.add_paragraph()
p_cand.text = "• Candidate: Hussain Mahabub (Student ID: 2111015642)  |  CSE 499A - Section 10 - Group 10"
p_cand.font.size = Pt(12)
p_cand.font.color.rgb = TEXT_WHITE
p_cand.space_after = Pt(4)

p_sup = tf_m.add_paragraph()
p_sup.text = "• Faculty Supervisor: Shahnewaz Siddique"
p_sup.font.size = Pt(12)
p_sup.font.color.rgb = TEXT_WHITE
p_sup.space_after = Pt(4)

p_mod = tf_m.add_paragraph()
p_mod.text = "• Scope: 6-Modality Architecture • 151k Scaled Cohort • Real-Time DSP • TreeSHAP Explainability • Stacking Late Fusion"
p_mod.font.size = Pt(11)
p_mod.font.color.rgb = TEXT_MUTED

# ==========================================================
# SLIDE 2: Global CVD Crisis & Healthcare Accessibility
# ==========================================================
s2 = prs.slides.add_slide(blank_layout)
apply_background(s2)
add_header(s2, "The Global Cardiovascular Disease Crisis (The Clinical 'What?')")

cards_s2 = [
    ("Global Mortality Scale", "17.9M Annual Deaths", [
        "Cardiovascular disease accounts for 32% of all global fatalities.",
        "Someone experiences an acute cardiac event every 40 seconds.",
        "$555 Billion annual economic burden in the US healthcare system alone.",
        "Developing nations experience over 75% of global cardiovascular deaths."
    ], ACCENT_CORAL),
    ("Traditional Screening Bottlenecks", "Specialized Facility Dependence", [
        "Standard workups demand on-site visits and specialized hospital equipment.",
        "Stationary 12-lead ECG carts are unavailable in rural medical posts.",
        "Severe specialist shortages create dangerous waiting lists for diagnostics.",
        "Mandatory phlebotomy lab tests delay triage decisions by days or weeks."
    ], ACCENT_CYAN),
    ("The Accessibility Gap", "Democratizing Triage", [
        "CardioRisk AI addresses systemic disparities through accessible screening.",
        "Eliminates the requirement for specialized external clinical hardware.",
        "Pairs camera rPPG, acoustic PCG, and SCG via mobile web telemetry.",
        "Early non-invasive screening directly prevents avoidable mortality."
    ], ACCENT_GREEN)
]

for i, (title, sub, bullets, col) in enumerate(cards_s2):
    left = Inches(0.8 + i * 3.9)
    create_card(s2, left, Inches(1.8), Inches(3.7), Inches(5.0))
    tb = s2.shapes.add_textbox(left + Inches(0.25), Inches(2.0), Inches(3.2), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = title.upper()
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = col
    p.space_after = Pt(4)

    p_s = tf.add_paragraph()
    p_s.text = sub
    p_s.font.bold = True
    p_s.font.size = Pt(14)
    p_s.font.color.rgb = TEXT_WHITE
    p_s.space_after = Pt(12)

    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = f"• {b}"
        pb.font.size = Pt(11)
        pb.font.color.rgb = TEXT_MUTED
        pb.space_after = Pt(6)

# ==========================================================
# SLIDE 3: Identified Research Gaps & Proposed Contributions
# ==========================================================
s3 = prs.slides.add_slide(blank_layout)
apply_background(s3)
add_header(s3, "Background, Research Gaps & Primary Contributions (The 'Why?')")

create_card(s3, Inches(0.8), Inches(1.8), Inches(5.7), Inches(5.0))
tb_g = s3.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.5))
tf_g = tb_g.text_frame
tf_g.word_wrap = True

p_gt = tf_g.paragraphs[0]
p_gt.text = "CRITICAL RESEARCH & CLINICAL GAPS"
p_gt.font.bold = True
p_gt.font.size = Pt(12)
p_gt.font.color.rgb = ACCENT_CORAL
p_gt.space_after = Pt(10)

gaps = [
    ("1. Single-Modality Isolation", "Prior literature focuses strictly on isolated tasks (e.g., pure ECG, pure audio, or tabular alone). Isolated models completely miss cross-domain correlations between electrical, acoustic, and vascular mechanics."),
    ("2. Purely Tabular Blindness", "Conventional clinical calculators (Framingham, ASCVD, QRISK3) achieve AUCs of only 0.70-0.75, require laboratory blood panels, and cannot detect acute electrophysiological abnormalities."),
    ("3. Black-Box Opacity", "Standard deep networks fail to explain predictions to clinicians, limiting clinical trust and omitting actionable patient roadmaps."),
    ("4. Deployment Deficit", "Academic models remain isolated in research scripts without production-ready client-server architectures.")
]
for g_title, g_desc in gaps:
    pgt = tf_g.add_paragraph()
    pgt.text = g_title
    pgt.font.bold = True
    pgt.font.size = Pt(11)
    pgt.font.color.rgb = TEXT_WHITE
    pgd = tf_g.add_paragraph()
    pgd.text = g_desc
    pgd.font.size = Pt(10)
    pgd.font.color.rgb = TEXT_MUTED
    pgd.space_after = Pt(6)

create_card(s3, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
tb_c = s3.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
tf_c = tb_c.text_frame
tf_c.word_wrap = True

p_ct = tf_c.paragraphs[0]
p_ct.text = "CARDIORISK AI KEY CONTRIBUTIONS"
p_ct.font.bold = True
p_ct.font.size = Pt(12)
p_ct.font.color.rgb = ACCENT_GREEN
p_ct.space_after = Pt(10)

contribs = [
    ("6-Modality Physiological Integration", "Unifies Tabular biometrics, 12-lead ECG, PCG heart sounds, camera rPPG, SCG kinetic vibrations, and Retinal U-Net imaging."),
    ("151k Scaled Multi-Cohort Classifier", "Expanded tabular training across 151,860 patient profiles, balancing sensitivity to 71.0% via ACC/AHA interaction markers and Youden's J thresholding."),
    ("Dynamic Confidence Late Fusion", "Prevents model dilution by dynamically weighting active channels based on diagnostic certainty rather than rigid static averages."),
    ("TreeSHAP & Counterfactual Actions", "Decomposes risk into exact log-odds contributions and generates quantifiable lifestyle improvement steps (e.g., -8.5% smoking cessation)."),
    ("Enterprise 4-Tier Web Platform", "Fully realized client-server system (Streamlit + FastAPI) with asynchronous processing and HL7/FHIR R4 compliance.")
]
for c_title, c_desc in contribs:
    pct = tf_c.add_paragraph()
    pct.text = c_title
    pct.font.bold = True
    pct.font.size = Pt(11)
    pct.font.color.rgb = TEXT_WHITE
    pcd = tf_c.add_paragraph()
    pcd.text = c_desc
    pcd.font.size = Pt(10)
    pcd.font.color.rgb = TEXT_MUTED
    pcd.space_after = Pt(6)

# ==========================================================
# SLIDE 4: Four-Tier Software Architecture
# ==========================================================
s4 = prs.slides.add_slide(blank_layout)
apply_background(s4)
add_header(s4, "Enterprise 4-Tier Software Architecture (The 'How?')")

arch_tiers = [
    ("Tier 1: Presentation Layer (Streamlit • Port 8501)", "Clinician Command Center, multi-modal intake forms, interactive gauge charts, real-time waveform annotation studio, and interactive What-If scenario simulator.", ACCENT_CYAN),
    ("Tier 2: API Gateway & DSP Layer (FastAPI • Port 8000)", "High-performance asynchronous endpoints, Pydantic data contracts, OpenCV spatial green-channel averaging, and 4th-order Butterworth temporal bandpass filtering.", ACCENT_GREEN),
    ("Tier 3: Model & Logic Layer (TensorFlow & XGBoost)", "Inference runtime housing the 1D-CNN (ECG, SCG), 2D-CNN (PCG), 2D U-Net (Retinal), Cost-Sensitive XGBoost (Tabular), and the Calibrated Stacking Meta-Learner.", ACCENT_CORAL),
    ("Tier 4: Persistence & Interoperability Layer (SQLite / FHIR)", "Encrypted patient database storing longitudinal patient trajectories, multi-visit consultation tracking, audit logging, and HL7/FHIR R4 JSON clinical dossier exports.", ACCENT_CYAN)
]

for i, (t_title, t_desc, col) in enumerate(arch_tiers):
    top = Inches(1.8 + i * 1.25)
    create_card(s4, Inches(0.8), top, Inches(11.73), Inches(1.05))
    tb = s4.shapes.add_textbox(Inches(1.05), top + Inches(0.12), Inches(11.2), Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = t_title
    p1.font.bold = True
    p1.font.size = Pt(13)
    p1.font.color.rgb = col
    p2 = tf.add_paragraph()
    p2.text = t_desc
    p2.font.size = Pt(11)
    p2.font.color.rgb = TEXT_WHITE

# ==========================================================
# SLIDE 5: Modality 1 - Tabular Risk Model Scaled to 151k Patients
# ==========================================================
s5 = prs.slides.add_slide(blank_layout)
apply_background(s5)
add_header(s5, "Modality 1: Tabular Classifier Scaled across 151,860 Records")

create_card(s5, Inches(0.8), Inches(1.8), Inches(6.8), Inches(5.0))
tb_t1 = s5.shapes.add_textbox(Inches(1.05), Inches(2.0), Inches(6.3), Inches(4.5))
tf_t1 = tb_t1.text_frame
tf_t1.word_wrap = True

p = tf_t1.paragraphs[0]
p.text = "LARGE-SCALE COHORT HARMONIZATION & BIOMARKERS"
p.font.bold = True
p.font.size = Pt(12)
p.font.color.rgb = ACCENT_CYAN
p.space_after = Pt(10)

t1_bullets = [
    ("Unified Multi-Source Cohort", "Aggregated 151,860 patient records from Kaggle CVD (70k), Framingham Heart Study, and UCI Cleveland datasets to resolve single-center demographic bias."),
    ("Non-Linear Hemodynamic Features", "Engineered clinical interaction ratios: Mean Arterial Pressure (MAP), Pulse Pressure (arterial stiffness indicator), and ACC/AHA hypertension stages."),
    ("Atherogenic Risk Proxy", "Formulated Cholesterol-to-Glucose ratios and Pack-Years smoking intensity metrics."),
    ("Cost-Sensitive XGBoost", "Trained using scale_pos_weight = 1.095 to aggressively penalize under-diagnosing high-risk cardiovascular patients."),
    ("Youden's J Probability Calibration", "Adjusted decision threshold from rigid 0.50 to J = 0.4769 via 5-fold cross-validated isotonic regression, achieving balanced 71.0% recall.")
]
for h, d in t1_bullets:
    ph = tf_t1.add_paragraph()
    ph.text = f"• {h}: "
    ph.font.bold = True
    ph.font.size = Pt(11)
    ph.font.color.rgb = TEXT_WHITE
    pd = tf_t1.add_paragraph()
    pd.text = f"  {d}"
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_MUTED
    pd.space_after = Pt(4)

metrics_s5 = [
    ("151,860", "Total Patient Records", ACCENT_CYAN),
    ("0.8139", "ROC-AUC Score", ACCENT_GREEN),
    ("71.00%", "Abnormal Recall (Sensitivity)", ACCENT_CORAL),
    ("0.7257", "Balanced F1-Score", ACCENT_CYAN)
]
for i, (m_val, m_lbl, col) in enumerate(metrics_s5):
    top = Inches(1.8 + i * 1.25)
    create_card(s5, Inches(7.9), top, Inches(4.6), Inches(1.1))
    tb_m = s5.shapes.add_textbox(Inches(8.1), top + Inches(0.12), Inches(4.2), Inches(0.85))
    tf_m = tb_m.text_frame
    pv = tf_m.paragraphs[0]
    pv.text = m_val
    pv.font.bold = True
    pv.font.size = Pt(24)
    pv.font.color.rgb = col
    pl = tf_m.add_paragraph()
    pl.text = m_lbl
    pl.font.size = Pt(11)
    pl.font.color.rgb = TEXT_MUTED

# ==========================================================
# SLIDE 6: Modalities 2 & 3 - ECG & PCG Deep Learning
# ==========================================================
s6 = prs.slides.add_slide(blank_layout)
apply_background(s6)
add_header(s6, "Modalities 2 & 3: Electrophysiological & Acoustic Deep Learning")

create_card(s6, Inches(0.8), Inches(1.8), Inches(5.7), Inches(5.0))
tb_e = s6.shapes.add_textbox(Inches(1.05), Inches(2.0), Inches(5.2), Inches(4.5))
tf_e = tb_e.text_frame
tf_e.word_wrap = True

p_e0 = tf_e.paragraphs[0]
p_e0.text = "MODALITY 2: 12-LEAD ECG 1D-CNN"
p_e0.font.bold = True
p_e0.font.size = Pt(12)
p_e0.font.color.rgb = ACCENT_CYAN
p_e0.space_after = Pt(6)

p_e1 = tf_e.add_paragraph()
p_e1.text = "PTB-XL Diagnostic Database (21,799 Multi-Lead Records)"
p_e1.font.bold = True
p_e1.font.size = Pt(13)
p_e1.font.color.rgb = TEXT_WHITE
p_e1.space_after = Pt(10)

ecg_points = [
    "Input Tensor: (1000, 12) spatial lead matrix capturing simultaneous multi-vector depolarization voltages.",
    "Architecture: 3-stage temporal convolutional blocks (64 → 128 → 256 filters) with batch normalization and Global Average Pooling.",
    "Validation Protocol: Evaluated on stratified holdout test splits against clinical diagnostic superclasses (Normal vs. Arrhythmia/Ischemia).",
    "Empirical Performance: 0.9360 ROC-AUC with 97.85% sensitivity on abnormal conduction."
]
for ep in ecg_points:
    pe = tf_e.add_paragraph()
    pe.text = f"• {ep}"
    pe.font.size = Pt(10.5)
    pe.font.color.rgb = TEXT_MUTED
    pe.space_after = Pt(6)

create_card(s6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
tb_h = s6.shapes.add_textbox(Inches(7.05), Inches(2.0), Inches(5.2), Inches(4.5))
tf_h = tb_h.text_frame
tf_h.word_wrap = True

p_h0 = tf_h.paragraphs[0]
p_h0.text = "MODALITY 3: PHONOCARDIOGRAM (PCG) 2D-CNN"
p_h0.font.bold = True
p_h0.font.size = Pt(12)
p_h0.font.color.rgb = ACCENT_CORAL
p_h0.space_after = Pt(6)

p_h1 = tf_h.add_paragraph()
p_h1.text = "PhysioNet / CinC 2016 Challenge (7,800+ WAV Audio Files)"
p_h1.font.bold = True
p_h1.font.size = Pt(13)
p_h1.font.color.rgb = TEXT_WHITE
p_h1.space_after = Pt(10)

pcg_points = [
    "DSP Audio Pipeline: Resampled to 2,000 Hz, noise-reduced, and transformed into 64×64 Log-Mel Spectrograms.",
    "Architecture: Deep 2D convolutional network (32, 64, 128 kernels) with spatial max-pooling and dense dropout layers.",
    "Diagnostic Target: Identifies valvular regurgitation, structural murmurs, and abnormal S3/S4 acoustic splitting.",
    "Validation Protocol: Verified on official CinC challenge validation splits, achieving 0.8000 ROC-AUC."
]
for hp in pcg_points:
    ph = tf_h.add_paragraph()
    ph.text = f"• {hp}"
    ph.font.size = Pt(10.5)
    ph.font.color.rgb = TEXT_MUTED
    ph.space_after = Pt(6)

# ==========================================================
# SLIDE 7: Modalities 4, 5 & 6 - Optical, SCG & Retinal Ingestion
# ==========================================================
s7 = prs.slides.add_slide(blank_layout)
apply_background(s7)
add_header(s7, "Modalities 4, 5 & 6: Optical, Kinetic & Microvascular Branches")

branches_s7 = [
    ("4. Optical PPG & HRV (Camera)", "Zero-Hardware Pulse Ingestion", [
        "Tracks facial or fingertip capillary blood absorption via standard webcam or smartphone camera.",
        "4th-order zero-phase Butterworth bandpass filter (0.7-3.5 Hz) isolates cardiac pulsation from motion.",
        "Deterministic DSP peak detection computes autonomic metrics: SDNN, RMSSD, and pNN50.",
        "Benchmarked against BIDMC clinical pulse-oximetry gold truth (98.40% Peak F1)."
    ], ACCENT_GREEN),
    ("5. SCG Accelerometer (IMU)", "Kinetic Cardiac Contraction", [
        "TaebiLab-MSCardio benchmark (108 subjects, 996 records).",
        "Tri-axial (X, Y, Z) IMU micro-vibration acceleration timeseries processed via 1D-CNN.",
        "Identifies Aortic Opening (AO) and Mitral Closure (MC) mechanical ejection boundaries.",
        "94.20% accuracy in peak timing identification relative to R-wave reference."
    ], ACCENT_CYAN),
    ("6. Retinal Microvasculature", "Deep U-Net Vessel Segmentation", [
        "Fundus-AVSeg and STARE fundus photographic imaging cohorts.",
        "Green-channel CLAHE contrast enhancement paired with deep 2D U-Net segmentation.",
        "Extracts objective end-organ damage biomarkers: vessel density ratio and vascular tortuosity index.",
        "70.94% Dice coefficient for microvascular vessel boundary extraction."
    ], ACCENT_CORAL)
]

for i, (title, sub, bullets, col) in enumerate(branches_s7):
    left = Inches(0.8 + i * 3.9)
    create_card(s7, left, Inches(1.8), Inches(3.7), Inches(5.0))
    tb = s7.shapes.add_textbox(left + Inches(0.2), Inches(2.0), Inches(3.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p1 = tf.paragraphs[0]
    p1.text = title.upper()
    p1.font.bold = True
    p1.font.size = Pt(11)
    p1.font.color.rgb = col
    p1.space_after = Pt(4)

    p2 = tf.add_paragraph()
    p2.text = sub
    p2.font.bold = True
    p2.font.size = Pt(13)
    p2.font.color.rgb = TEXT_WHITE
    p2.space_after = Pt(10)

    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = f"• {b}"
        pb.font.size = Pt(10)
        pb.font.color.rgb = TEXT_MUTED
        pb.space_after = Pt(6)

# ==========================================================
# SLIDE 8: Late-Fusion Decision Engine & Noise Resilience
# ==========================================================
s8 = prs.slides.add_slide(blank_layout)
apply_background(s8)
add_header(s8, "Late-Fusion Decision Engine & Dynamic Fault Tolerance")

create_card(s8, Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
tb_lf = s8.shapes.add_textbox(Inches(1.05), Inches(2.0), Inches(5.5), Inches(4.5))
tf_lf = tb_lf.text_frame
tf_lf.word_wrap = True

p_f0 = tf_lf.paragraphs[0]
p_f0.text = "DYNAMIC CONFIDENCE-WEIGHTED LATE FUSION"
p_f0.font.bold = True
p_f0.font.size = Pt(12)
p_f0.font.color.rgb = ACCENT_CYAN
p_f0.space_after = Pt(8)

p_f1 = tf_lf.add_paragraph()
p_f1.text = "The Static Dilution Problem:\nTraditional static averaging (e.g., 40% ECG + 40% PCG + 20% Tabular) causes low-confidence or noisy channels to dilute decisive signals, lowering overall system accuracy to ~76.79%."
p_f1.font.size = Pt(10.5)
p_f1.font.color.rgb = TEXT_MUTED
p_f1.space_after = Pt(8)

p_f2 = tf_lf.add_paragraph()
p_f2.text = "Dynamic Confidence Weighting Formulation:\nConfidence is computed from distance to the ambiguous 0.50 boundary:\n  • Confidence_m = 2.0 * |P_m - 0.50|\n  • Weight_m = Confidence_m / Σ (Confidence_k)\n  • Fused_Risk = Σ (Weight_m * P_m)"
p_f2.font.size = Pt(10.5)
p_f2.font.color.rgb = TEXT_WHITE
p_f2.space_after = Pt(8)

p_f3 = tf_lf.add_paragraph()
p_f3.text = "• Decisive channels (P → 1.0 or 0.0) receive dominant influence.\n• Ambiguous or corrupted channels (P ≈ 0.50) are safely down-weighted.\n• Incorporates a 95% empirical confidence interval [P - 6.5%, P + 6.5%]."
p_f3.font.size = Pt(10.5)
p_f3.font.color.rgb = TEXT_MUTED

create_card(s8, Inches(7.1), Inches(1.8), Inches(5.4), Inches(5.0))
tb_res = s8.shapes.add_textbox(Inches(7.35), Inches(2.0), Inches(4.9), Inches(4.5))
tf_res = tb_res.text_frame
tf_res.word_wrap = True

p_r0 = tf_res.paragraphs[0]
p_r0.text = "FAULT TOLERANCE STRESS TEST: ECG NOISE INJECTION"
p_r0.font.bold = True
p_r0.font.size = Pt(12)
p_r0.font.color.rgb = ACCENT_CORAL
p_r0.space_after = Pt(10)

p_r1 = tf_res.add_paragraph()
p_r1.text = "Simulated Hardware Failure (Loose Leads / Motion Artifacts):\nArtificial Gaussian noise was injected into the active ECG channel, driving its diagnostic probability to maximum uncertainty (P ≈ 0.51)."
p_r1.font.size = Pt(10.5)
p_r1.font.color.rgb = TEXT_MUTED
p_r1.space_after = Pt(10)

p_r2 = tf_res.add_paragraph()
p_r2.text = "Automated System Weight Re-Distribution:\n  • ECG Weight: 41.25% ➔ 3.13% (Automated ↓38.12% drop)\n  • Heart Sound Weight: 30.00% ➔ 60.00% (Compensated ↑30.00%)\n  • Tabular Weight: 28.75% ➔ 36.87% (Compensated ↑8.12%)"
p_r2.font.bold = True
p_r2.font.size = Pt(11)
p_r2.font.color.rgb = TEXT_WHITE
p_r2.space_after = Pt(10)

p_r3 = tf_res.add_paragraph()
p_r3.text = "Clinical Outcome: The system maintained an accuracy of 81.55%, proving multi-sensor resilience against individual sensor corruption."
p_r3.font.size = Pt(10.5)
p_r3.font.color.rgb = ACCENT_GREEN

# ==========================================================
# SLIDE 9: Explainable AI (XAI) & Counterfactual Roadmaps
# ==========================================================
s9 = prs.slides.add_slide(blank_layout)
apply_background(s9)
add_header(s9, "Explainable AI (TreeSHAP) & Prescriptive Counterfactual Actions")

create_card(s9, Inches(0.8), Inches(1.8), Inches(5.7), Inches(5.0))
tb_x1 = s9.shapes.add_textbox(Inches(1.05), Inches(2.0), Inches(5.2), Inches(4.5))
tf_x1 = tb_x1.text_frame
tf_x1.word_wrap = True

p_x0 = tf_x1.paragraphs[0]
p_x0.text = "TREESHAP ATTRIBUTION ENGINE"
p_x0.font.bold = True
p_x0.font.size = Pt(12)
p_x0.font.color.rgb = ACCENT_CYAN
p_x0.space_after = Pt(8)

p_x1 = tf_x1.add_paragraph()
p_x1.text = "Exact Shapley Value Decompositions:\nEliminates 'black-box' opacity by computing exact local feature attributions across power sets of clinical covariates:"
p_x1.font.size = Pt(10.5)
p_x1.font.color.rgb = TEXT_MUTED
p_x1.space_after = Pt(10)

shap_table = [
    ("Systolic Blood Pressure", "+0.152", "Major Risk Increase (Hypertension)"),
    ("Patient Age (Years)", "+0.124", "Non-modifiable demographic risk"),
    ("Total Serum Cholesterol", "+0.081", "Elevated lipid biomarker"),
    ("Active Cigarette Smoking", "+0.062", "Behavioral endothelial damage"),
    ("Body Mass Index (BMI)", "+0.043", "Adiposity risk factor"),
    ("Physical Exercise", "-0.038", "Cardioprotective reduction")
]
for feat, val, comm in shap_table:
    px = tf_x1.add_paragraph()
    px.text = f"• {feat}: {val} ({comm})"
    px.font.size = Pt(10)
    px.font.color.rgb = TEXT_WHITE
    px.space_after = Pt(4)

create_card(s9, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
tb_x2 = s9.shapes.add_textbox(Inches(7.05), Inches(2.0), Inches(5.2), Inches(4.5))
tf_x2 = tb_x2.text_frame
tf_x2.word_wrap = True

p_cf0 = tf_x2.paragraphs[0]
p_cf0.text = "ACTIONABLE COUNTERFACTUAL ROADMAP"
p_cf0.font.bold = True
p_cf0.font.size = Pt(12)
p_cf0.font.color.rgb = ACCENT_GREEN
p_cf0.space_after = Pt(8)

p_cf1 = tf_x2.add_paragraph()
p_cf1.text = "Prescriptive Lifestyle Interventions:\nRather than only outputting a static risk score, CardioRisk AI computes the shortest mathematical path to de-escalate patient risk from High (85%) to Moderate (65%):"
p_cf1.font.size = Pt(10.5)
p_cf1.font.color.rgb = TEXT_MUTED
p_cf1.space_after = Pt(10)

cf_actions = [
    ("1. Smoking Cessation", "Active Smoker ➔ Non-Smoker", "-8.5% Absolute Risk"),
    ("2. SBP Optimization", "142 mmHg ➔ 120 mmHg (Target)", "-5.1% Absolute Risk"),
    ("3. Aerobic Exercise", "Sedentary ➔ 150 min/week", "-6.2% Absolute Risk"),
    ("4. BMI Normalization", "28.5 ➔ 24.0 kg/m²", "-3.4% Absolute Risk"),
    ("5. Lipid Management", "Above Normal ➔ Normal", "-2.8% Absolute Risk")
]
for act, detail, red in cf_actions:
    pcf = tf_x2.add_paragraph()
    pcf.text = f"✅ {act} ({detail}): {red}"
    pcf.font.bold = True
    pcf.font.size = Pt(10.5)
    pcf.font.color.rgb = TEXT_WHITE
    pcf.space_after = Pt(6)

# ==========================================================
# SLIDE 10: Empirical Implementation Results Table
# ==========================================================
s10 = prs.slides.add_slide(blank_layout)
apply_background(s10)
add_header(s10, "Empirical Implementation Results & Comparative Benchmarks")

table_shape = s10.shapes.add_table(7, 5, Inches(0.8), Inches(1.8), Inches(11.73), Inches(4.8))
table = table_shape.table

headers = ["Diagnostic Modality", "Model Architecture", "Training Cohort", "Primary Performance Metric", "Validation Protocol"]
for col_idx, text in enumerate(headers):
    cell = table.cell(0, col_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = CARD_BG
    p = cell.text_frame.paragraphs[0]
    p.text = text
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = ACCENT_CYAN

results_data = [
    ("1. Clinical Tabular Risk", "Cost-Sensitive XGBoost", "151,860 Patients", "ROC-AUC: 0.8139 • Recall: 71.0%", "15% Holdout (Youden's J = 0.4769)"),
    ("2. 12-Lead ECG", "1D-CNN Multi-Lead", "21,799 PTB-XL Records", "ROC-AUC: 0.9360 • Acc: 81.0%", "Stratified Diagnostic Superclass Split"),
    ("3. Heart Sound (PCG)", "2D-CNN Mel-Spectrogram", "3,126 CinC Recordings", "ROC-AUC: 0.8000 • Acc: 75.3%", "Official CinC Challenge Split"),
    ("★ Dual Ensemble (ECG + HS)", "Dynamic Confidence Late Fusion", "Combined Multi-Modal", "ROC-AUC: 0.9420 • Acc: 86.0%", "Compensates for tabular signal dilution"),
    ("4. Camera rPPG / HRV", "Butterworth Bandpass DSP", "PPG_DATASET / BIDMC", "Peak Detection F1: 98.40%", "Validated against reference pulse-oximetry"),
    ("5. SCG Accelerometer", "1D-CNN Kinematic Ingest", "TaebiLab-MSCardio (108 Subj)", "AO Detection Acc: 94.20%", "Tri-Axial Kinetic Mechanical Windowing")
]

for row_idx, row in enumerate(results_data, start=1):
    for col_idx, val in enumerate(row):
        cell = table.cell(row_idx, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = BG_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.size = Pt(10)
        p.font.color.rgb = ACCENT_GREEN if "★" in row[0] else (TEXT_WHITE if col_idx == 0 else TEXT_MUTED)

# ==========================================================
# SLIDE 11: Roadmap for CSE 499B & Engineering Milestones
# ==========================================================
s11 = prs.slides.add_slide(blank_layout)
apply_background(s11)
add_header(s11, "Future Work: Comprehensive Roadmap for CSE 499B")

rw_cards = [
    ("1. Physical IoT Diagnostic Hardware", "Low-Cost Wearable (~$95 BOM)", [
        "ESP32 dual-core microcontroller with Wi-Fi/BLE ($15).",
        "AD8232 single-lead ECG front-end analog sensor ($20).",
        "MAX30102 optical pulse-oximetry and heart-rate sensor ($15).",
        "MAX9814 electret stethoscope microphone ($10).",
        "ADXL345 3-axis digital accelerometer for chest SCG ($10).",
        "0.96-inch I2C OLED display and 1200mAh LiPo battery ($25)."
    ], ACCENT_CYAN),
    ("2. Firmware & Microsecond Time Sync", "FreeRTOS Real-Time Scheduler", [
        "Deterministic FreeRTOS task scheduling on ESP32 cores.",
        "Hardware timer interrupts for synchronized signal sampling.",
        "Microsecond-accurate time alignment across ECG R-peaks, PCG S1/S2 audio, and optical pulse arrival times.",
        "Enables non-invasive Pulse Transit Time (PTT) continuous BP."
    ], ACCENT_GREEN),
    ("3. Neural Stacking & Prospective Trials", "Advanced Meta-Learner & Clinical Pilot", [
        "Transition heuristic confidence weighting to a trained neural stacking network.",
        "Conduct IRB-compliant multi-subject clinical pilot studies.",
        "Prospective validation against hospital electronic health records.",
        "On-device quantization (TensorFlow Lite) for edge execution."
    ], ACCENT_CORAL)
]

for i, (title, sub, bullets, col) in enumerate(rw_cards):
    left = Inches(0.8 + i * 3.9)
    create_card(s11, left, Inches(1.8), Inches(3.7), Inches(5.0))
    tb = s11.shapes.add_textbox(left + Inches(0.2), Inches(2.0), Inches(3.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p1 = tf.paragraphs[0]
    p1.text = title.upper()
    p1.font.bold = True
    p1.font.size = Pt(11)
    p1.font.color.rgb = col
    p1.space_after = Pt(4)

    p2 = tf.add_paragraph()
    p2.text = sub
    p2.font.bold = True
    p2.font.size = Pt(13)
    p2.font.color.rgb = TEXT_WHITE
    p2.space_after = Pt(10)

    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = f"• {b}"
        pb.font.size = Pt(10)
        pb.font.color.rgb = TEXT_MUTED
        pb.space_after = Pt(5)

# ==========================================================
# SLIDE 12: Defense Conclusion & Question Session
# ==========================================================
s12 = prs.slides.add_slide(blank_layout)
apply_background(s12)
add_header(s12, "Summary of Contributions & Committee Q&A Session")

create_card(s12, Inches(0.8), Inches(1.8), Inches(11.73), Inches(3.2))
tb_sum = s12.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(11.1), Inches(2.8))
tf_sum = tb_sum.text_frame
tf_sum.word_wrap = True

p_st = tf_sum.paragraphs[0]
p_st.text = "RESEARCH & ENGINEERING DEFENSE SUMMARY"
p_st.font.bold = True
p_st.font.size = Pt(13)
p_st.font.color.rgb = ACCENT_CYAN
p_st.space_after = Pt(8)

sum_points = [
    "✅ Full-Stack Multimodal Platform: Engineered an enterprise 4-tier decoupled platform integrating 6 heterogeneous biological modalities.",
    "✅ Scaled 151k Clinical Cohort: Eliminated single-hospital sample bias and pushed abnormal sensitivity to 71.0% with Youden's J calibration.",
    "✅ Dual Ensemble Superiority: Achieved 86.00% accuracy and 0.9420 ROC-AUC, resolving tabular signal dilution.",
    "✅ Dynamic Fault Tolerance: Validated 38.12% automated noise suppression under severe single-channel sensor failure conditions.",
    "✅ Clinical Interpretability: Delivered exact TreeSHAP attributions, prescriptive counterfactuals, and HL7/FHIR R4 medical exports."
]
for sp in sum_points:
    psp = tf_sum.add_paragraph()
    psp.text = sp
    psp.font.size = Pt(11)
    psp.font.color.rgb = TEXT_WHITE
    psp.space_after = Pt(4)

create_card(s12, Inches(0.8), Inches(5.2), Inches(11.73), Inches(1.6))
tb_cls = s12.shapes.add_textbox(Inches(1.1), Inches(5.35), Inches(11.1), Inches(1.3))
tf_cls = tb_cls.text_frame
tf_cls.word_wrap = True

p_c0 = tf_cls.paragraphs[0]
p_c0.text = "DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING • NORTH SOUTH UNIVERSITY"
p_c0.font.bold = True
p_c0.font.size = Pt(11)
p_c0.font.color.rgb = ACCENT_GREEN

p_c1 = tf_cls.add_paragraph()
p_c1.text = "Candidate: Hussain Mahabub (ID: 2111015642)  |  Supervisor: Shahnewaz Siddique  |  CSE 499A - Section 10"
p_c1.font.bold = True
p_c1.font.size = Pt(13)
p_c1.font.color.rgb = TEXT_WHITE

p_c2 = tf_cls.add_paragraph()
p_c2.text = "Thank you for your consideration. The floor is now open for questions and technical discussion."
p_c2.font.size = Pt(11)
p_c2.font.color.rgb = ACCENT_CYAN

# Export Presentation
out_file = "CardioRisk_AI_CSE499A_Presentation_Enhanced.pptx"
prs.save(out_file)
print(f"============================================================")
print(f"✅ SUCCESS: Presentation generated successfully!")
print(f"📁 File saved as: {out_file}")
print(f"============================================================")