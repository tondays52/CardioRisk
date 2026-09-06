import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# Theme Colors
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

def add_header(slide, title_text, category="NORTH SOUTH UNIVERSITY • CSE 499A CAPSTONE DEFENSE"):
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(0.3))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.name = "Arial"
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_CYAN

    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.75))
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

def try_add_image(slide, img_name, left, top, width, height=None):
    """Safely adds image if it exists in current or desktop directory."""
    paths_to_check = [
        img_name,
        os.path.join(".", img_name),
        os.path.join(os.path.expanduser("~"), "Desktop", img_name),
        os.path.join(os.path.expanduser("~"), "Desktop", "presentations", img_name),
        os.path.join(os.path.expanduser("~"), "Desktop", "hrp", img_name)
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            if height:
                return slide.shapes.add_picture(p, left, top, width, height)
            return slide.shapes.add_picture(p, left, top, width)
    # If image not found, draw placeholder
    box = create_card(slide, left, top, width, height or Inches(3.5))
    tb = slide.shapes.add_textbox(left, top + Inches(0.5), width, Inches(1.5))
    p = tb.text_frame.paragraphs[0]
    p.text = f"[Image: {img_name}]"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MUTED
    p.alignment = PP_ALIGN.CENTER
    return box

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
# SLIDE 2: Clinical Motivation & Problem Statement
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
    create_card(s2, left, Inches(1.6), Inches(3.7), Inches(5.3))
    tb = s2.shapes.add_textbox(left + Inches(0.25), Inches(1.8), Inches(3.2), Inches(4.8))
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
# SLIDE 3: Clinical Command Center Dashboard (IMAGE 1)
# ==========================================================
s3 = prs.slides.add_slide(blank_layout)
apply_background(s3)
add_header(s3, "Clinical Command Center & Real-Time Telemetry Interface")

create_card(s3, Inches(0.8), Inches(1.5), Inches(4.6), Inches(5.4))
tb_c3 = s3.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(4.2), Inches(5.0))
tf_c3 = tb_c3.text_frame
tf_c3.word_wrap = True

p = tf_c3.paragraphs[0]
p.text = "EXECUTIVE TRIAGE SURVEILLANCE"
p.font.bold = True
p.font.size = Pt(12)
p.font.color.rgb = ACCENT_CYAN
p.space_after = Pt(10)

c3_points = [
    ("Real-Time KPI Monitoring", "Aggregates active patients, high-risk triage cases, and benchmark metrics at a glance."),
    ("Standalone Telemetry Tools", "Provides zero-hardware web access to Camera PPG, Heart Sound PCG, and IMU Accelerometer streams."),
    ("Asynchronous Inference Engine", "Active status indicator confirming live communication with 6 trained deep neural branches on port 8000."),
    ("Multi-Language Support", "Instant bilingual localization between English and Bangla for decentralized rural deployment.")
]
for h, d in c3_points:
    ph = tf_c3.add_paragraph()
    ph.text = f"• {h}: "
    ph.font.bold = True
    ph.font.size = Pt(11)
    ph.font.color.rgb = TEXT_WHITE
    pd = tf_c3.add_paragraph()
    pd.text = f"  {d}"
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_MUTED
    pd.space_after = Pt(6)

# Right: Screenshot_3-9-2026_113638_localhost.jpeg
try_add_image(s3, "Screenshot_3-9-2026_113638_localhost.jpeg", Inches(5.6), Inches(1.5), Inches(6.9), Inches(5.4))

# ==========================================================
# SLIDE 4: Multimodal Ingestion Pipeline (IMAGE 2 & 3)
# ==========================================================
s4 = prs.slides.add_slide(blank_layout)
apply_background(s4)
add_header(s4, "Multi-Modal Intake & Sensor Ingestion Pipeline")

create_card(s4, Inches(0.8), Inches(1.5), Inches(4.8), Inches(5.4))
tb_s4 = s4.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(4.4), Inches(5.0))
tf_s4 = tb_s4.text_frame
tf_s4.word_wrap = True

p = tf_s4.paragraphs[0]
p.text = "HETEROGENEOUS SIGNAL INGESTION"
p.font.bold = True
p.font.size = Pt(12)
p.font.color.rgb = ACCENT_GREEN
p.space_after = Pt(10)

s4_bullets = [
    ("Modular Architecture", "Patients can be evaluated with any subset of available inputs; absent modalities default to neutral prior probabilities without breaking the pipeline."),
    ("Zero-Hardware Sensors", "Incorporates camera rPPG, microphone stethoscope auscultation, and smartphone IMU seismocardiography."),
    ("EHR Batch Ingestion", "Includes automated bulk CSV cohort uploader for rapid screening of hospital clinics."),
    ("Multimodal Framework", "Harmonizes electrical, acoustic, optical, and kinetic cardiac physics into one late-fusion pipeline.")
]
for h, d in s4_bullets:
    ph = tf_s4.add_paragraph()
    ph.text = f"• {h}: "
    ph.font.bold = True
    ph.font.size = Pt(11)
    ph.font.color.rgb = TEXT_WHITE
    pd = tf_s4.add_paragraph()
    pd.text = f"  {d}"
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_MUTED
    pd.space_after = Pt(6)

# Right: Stacked images (Screenshot_3-9-2026_113714 & 113731)
try_add_image(s4, "Screenshot_3-9-2026_113714_localhost.jpeg", Inches(5.8), Inches(1.5), Inches(6.7), Inches(2.6))
try_add_image(s4, "Screenshot_3-9-2026_113731_localhost.jpeg", Inches(5.8), Inches(4.25), Inches(6.7), Inches(2.65))

# ==========================================================
# SLIDE 5: Modality 1 - Scaled Tabular Classifier (151k)
# ==========================================================
s5 = prs.slides.add_slide(blank_layout)
apply_background(s5)
add_header(s5, "Modality 1: Tabular Classifier Scaled across 151,860 Records")

create_card(s5, Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.4))
tb_t1 = s5.shapes.add_textbox(Inches(1.05), Inches(1.7), Inches(6.3), Inches(5.0))
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
    top = Inches(1.5 + i * 1.35)
    create_card(s5, Inches(7.9), top, Inches(4.6), Inches(1.2))
    tb_m = s5.shapes.add_textbox(Inches(8.1), top + Inches(0.15), Inches(4.2), Inches(0.9))
    tf_m = tb_m.text_frame
    pv = tf_m.paragraphs[0]
    pv.text = m_val
    pv.font.bold = True
    pv.font.size = Pt(26)
    pv.font.color.rgb = col
    pl = tf_m.add_paragraph()
    pl.text = m_lbl
    pl.font.size = Pt(11)
    pl.font.color.rgb = TEXT_MUTED

# ==========================================================
# SLIDE 6: Waveform Annotation Studio (IMAGE 4)
# ==========================================================
s6 = prs.slides.add_slide(blank_layout)
apply_background(s6)
add_header(s6, "Physiological Waveform Morphology Sandbox (Lead II Demo)")

create_card(s6, Inches(0.8), Inches(1.5), Inches(4.6), Inches(5.4))
tb_s6 = s6.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(4.2), Inches(5.0))
tf_s6 = tb_s6.text_frame
tf_s6.word_wrap = True

p = tf_s6.paragraphs[0]
p.text = "WAVEFORM ANNOTATION & MORPHOLOGY"
p.font.bold = True
p.font.size = Pt(12)
p.font.color.rgb = ACCENT_CORAL
p.space_after = Pt(10)

s6_bullets = [
    ("Interactive Bio-Signal Sandbox", "Allows clinicians to adjust cardiac rhythm (BPM), voltage gain, noise level, and lead configuration in real time."),
    ("Fiducial Point Detection", "Simulates canonical P-Q-R-S-T complexes with automated diamond marker tracking on ventricular R-peaks."),
    ("Clinical Calibration", "Operates under standard 25 mm/s paper speed equivalent and 10 mm/mV voltage calibration."),
    ("Educational & Verification Tool", "Demonstrates the biophysical foundations of 1D-CNN temporal feature extraction.")
]
for h, d in s6_bullets:
    ph = tf_s6.add_paragraph()
    ph.text = f"• {h}: "
    ph.font.bold = True
    ph.font.size = Pt(11)
    ph.font.color.rgb = TEXT_WHITE
    pd = tf_s6.add_paragraph()
    pd.text = f"  {d}"
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_MUTED
    pd.space_after = Pt(6)

# Right: Screenshot_3-9-2026_113834_localhost.jpeg
try_add_image(s6, "Screenshot_3-9-2026_113834_localhost.jpeg", Inches(5.6), Inches(1.5), Inches(6.9), Inches(5.4))

# ==========================================================
# SLIDE 7: Unified Results & Risk Stratification (IMAGE 5)
# ==========================================================
s7 = prs.slides.add_slide(blank_layout)
apply_background(s7)
add_header(s7, "Multi-Modal Fused Risk Triage & Uncertainty Bounds")

create_card(s7, Inches(0.8), Inches(1.5), Inches(4.6), Inches(5.4))
tb_s7 = s7.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(4.2), Inches(5.0))
tf_s7 = tb_s7.text_frame
tf_s7.word_wrap = True

p = tf_s7.paragraphs[0]
p.text = "LATE-FUSION DIAGNOSTIC OUTPUTS"
p.font.bold = True
p.font.size = Pt(12)
p.font.color.rgb = ACCENT_CORAL
p.space_after = Pt(10)

s7_bullets = [
    ("Unified Risk Stratification", "Stratifies patients into actionable tiers: Low (<30%), Moderate (30-60%), and High (>=60%) risk."),
    ("95% Empirical Confidence Intervals", "Every prediction is bound by confidence bounds ([75.9% - 88.9%] on high-risk case shown) to prevent diagnostic overconfidence."),
    ("Dynamic Confidence Weights", "Visualizes the percentage contribution of each active modality to the final decision."),
    ("Clinical Triage Speed", "Full multi-modal evaluation completes in under 350 ms on standard CPU hardware.")
]
for h, d in s7_bullets:
    ph = tf_s7.add_paragraph()
    ph.text = f"• {h}: "
    ph.font.bold = True
    ph.font.size = Pt(11)
    ph.font.color.rgb = TEXT_WHITE
    pd = tf_s7.add_paragraph()
    pd.text = f"  {d}"
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_MUTED
    pd.space_after = Pt(6)

# Right: Screenshot_3-9-2026_113747_localhost.jpeg
try_add_image(s7, "Screenshot_3-9-2026_113747_localhost.jpeg", Inches(5.6), Inches(1.5), Inches(6.9), Inches(5.4))

# ==========================================================
# SLIDE 8: Explainable AI with TreeSHAP (IMAGE 6)
# ==========================================================
s8 = prs.slides.add_slide(blank_layout)
apply_background(s8)
add_header(s8, "Explainable AI (TreeSHAP Local Feature Attributions)")

create_card(s8, Inches(0.8), Inches(1.5), Inches(4.6), Inches(5.4))
tb_s8 = s8.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(4.2), Inches(5.0))
tf_s8 = tb_s8.text_frame
tf_s8.word_wrap = True

p = tf_s8.paragraphs[0]
p.text = "TRANSPARENT CLINICAL REASONING"
p.font.bold = True
p.font.size = Pt(12)
p.font.color.rgb = ACCENT_CYAN
p.space_after = Pt(10)

s8_bullets = [
    ("Exact Shapley Attribution", "Decomposes risk into additive positive and negative feature contributions relative to population baseline."),
    ("Primary Risk Accelerators", "On patient shown, Systolic BP (ap_hi) contributed +1.352 to log-odds risk, followed by Diastolic BP (+0.174) and Age (+0.042)."),
    ("Protective Offsets", "Total serum cholesterol (-0.086) and physical activity (-0.025) provided negative risk offsets."),
    ("Eliminating Black-Box Skepticism", "Directly explains to clinicians WHY a patient was triaged into the High Risk category.")
]
for h, d in s8_bullets:
    ph = tf_s8.add_paragraph()
    ph.text = f"• {h}: "
    ph.font.bold = True
    ph.font.size = Pt(11)
    ph.font.color.rgb = TEXT_WHITE
    pd = tf_s8.add_paragraph()
    pd.text = f"  {d}"
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_MUTED
    pd.space_after = Pt(6)

# Right: Screenshot_3-9-2026_11384_localhost.jpeg
try_add_image(s8, "Screenshot_3-9-2026_11384_localhost.jpeg", Inches(5.6), Inches(1.5), Inches(6.9), Inches(5.4))

# ==========================================================
# SLIDE 9: What-If Simulator & Counterfactuals (IMAGE 7)
# ==========================================================
s9 = prs.slides.add_slide(blank_layout)
apply_background(s9)
add_header(s9, "Interactive What-If Simulator & Counterfactual Interventions")

create_card(s9, Inches(0.8), Inches(1.5), Inches(4.6), Inches(5.4))
tb_s9 = s9.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(4.2), Inches(5.0))
tf_s9 = tb_s9.text_frame
tf_s9.word_wrap = True

p = tf_s9.paragraphs[0]
p.text = "ACTIONABLE CLINICAL PRESCRIPTIONS"
p.font.bold = True
p.font.size = Pt(12)
p.font.color.rgb = ACCENT_GREEN
p.space_after = Pt(10)

s9_bullets = [
    ("Moving Beyond Passive Prediction", "Provides patients and clinicians with specific, quantified lifestyle targets rather than a static alarm."),
    ("Ranked Action Roadmap", "Prioritizes high-yield interventions: SBP reduction (-5.1% Risk) and BMI normalization (-3.4% Risk)."),
    ("Interactive Simulation", "Clinicians can slide target parameters in real time to visualize projected risk reduction before prescribing treatments."),
    ("Seamless EHR Export", "Single-click export to official PDF Clinical Dossiers and interoperable HL7/FHIR R4 JSON bundles.")
]
for h, d in s9_bullets:
    ph = tf_s9.add_paragraph()
    ph.text = f"• {h}: "
    ph.font.bold = True
    ph.font.size = Pt(11)
    ph.font.color.rgb = TEXT_WHITE
    pd = tf_s9.add_paragraph()
    pd.text = f"  {d}"
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_MUTED
    pd.space_after = Pt(6)

# Right: Screenshot_3-9-2026_113816_localhost.jpeg
try_add_image(s9, "Screenshot_3-9-2026_113816_localhost.jpeg", Inches(5.6), Inches(1.5), Inches(6.9), Inches(5.4))

# ==========================================================
# SLIDE 10: Cohort Registry & Longitudinal Tracking (IMAGE 8 & 9)
# ==========================================================
s10 = prs.slides.add_slide(blank_layout)
apply_background(s10)
add_header(s10, "Patient Cohort Registry & Longitudinal Multi-Visit Tracking")

create_card(s10, Inches(0.8), Inches(1.5), Inches(4.8), Inches(5.4))
tb_s10 = s10.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(4.4), Inches(5.0))
tf_s10 = tb_s10.text_frame
tf_s10.word_wrap = True

p = tf_s10.paragraphs[0]
p.text = "POPULATION SURVEILLANCE & TRAJECTORIES"
p.font.bold = True
p.font.size = Pt(12)
p.font.color.rgb = ACCENT_CYAN
p.space_after = Pt(10)

s10_bullets = [
    ("Cohort Risk Breakdown", "Automated distribution analytics (e.g. 50% Moderate, 33.3% Low, 16.7% High Risk in monitored cohort)."),
    ("Age vs. Risk Scatter Plot", "Visualizes population risk progression with color-coded risk boundaries and polynomial cohort trajectories."),
    ("Multi-Visit Longitudinal Tracking", "Tracks individual patient recovery over time (e.g. Patient #1 dropped from 18.8% to 7.5% risk following BP reduction)."),
    ("Recalibration Engine", "Includes automated database-wide risk category synchronization upon guideline revisions.")
]
for h, d in s10_bullets:
    ph = tf_s10.add_paragraph()
    ph.text = f"• {h}: "
    ph.font.bold = True
    ph.font.size = Pt(11)
    ph.font.color.rgb = TEXT_WHITE
    pd = tf_s10.add_paragraph()
    pd.text = f"  {d}"
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_MUTED
    pd.space_after = Pt(6)

# Right: Screenshot_3-9-2026_113854 & 113920
try_add_image(s10, "Screenshot_3-9-2026_113854_localhost.jpeg", Inches(5.8), Inches(1.5), Inches(6.7), Inches(2.6))
try_add_image(s10, "Screenshot_3-9-2026_113920_localhost.jpeg", Inches(5.8), Inches(4.25), Inches(6.7), Inches(2.65))

# ==========================================================
# SLIDE 11: Empirical Model Benchmarks (IMAGE 10)
# ==========================================================
s11 = prs.slides.add_slide(blank_layout)
apply_background(s11)
add_header(s11, "Empirical Benchmarks & Biophysical Feature Validation")

create_card(s11, Inches(0.8), Inches(1.5), Inches(4.6), Inches(5.4))
tb_s11 = s11.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(4.2), Inches(5.0))
tf_s11 = tb_s11.text_frame
tf_s11.word_wrap = True

p = tf_s11.paragraphs[0]
p.text = "GOLD-STANDARD CLINICAL BENCHMARKS"
p.font.bold = True
p.font.size = Pt(12)
p.font.color.rgb = ACCENT_CYAN
p.space_after = Pt(10)

s11_bullets = [
    ("Diagnostic AI Classifiers", "Trained on verified clinical cohorts: PTB-XL (ECG 0.9360 AUC), CinC 2016 (PCG 0.8000 AUC), and Scaled Multi-Cohort (Tabular 0.8139 AUC)."),
    ("Biophysical DSP Engines", "PPG chrominance pipeline achieves 98.40% peak detection F1 with <1.8 BPM MAE against clinical pulse oximetry."),
    ("Kinetic Accelerometer Engine", "TaebiLab-MSCardio SCG achieves 94.20% Aortic Opening (AO) detection accuracy with <12 ms timing error."),
    ("Retinal U-Net Segmentation", "Fundus-AVSeg vessel model delivers 70.94% Dice coefficient with 99.17% background specificity.")
]
for h, d in s11_bullets:
    ph = tf_s11.add_paragraph()
    ph.text = f"• {h}: "
    ph.font.bold = True
    ph.font.size = Pt(11)
    ph.font.color.rgb = TEXT_WHITE
    pd = tf_s11.add_paragraph()
    pd.text = f"  {d}"
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_MUTED
    pd.space_after = Pt(6)

# Right: Screenshot_3-9-2026_113941_localhost.jpeg
try_add_image(s11, "Screenshot_3-9-2026_113941_localhost.jpeg", Inches(5.6), Inches(1.5), Inches(6.9), Inches(5.4))

# ==========================================================
# SLIDE 12: Roadmap for CSE 499B & Hardware BOM
# ==========================================================
s12 = prs.slides.add_slide(blank_layout)
apply_background(s12)
add_header(s12, "Future Work: Comprehensive Roadmap for CSE 499B")

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
    create_card(s12, left, Inches(1.6), Inches(3.7), Inches(5.3))
    tb = s12.shapes.add_textbox(left + Inches(0.2), Inches(1.8), Inches(3.3), Inches(4.8))
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
# SLIDE 13: Conclusion & Defense Summary
# ==========================================================
s13 = prs.slides.add_slide(blank_layout)
apply_background(s13)
add_header(s13, "Summary of Contributions & Committee Q&A Session")

create_card(s13, Inches(0.8), Inches(1.6), Inches(11.73), Inches(3.4))
tb_sum = s13.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(3.0))
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
    psp.space_after = Pt(6)

create_card(s13, Inches(0.8), Inches(5.2), Inches(11.73), Inches(1.7))
tb_cls = s13.shapes.add_textbox(Inches(1.1), Inches(5.35), Inches(11.1), Inches(1.4))
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

out_file = "CardioRisk_AI_CSE499A_Presentation_With_Screenshots.pptx"
prs.save(out_file)
print("="*60)
print(f"✅ SUCCESS: Presentation with screenshots generated successfully!")
print(f"📁 File saved as: {out_file}")
print("="*60)