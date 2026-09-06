\# CardioRisk AI: A Multimodal Deep Learning System for Smartphone-Based Cardiovascular Risk Assessment



\*\*Author:\*\* \[Your Name]

\*\*Course:\*\* CSE 499A - Capstone Project

\*\*Institution:\*\* \[Your University]

\*\*Date:\*\* September 2026



\---



\## Abstract



CardioRisk AI is a smartphone-based multimodal deep learning system that predicts an individual's 10-year risk of heart attack by fusing five physiological modalities: clinical tabular data, photoplethysmography (PPG) derived heart rate variability (HRV), electrocardiography (ECG), heart sounds (phonocardiography), and seismocardiography (SCG). The system processes each modality through dedicated signal processing pipelines and deep learning models, then fuses their outputs using weighted averaging with uncertainty quantification. A Streamlit-based dashboard provides risk visualization, SHAP-based feature importance, counterfactual recommendations, a what-if simulator, and PDF report generation. Preliminary results on the Kaggle Cardiovascular Disease dataset show a tabular model AUC of 0.798, with multimodal fusion providing more balanced risk estimates through confidence intervals.



\---



\## 1. Introduction and Problem Statement



\### 1.1 Background



Cardiovascular diseases (CVDs) are the leading cause of death globally, accounting for approximately 17.9 million deaths annually. Heart attacks (myocardial infarctions) often occur without warning, and many individuals remain unaware of their elevated risk until a critical event occurs. Traditional risk assessment tools such as the Framingham Risk Score and ASCVD estimator rely on static clinical variables and require laboratory tests, limiting their accessibility in low-resource settings.



\### 1.2 Problem Statement



How can we build an accessible, accurate, and interpretable system for early prediction of heart attack risk by leveraging smartphone sensors and deep learning?



\### 1.3 Objectives



\- Develop signal processing pipelines for five physiological modalities

\- Train machine learning models for each modality

\- Implement multimodal fusion with uncertainty estimation

\- Build an explainable AI dashboard with actionable recommendations



\---



\## 2. Background and Related Works



\### 2.1 Clinical Risk Scores

\- Framingham Risk Score

\- ASCVD Estimator

\- WHO/ISH Charts



\### 2.2 Machine Learning for CVD Prediction

\- XGBoost, Random Forest, Neural Networks on tabular data



\### 2.3 Deep Learning on Physiological Signals

\- ECG CNN architectures

\- Heart sound classification

\- PPG/HRV analysis



\### 2.4 Multimodal Fusion

\- Early fusion, late fusion, learned fusion



\### 2.5 Explainable AI

\- SHAP, Grad-CAM, Counterfactual explanations



\---



\## 3. Proposed Solution



\### 3.1 System Architecture



\[Describe the 5-layer architecture]



\### 3.2 Data Acquisition Module



\[Describe each input channel]



\### 3.3 Signal Processing Pipelines



\[Describe PPG, ECG, heart sound, SCG processing]



\### 3.4 Deep Learning Models



\[Describe each model architecture]



\### 3.5 Multimodal Fusion



\[Describe weighted averaging + CI]



\### 3.6 Explainability Module



\[Describe SHAP, counterfactuals, what-if simulator]



\### 3.7 Web Application



\[Describe Streamlit dashboard pages]



\---



\## 4. Preliminary Results



\### 4.1 Tabular Model Performance



| Model | AUC |

|---|---|

| Logistic Regression | 0.786 |

| Random Forest | 0.798 |

| XGBoost | 0.797 |

| MLP | 0.790 |



\### 4.2 Multimodal Fusion Results



\[Include example fusion output]



\### 4.3 PPG/HRV Feasibility



\[Describe synthetic test results]



\---



\## 5. What Will Be Completed Next in 499B



\- Retinal imaging module

\- Longitudinal risk tracking

\- Federated learning

\- Causal treatment effect estimation

\- Real-time camera integration

\- Deployment to cloud



\---



\## 6. References



\[Include all references in IEEE format]



\---



\*End of Report\*

