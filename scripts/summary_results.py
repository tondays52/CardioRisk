"""
CardioRisk AI - Final Model Summary
Generates a summary table of all model results.
"""
import pandas as pd

def create_summary():
    """Create and display the final model comparison table."""
    
    results = {
        'Model': ['Heart Sound CNN', 'ECG CNN', 'Ensemble (ECG+HS)'],
        'Dataset': ['CinC 2016', 'PTB-XL', 'Combined'],
        'Accuracy': ['75.31%', '81.00%', '86.00%'],
        'AUC': ['0.800', '0.936', '0.942'],
        'Normal Recall': ['~75%', '66.36%', '74.77%'],
        'Abnormal Recall': ['~75%', '97.85%', '98.92%']
    }
    
    df = pd.DataFrame(results)
    
    print("=" * 80)
    print("CARDIO RISK AI - FINAL MODEL COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # Save to CSV
    df.to_csv("models/final_results_summary.csv", index=False)
    print("\n✅ Results saved to models/final_results_summary.csv")
    
    return df

if __name__ == "__main__":
    create_summary()