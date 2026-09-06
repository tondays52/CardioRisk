"""
CardioRisk AI - Analyze Captured PPG Video
"""
import os
import sys
import json
from typing import Any, Dict

# Ensure project root and scripts directory are in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
for _p in [_current_dir, _project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from scripts.ppg_processing import process_ppg_video
except ImportError:
    from ppg_processing import process_ppg_video  # type: ignore[import-not-found]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_capture.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Optional demographics
    age = int(input("Age: ") or 45)
    gender = int(input("Gender (1=F, 2=M): ") or 2)
    height_cm = float(input("Height (cm): ") or 175)
    weight_kg = float(input("Weight (kg): ") or 80)
    
    print("\nAnalyzing PPG signal...")
    result: Dict[str, Any] = process_ppg_video(video_path, age, gender, height_cm, weight_kg)
    
    print("\n" + "=" * 40)
    print("PPG Analysis Results")
    print("=" * 40)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        hrv: Dict[str, Any] = result['hrv']
        stiffness: Dict[str, Any] = result['stiffness']
        bp: Any = result.get('blood_pressure')
        
        print(f"\nHeart Rate: {hrv['heart_rate_bpm']} bpm")
        print(f"SDNN: {hrv['sdnn_ms']} ms")
        print(f"RMSSD: {hrv['rmssd_ms']} ms")
        print(f"pNN50: {hrv['pnn50_pct']}%")
        print(f"LF/HF ratio: {hrv['lf_hf_ratio']}")
        
        print(f"\nStiffness Index: {stiffness['stiffness_index']}")
        print(f"Reflection Index: {stiffness['reflection_index']}")
        
        if bp:
            print(f"\nEstimated BP: {bp['estimated_systolic_bp']}/{bp['estimated_diastolic_bp']} mmHg")
        
        print(f"\nSignal Quality:")
        print(f"  Beats detected: {result['signal_quality']['num_beats']}")
        print(f"  SNR: {result['signal_quality']['snr_estimate']}")
    
    # Save results
    with open("data/raw/ppg/analysis_results.json", "w") as f:
        json.dump(result, f, indent=4, default=str)
    print("\nResults saved to data/raw/ppg/analysis_results.json")