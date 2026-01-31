"""
Quick Evaluation Runner Script

Runs all baseline model evaluations in one go.
Generates complete metrics, visualizations, and report.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"✗ {description} failed")
        print(f"Error: {result.stderr}")
        return False
    
    return True

def main():
    print("\n" + "="*80)
    print("BASELINE MODEL EVALUATION PIPELINE")
    print("="*80)
    
    # Check if models exist
    models_dir = Path("models/baseline")
    if not models_dir.exists():
        print("\n✗ Error: models/baseline directory not found")
        print("Please run phase_h_baseline_models.py first to train models")
        return
    
    if not (models_dir / "xgboost_model.pkl").exists():
        print("\n✗ Error: XGBoost model not found")
        print("Please run phase_h_baseline_models.py first to train models")
        return
    
    print("\n✓ Models directory found")
    print(f"  XGBoost: {'✓' if (models_dir / 'xgboost_model.pkl').exists() else '✗'}")
    print(f"  MLP: {'✓' if (models_dir / 'mlp_model_best.pt').exists() else '✗'}")
    
    # Run enhanced evaluation
    success = run_command(
        "python scripts/enhanced_model_evaluation.py",
        "Enhanced Model Evaluation (with CI, DeLong test, complete subgroups)"
    )
    
    if not success:
        print("\n⚠ Enhanced evaluation failed, but continuing...")
    
    # Check generated files
    print("\n" + "="*80)
    print("GENERATED ARTIFACTS")
    print("="*80)
    
    artifacts = {
        "Model Files": [
            "models/baseline/xgboost_model.pkl",
            "models/baseline/mlp_model_best.pt"
        ],
        "Evaluation Metrics": [
            "models/baseline/model_evaluation_metrics.csv",
            "models/baseline/subgroup_analysis.csv",
            "models/baseline/subgroup_analysis_complete.csv",
            "models/baseline/fairness_disparity_analysis.csv"
        ],
        "Visualizations": [
            "models/baseline/plots/roc_curves.png",
            "models/baseline/plots/roc_curves_with_ci.png",
            "models/baseline/plots/precision_recall_curves.png",
            "models/baseline/plots/calibration_plots.png",
            "models/baseline/plots/subgroup_performance.png",
            "models/baseline/plots/subgroup_comparison.png"
        ],
        "Reports": [
            "models/baseline/baseline_models_report.md",
            "models/baseline/phase_h_summary.txt"
        ],
        "Documentation": [
            "docs/MODEL_IMPROVEMENT_RECOMMENDATIONS.md",
            "EVALUATION_SUMMARY.md"
        ]
    }
    
    all_found = True
    for category, files in artifacts.items():
        print(f"\n{category}:")
        for file_path in files:
            exists = Path(file_path).exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file_path}")
            if not exists:
                all_found = False
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    if all_found:
        print("\n✓ All evaluations complete!")
    else:
        print("\n⚠ Some files are missing (may not have been generated yet)")
    
    print("\nNext steps:")
    print("1. Review baseline_models_report.md for comprehensive results")
    print("2. Check EVALUATION_SUMMARY.md for quick assessment")
    print("3. See MODEL_IMPROVEMENT_RECOMMENDATIONS.md for suggested improvements")
    print("4. Address age bias (Priority #1) before clinical deployment")
    
    print("\nKey Metrics (from CSV):")
    try:
        import pandas as pd
        df = pd.read_csv("models/baseline/model_evaluation_metrics.csv")
        test_metrics = df[df['Split'] == 'Test']
        print("\n", test_metrics.to_string(index=False))
    except Exception as e:
        print(f"  (Could not load metrics: {e})")
    
    print("\nFairness Check:")
    try:
        import pandas as pd
        fairness_df = pd.read_csv("models/baseline/fairness_disparity_analysis.csv")
        print("\n", fairness_df.to_string(index=False))
    except Exception as e:
        print(f"  (Could not load fairness metrics: {e})")

if __name__ == "__main__":
    main()
