"""
CardioRisk AI - Model Performance Visualization
Creates professional comparison plots for all models.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

def create_comparison_plots():
    """Generate bar charts comparing model performance."""
    
    # Data
    models = ['Heart Sound\nCNN', 'ECG\nCNN', 'Ensemble\n(ECG+HS)']
    accuracy = [75.31, 81.00, 86.00]
    auc = [0.800, 0.936, 0.942]
    normal_recall = [75.00, 66.36, 74.77]
    abnormal_recall = [75.00, 97.85, 98.92]
    
    # Colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CardioRisk AI - Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Plot
    bars1 = ax1.bar(models, accuracy, color=colors, edgecolor='black', linewidth=1)
    ax1.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(70, 90)
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, accuracy)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. AUC Plot
    bars2 = ax2.bar(models, auc, color=colors, edgecolor='black', linewidth=1)
    ax2.set_title('ROC-AUC Score', fontsize=14, fontweight='bold')
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_ylim(0.7, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, auc)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Recall Plot
    x = np.arange(len(models))
    width = 0.35
    bars3 = ax3.bar(x - width/2, normal_recall, width, label='Normal Recall', 
                    color='#4ECDC4', edgecolor='black', linewidth=1)
    bars4 = ax3.bar(x + width/2, abnormal_recall, width, label='Abnormal Recall', 
                    color='#FF6B6B', edgecolor='black', linewidth=1)
    ax3.set_title('Recall by Class', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Recall (%)', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.set_ylim(60, 105)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, normal_recall):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars4, abnormal_recall):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Summary Table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [
        ['Model', 'Accuracy', 'AUC', 'Normal Recall', 'Abnormal Recall'],
        ['Heart Sound CNN', '75.31%', '0.800', '~75%', '~75%'],
        ['ECG CNN', '81.00%', '0.936', '66.36%', '97.85%'],
        ['Ensemble', '86.00%', '0.942', '74.77%', '98.92%']
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center', 
                      colWidths=[0.25, 0.15, 0.15, 0.22, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    # Style the header row
    for j in range(5):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Style the best row (Ensemble)
    for j in range(5):
        table[(3, j)].set_facecolor('#45B7D1')
        table[(3, j)].set_text_props(weight='bold')
    
    ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Plot saved as models/model_comparison.png")
    
    plt.show()
    
    return fig

def create_confusion_matrices():
    """Create confusion matrix visualizations."""
    
    # Data from your results
    models = ['ECG CNN', 'Ensemble']
    
    # Confusion matrices: [[TN, FP], [FN, TP]]
    ecg_cm = np.array([[71, 36], [2, 91]])
    ensemble_cm = np.array([[80, 27], [1, 92]])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold')
    
    # Plot ECG confusion matrix
    im1 = ax1.imshow(ecg_cm, cmap='Blues', interpolation='nearest')
    ax1.set_title('ECG CNN (81.00% Accuracy)', fontsize=13, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Predicted Normal', 'Predicted Abnormal'])
    ax1.set_yticklabels(['Actual Normal', 'Actual Abnormal'])
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('Actual', fontsize=11)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, ecg_cm[i, j],
                          ha="center", va="center", color="white" if ecg_cm[i, j] > 50 else "black",
                          fontsize=14, fontweight='bold')
    
    # Plot Ensemble confusion matrix
    im2 = ax2.imshow(ensemble_cm, cmap='Greens', interpolation='nearest')
    ax2.set_title('Ensemble (86.00% Accuracy)', fontsize=13, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Predicted Normal', 'Predicted Abnormal'])
    ax2.set_yticklabels(['Actual Normal', 'Actual Abnormal'])
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Actual', fontsize=11)
    
    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, ensemble_cm[i, j],
                          ha="center", va="center", color="white" if ensemble_cm[i, j] > 50 else "black",
                          fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✅ Plot saved as models/confusion_matrices.png")
    
    plt.show()
    
    return fig

def create_training_curves():
    """Plot training history if available."""
    
    history_path = "models/ecg/training_history_final.csv"
    
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('ECG CNN Training History', fontsize=16, fontweight='bold')
        
        # Accuracy
        ax1.plot(df['accuracy'], label='Training', linewidth=2)
        ax1.plot(df['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Accuracy Over Epochs', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Loss
        ax2.plot(df['loss'], label='Training', linewidth=2)
        ax2.plot(df['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Loss Over Epochs', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_curves.png', dpi=300, bbox_inches='tight')
        print("✅ Training curves saved as models/training_curves.png")
        
        plt.show()
        return fig
    else:
        print("⚠️ Training history not found. Skipping training curves.")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("CardioRisk AI - Generating Visualizations")
    print("=" * 60)
    
    # Create all plots
    create_comparison_plots()
    create_confusion_matrices()
    create_training_curves()
    
    print("\n" + "=" * 60)
    print("🎉 All visualizations created successfully!")
    print("📁 Check the 'models/' folder for saved images.")
    print("=" * 60)