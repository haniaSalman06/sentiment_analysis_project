from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, name="Model", results_dir="results"):
    y_pred = model.predict(X_test)
    metrics = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1 Score':  f1_score(y_test, y_pred),
    }
    print(f"\n--- {name} Evaluation ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative','Positive'],
                yticklabels=['Negative','Positive'])
    plt.title(f'Confusion Matrix — {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = os.path.join(results_dir, f'confusion_matrix_{name.lower().replace(" ","_")}.png')
    plt.savefig(path)
    plt.show()
    print(f"Confusion matrix saved to {path}")
    return metrics