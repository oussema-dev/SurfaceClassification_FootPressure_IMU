import numpy as np

def print_metrics(accuracies, f1_scores, sensitivities, specificities):
    metrics = {
        "Accuracy": accuracies,
        "F1-score": f1_scores,
        "Sensitivity": sensitivities,
        "Specificity": specificities
    }
    for name, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{name}: {mean:.2f} ({std:.2f})")