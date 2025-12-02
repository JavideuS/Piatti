import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
import time
import joblib


def evaluate_model(
    pipeline, model_name,
    X_train, y_train,
    X_test, y_test,
    le,
    verbose=True,
    save_model_path=None,
    save_func=None
):
    """
    Train and evaluate a single model.
    Returns: dict of metrics, predictions, pipeline, and timing info.
    """

    # Train
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time

    if verbose:
        print(f"✓ Trained in {train_time:.2f}s")

    # Predict
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    recall    = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    f1        = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    overfit   = train_acc - test_acc

    # Try probabilities & ROC-AUC
    try:
        y_proba = pipeline.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except Exception:
        y_proba = None
        roc_auc = np.nan

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    results = {
        'model_name': model_name,
        'pipeline': pipeline,
        'train_time': train_time,
        'inference_time_ms': inference_time * 1000,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'overfit': overfit,
        'y_test': y_test,
        'y_pred': y_test_pred,
        'y_proba': y_proba,
        'confusion_matrix': cm,
        'X_test': X_test,
        'label_encoder': le
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  {model_name} - Summary")
        print(f"{'─'*60}")
        print(f"  Train Acc:   {train_acc:.4f}")
        print(f"  Test Acc:    {test_acc:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  F1:          {f1:.4f}")
        if not np.isnan(roc_auc):
            print(f"  ROC-AUC:     {roc_auc:.4f}")
        print(f"  Overfit:     {overfit:.4f}")
        print(f"  Train Time:  {train_time:.2f}s")
        print(f"  Inference:   {inference_time * 1000:.3f}ms/sample")
        print(f"{'─'*60}\n")

    # Save model if requested
    if save_model_path:
        original_descriptor_extractor = None
        encoding_step = None

        # Locate the encoding step and save original descriptor_extractor
        for name, step in pipeline.steps:
            if hasattr(step, 'descriptor_extractor'):
                encoding_step = step
                original_descriptor_extractor = step.descriptor_extractor
                break

        if encoding_step is not None and save_func is not None:
            # Patch temporarily
            encoding_step.descriptor_extractor = save_func
            if verbose:
                print(f"✓ Updated descriptor extractor to {save_func.__name__}")
        elif verbose:
            print("⚠ Warning: No step with 'descriptor_extractor' found!" if encoding_step is None else
                "⚠ Warning: save_func is None — skipping patch.")

        try:
            model_bundle = {
                'pipeline': pipeline,
                'label_encoder': le
            }
            joblib.dump(model_bundle, save_model_path)
            if verbose:
                print(f"✓ Model saved to {save_model_path}")
        finally:
            # Always restore original, even on error
            if encoding_step is not None and original_descriptor_extractor is not None:
                encoding_step.descriptor_extractor = original_descriptor_extractor
                if verbose:
                    print("✓ Restored original descriptor extractor.")

    return results

def plot_confusion_matrix(results, normalize=False, figsize=(10, 8), save_path=None):
    cm = results['confusion_matrix']
    le = results['label_encoder']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = f"{results['model_name']}\nNormalized Confusion Matrix"
    else:
        fmt = 'd'
        title = f"{results['model_name']}\nConfusion Matrix - Test accuracy {results['test_accuracy']:.4f}"

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.show()


def plot_roc_curve(results, figsize=(12, 5), save_path=None):
    y_proba = results['y_proba']
    if y_proba is None:
        print("⚠ Skipping ROC: model doesn’t support predict_proba()")
        return

    y_test = results['y_test']
    le = results['label_encoder']
    n_classes = len(le.classes_)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fpr = dict()
    tpr = dict()

    # Compute per-class ROC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = np.trapz(mean_tpr, all_fpr)

    # Plot
    ax = axes[0]
    ax.plot(all_fpr, mean_tpr, linewidth=3, label=f'Macro-avg (AUC={macro_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    ax.set_xlabel('FPR', fontweight='bold')
    ax.set_ylabel('TPR', fontweight='bold')
    ax.set_title(f"{results['model_name']}\nMacro-Average ROC", fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # Per-class (top 10)
    ax = axes[1]
    for i in range(min(n_classes, 10)):
        auc_i = np.trapz(tpr[i], fpr[i])
        ax.plot(fpr[i], tpr[i], label=f'{le.classes_[i]} ({auc_i:.2f})', linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('FPR', fontweight='bold')
    ax.set_ylabel('TPR', fontweight='bold')
    ax.set_title(f"{results['model_name']}\nPer-Class ROC", fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    plt.show()

def print_classification_report(results):
    y_test, y_pred = results['y_test'], results['y_pred']
    le = results['label_encoder']
    print(f"\n{'='*70}")
    print(f"Classification Report: {results['model_name']}")
    print(f"{'='*70}\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))