import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score
)

import config


DATA_DIR    = config.OUTPUT_DIR
OUTPUT_DIR  = config.OUTPUT_DIR
MODEL_PATH  = os.path.join(OUTPUT_DIR, "best_sound_model.keras")
THRESH_PATH = os.path.join(OUTPUT_DIR, "optimal_threshold.npy")

if os.path.exists(THRESH_PATH):
    THRESHOLD = float(np.load(THRESH_PATH))
else:
    THRESHOLD = 0.65
    print(f"  [WARN] optimal_threshold.npy not found — using default {THRESHOLD}")

CLASS_NAMES = ["Faulty (0)", "Healthy (1)"]

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "fordA":  "#2563EB",   # blue   — clean test
    "fordB":  "#DC2626",   # red    — noisy test
    "good":   "#16A34A",   # green
    "bad":    "#DC2626",   # red
    "warn":   "#D97706",   # amber
}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:

    y_pred = (y_prob >= threshold).astype(int)

    missed_faults = int(np.sum((y_pred == 1) & (y_true == 0)))
    false_alarms  = int(np.sum((y_pred == 0) & (y_true == 1)))
    n_faulty      = int((y_true == 0).sum())
    n_healthy     = int((y_true == 1).sum())

    return {
        "accuracy"      : accuracy_score(y_true, y_pred),
        "precision"     : precision_score(y_true, y_pred, zero_division=0),
        "recall"        : recall_score(y_true, y_pred, zero_division=0),
        "f1"            : f1_score(y_true, y_pred, zero_division=0),
        "auc"           : roc_auc_score(y_true, y_prob),
        "missed_faults" : missed_faults,
        "false_alarms"  : false_alarms,
        "fault_recall"  : 1.0 - missed_faults / (n_faulty  + 1e-8),
        "missed_rate"   : missed_faults / (n_faulty  + 1e-8),
        "alarm_rate"    : false_alarms  / (n_healthy + 1e-8),
        "n_faulty"      : n_faulty,
        "n_healthy"     : n_healthy,
        "y_pred"        : y_pred,
        "threshold"     : threshold,
    }



def print_safety_report(metrics: dict, split_name: str):

    t  = metrics["threshold"]
    mf = metrics["missed_faults"]
    fa = metrics["false_alarms"]
    nf = metrics["n_faulty"]
    nh = metrics["n_healthy"]

    print(f"\n  {'═'*56}")
    print(f"  SAFETY REPORT — {split_name}")
    print(f"  Threshold = {t:.3f}  |  model output = P(Healthy)")
    print(f"  {'═'*56}")
    print(f"""
  PROBABILITY INTERPRETATION:
    prob >= {t:.3f}  →  HEALTHY prediction
    prob <  {t:.3f}  →  FAULTY  prediction

  ┌─────────────────────────────────────────────────────┐
  │  ❌  MISSED FAULTS  : {mf:>4}  /  {nf}  faulty samples  │
  │      (Faulty engine predicted as Healthy)            │
  │      Missed-fault rate : {metrics['missed_rate']*100:>5.1f}%                   │
  │                                                      │
  │  ⚠️   FALSE ALARMS   : {fa:>4}  /  {nh}  healthy samples │
  │      (Healthy engine predicted as Faulty)            │
  │      False-alarm rate  : {metrics['alarm_rate']*100:>5.1f}%                   │
  └─────────────────────────────────────────────────────┘

  STANDARD METRICS (at threshold = {t:.3f}):
    Accuracy      : {metrics['accuracy']:.4f}
    Precision     : {metrics['precision']:.4f}
    Recall        : {metrics['recall']:.4f}   (of Healthy class)
    F1-Score      : {metrics['f1']:.4f}
    AUC (ROC)     : {metrics['auc']:.4f}   (threshold-independent)
    Fault Recall  : {metrics['fault_recall']:.4f}   (of Faulty class)  ← KEY SAFETY METRIC
""")


def print_comparison_table(metrics_A: dict, metrics_B: dict):
    """Side-by-side summary of FordA vs FordB results."""
    print("\n" + "=" * 62)
    print("  FINAL COMPARISON TABLE")
    print("=" * 62)
    print(f"\n  {'Metric':<18}  {'FordA (Clean)':>14}  {'FordB (Noisy)':>14}  {'Drop':>8}")
    print(f"  {'─'*60}")

    rows = [
        ("Accuracy",       "accuracy"),
        ("Precision",      "precision"),
        ("Recall (Hlthy)", "recall"),
        ("F1-Score",       "f1"),
        ("AUC",            "auc"),
        ("Fault Recall",   "fault_recall"),
    ]
    for label, key in rows:
        a = metrics_A[key]
        b = metrics_B[key]
        print(f"  {label:<18}  {a:>14.4f}  {b:>14.4f}  {a-b:>+8.4f}")

    print(f"\n  {'Missed Faults':<18}  {metrics_A['missed_faults']:>14}  "
          f"{metrics_B['missed_faults']:>14}")
    print(f"  {'False Alarms':<18}  {metrics_A['false_alarms']:>14}  "
          f"{metrics_B['false_alarms']:>14}")
    print(f"  {'Threshold':<18}  {metrics_A['threshold']:>14.3f}  "
          f"{metrics_B['threshold']:>14.3f}")


def domain_shift_analysis(metrics_A: dict, metrics_B: dict):
    """Print domain shift verdict and safety interpretation."""
    f1_drop  = metrics_A["f1"]          - metrics_B["f1"]
    auc_drop = metrics_A["auc"]         - metrics_B["auc"]
    mf_diff  = metrics_B["missed_faults"] - metrics_A["missed_faults"]

    print("\n" + "=" * 62)
    print("  DOMAIN SHIFT ANALYSIS")
    print("=" * 62)

    print(f"""
  Metric drops  (FordA → FordB):
  ┌───────────────────────────────────────────┐
  │  AUC drop         : {auc_drop:+.4f}              │
  │  F1 drop          : {f1_drop:+.4f}              │
  │  Extra missed faults on FordB : {mf_diff:>+4}        │
  └───────────────────────────────────────────┘
""")

    # Generalisation verdict based on AUC drop (threshold-independent)
    if abs(auc_drop) < 0.05:
        verdict = "STRONG GENERALISATION ✅"
        detail  = (
            "AUC drop < 5%. The model's ranking quality is consistent\n"
            "  across clean and noisy domains. Augmentation successfully\n"
            "  simulated field conditions during training."
        )
    elif abs(auc_drop) < 0.12:
        verdict = "MODERATE GENERALISATION ⚠️"
        detail  = (
            "Some ranking degradation on the noisy domain.\n"
            "  The model partially crosses the domain gap but is not\n"
            "  fully robust to real microphone conditions."
        )
    else:
        verdict = "DOMAIN SHIFT IMPACT HIGH 🔴"
        detail  = (
            "Significant AUC drop on noisy domain.\n"
            "  Consider: lower AUG_NOISE_SNR_DB (more noise in training),\n"
            "  or collect labelled FordB samples for fine-tuning."
        )

    print(f"  Generalisation: {verdict}")
    print(f"\n  {detail}")

    # Safety verdict based on missed faults
    print(f"\n  Safety (FordB missed faults = {metrics_B['missed_faults']}):")
    if metrics_B["missed_fault_rate"] < 0.10:
        print("    Missed-fault rate < 10% — acceptable for production.")
    elif metrics_B["missed_fault_rate"] < 0.20:
        print("     Missed-fault rate 10–20% — marginal; raise threshold.")
    else:
        print("    Missed-fault rate > 20% — not safe for production.")


# 3. PLOT FUNCTIONS
def plot_confusion_matrix(cm: np.ndarray, title: str, color: str, ax: plt.Axes):
    """
    Draw annotated confusion matrix heatmap.
    Cells show raw counts and row-percentages.
    """
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([
        [f"{cnt}\n({pct:.1f}%)" for cnt, pct in zip(row_c, row_p)]
        for row_c, row_p in zip(cm, cm_pct)
    ])
    sns.heatmap(
        cm, annot=labels, fmt="", ax=ax,
        cmap=sns.light_palette(color, as_cmap=True),
        linewidths=1.5, linecolor="white",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar=False,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)


def plot_roc_curves(
    y_true_A, y_prob_A,
    y_true_B, y_prob_B,
    save_dir: str,
):
    """ROC curves for FordA and FordB on the same axes."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("ROC Curve Comparison — FordA vs FordB",
                 fontsize=13, fontweight="bold")

    for y_true, y_prob, label, color in [
        (y_true_A, y_prob_A, "FordA (clean)", COLORS["fordA"]),
        (y_true_B, y_prob_B, "FordB (noisy)", COLORS["fordB"]),
    ]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{label}  (AUC = {auc_val:.4f})")

    # Mark operating point on FordB ROC
    y_pred_B = (y_prob_B >= THRESHOLD).astype(int)
    tp = int(np.sum((y_pred_B == 1) & (y_true_B == 1)))
    fp = int(np.sum((y_pred_B == 1) & (y_true_B == 0)))
    fn = int(np.sum((y_pred_B == 0) & (y_true_B == 1)))
    tn = int(np.sum((y_pred_B == 0) & (y_true_B == 0)))
    op_fpr = fp / (fp + tn + 1e-8)
    op_tpr = tp / (tp + fn + 1e-8)
    ax.scatter([op_fpr], [op_tpr], color="black", s=80, zorder=5,
               label=f"Operating point (thresh={THRESHOLD:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    path = os.path.join(save_dir, "roc_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC comparison saved → {path}")


def plot_metric_comparison(metrics_A: dict, metrics_B: dict, save_dir: str):
    """Bar chart comparing standard metrics between FordA and FordB."""
    keys   = ["accuracy", "precision", "recall", "f1", "auc", "fault_recall"]
    vals_A = [metrics_A[k] for k in keys]
    vals_B = [metrics_B[k] for k in keys]
    drops  = [a - b for a, b in zip(vals_A, vals_B)]

    x, width = np.arange(len(keys)), 0.35
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC", "FaultRecall"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Performance — Clean (FordA) vs Noisy (FordB)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    bars_A = ax.bar(x - width/2, vals_A, width, label="FordA (clean)",
                    color=COLORS["fordA"], alpha=0.87, edgecolor="white", lw=1.5)
    bars_B = ax.bar(x + width/2, vals_B, width, label="FordB (noisy)",
                    color=COLORS["fordB"], alpha=0.87, edgecolor="white", lw=1.5)
    for bar in list(bars_A) + list(bars_B):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim([0, 1.14]); ax.set_ylabel("Score")
    ax.set_title("All Metrics Side-by-Side", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    bar_colors = [COLORS["good"] if d >= 0 else COLORS["bad"] for d in drops]
    bars_d = ax2.bar(labels, drops, color=bar_colors,
                     edgecolor="white", lw=1.5, alpha=0.87)
    ax2.axhline(0, color="black", lw=1.2, linestyle="--")
    for bar, drop in zip(bars_d, drops):
        va = "bottom" if drop >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.003 if drop >= 0 else -0.003),
                 f"{drop:+.3f}", ha="center", va=va, fontsize=9, fontweight="bold")
    ax2.set_title("Drop  (FordA − FordB)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Δ Score  (positive = FordA better)")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "metric_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Metric comparison saved → {path}")


def plot_full_dashboard(
    cm_A, cm_B,
    y_true_A, y_prob_A,
    y_true_B, y_prob_B,
    metrics_A, metrics_B,
    save_dir: str,
):
    """
    Master evaluation dashboard:
      Row 0: confusion matrices + per-class accuracy bar chart
      Row 1: prediction distributions + safety summary text box
    """
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"Ford Engine CNN — Evaluation Dashboard  (Threshold = {THRESHOLD:.3f})\n"
        f"FordA (Clean) vs FordB (Noisy Domain Shift)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    ax_cm_A = fig.add_subplot(gs[0, 0])
    ax_cm_B = fig.add_subplot(gs[0, 1])
    plot_confusion_matrix(
        cm_A,
        f"FordA (Clean) — Missed: {metrics_A['missed_faults']}  Alarms: {metrics_A['false_alarms']}",
        COLORS["fordA"], ax_cm_A,
    )
    plot_confusion_matrix(
        cm_B,
        f"FordB (Noisy) — Missed: {metrics_B['missed_faults']}  Alarms: {metrics_B['false_alarms']}",
        COLORS["fordB"], ax_cm_B,
    )

    ax_bar = fig.add_subplot(gs[0, 2])
    for cm, label, color, offset in [
        (cm_A, "FordA", COLORS["fordA"], -0.18),
        (cm_B, "FordB", COLORS["fordB"],  0.18),
    ]:
        pca = cm.diagonal() / cm.sum(axis=1)
        ax_bar.bar(np.array([0, 1]) + offset, pca, 0.34,
                   label=label, color=color, alpha=0.85, edgecolor="white")
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax_bar.set_ylim([0, 1.14])
    ax_bar.set_ylabel("Per-Class Accuracy")
    ax_bar.set_title("Per-Class Accuracy", fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=9); ax_bar.grid(True, alpha=0.3, axis="y")


    for col, (y_true, y_prob, name, color) in enumerate([
        (y_true_A, y_prob_A, "FordA (Clean)", COLORS["fordA"]),
        (y_true_B, y_prob_B, "FordB (Noisy)", COLORS["fordB"]),
    ]):
        ax_d = fig.add_subplot(gs[1, col])
        for cls, cls_name, alpha in [(0, "Faulty", 0.70), (1, "Healthy", 0.50)]:
            mask = y_true == cls
            ax_d.hist(y_prob[mask], bins=40, alpha=alpha, color=color,
                      label=cls_name, edgecolor="none", density=True)
        ax_d.axvline(THRESHOLD, color="black", linestyle="--", lw=1.8,
                     label=f"Threshold = {THRESHOLD:.3f}")
        ax_d.set_title(f"{name} — P(Healthy) Distribution",
                       fontsize=10, fontweight="bold")
        ax_d.set_xlabel("Predicted Probability  [P(Healthy)]")
        ax_d.set_ylabel("Density")
        ax_d.legend(fontsize=9); ax_d.grid(True, alpha=0.3)
    
    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis("off")

    summary = (
        f"SAFETY SUMMARY\n"
        f"──────────────────────────────\n\n"
        f"Threshold = {THRESHOLD:.3f}\n"
        f"(model output = P(Healthy))\n\n"
        f"MISSED FAULTS (❌ dangerous):\n"
        f"  FordA : {metrics_A['missed_faults']:>4} / {metrics_A['n_faulty']}\n"
        f"  FordB : {metrics_B['missed_faults']:>4} / {metrics_B['n_faulty']}\n\n"
        f"FALSE ALARMS (⚠️ acceptable):\n"
        f"  FordA : {metrics_A['false_alarms']:>4} / {metrics_A['n_healthy']}\n"
        f"  FordB : {metrics_B['false_alarms']:>4} / {metrics_B['n_healthy']}\n\n"
        f"FAULT RECALL:\n"
        f"  FordA : {metrics_A['fault_recall']:.4f}\n"
        f"  FordB : {metrics_B['fault_recall']:.4f}\n\n"
        f"AUC (threshold-independent):\n"
        f"  FordA : {metrics_A['auc']:.4f}\n"
        f"  FordB : {metrics_B['auc']:.4f}"
    )
    ax_txt.text(
        0.05, 0.97, summary,
        transform=ax_txt.transAxes,
        fontsize=9.5, verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#FFF7ED",
                  edgecolor="#D97706", linewidth=2),
    )

    path = os.path.join(save_dir, "full_evaluation_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Full dashboard saved → {path}")


# 4. MAIN
def main():
    # ── Load model ─────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  LOADING MODEL")
    print("=" * 62)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun train.py first."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Input   : {model.input_shape}")
    print(f"  Output  : {model.output_shape}")
    print(f"  Threshold: {THRESHOLD:.3f}  "
          f"({'loaded from file' if os.path.exists(THRESH_PATH) else 'default'})\n")

    print("=" * 62)
    print("  LOADING TEST SETS")
    print("=" * 62)
    X_testA = np.load(os.path.join(DATA_DIR, "final_X_testA.npy"))
    y_testA = np.load(os.path.join(DATA_DIR, "final_y_testA.npy"))
    X_testB = np.load(os.path.join(DATA_DIR, "final_X_testB.npy"))
    y_testB = np.load(os.path.join(DATA_DIR, "final_y_testB.npy"))

    print(f"  FordA: X={X_testA.shape}  y={y_testA.shape}"
          f"  Faulty={int((y_testA==0).sum())}  Healthy={int((y_testA==1).sum())}")
    print(f"  FordB: X={X_testB.shape}  y={y_testB.shape}"
          f"  Faulty={int((y_testB==0).sum())}  Healthy={int((y_testB==1).sum())}\n")

    # ── Predict ────────────────────────────────────────────────────────────────
    print("  Running inference …")
    y_prob_A = model.predict(X_testA, verbose=0).flatten()
    y_prob_B = model.predict(X_testB, verbose=0).flatten()

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics_A = compute_metrics(y_testA, y_prob_A, THRESHOLD)
    metrics_B = compute_metrics(y_testB, y_prob_B, THRESHOLD)

    # Attach missed_fault_rate for domain_shift_analysis
    metrics_A["missed_fault_rate"] = metrics_A["missed_faults"] / (metrics_A["n_faulty"] + 1e-8)
    metrics_B["missed_fault_rate"] = metrics_B["missed_faults"] / (metrics_B["n_faulty"] + 1e-8)

    # ── Safety reports ─────────────────────────────────────────────────────────
    print_safety_report(metrics_A, "FordA Test (Clean)")
    print_safety_report(metrics_B, "FordB Test (Noisy — Domain Shift)")

    # ── Sklearn classification reports ────────────────────────────────────────
    print("=" * 62)
    print("  CLASSIFICATION REPORT — FordA (Clean)")
    print("=" * 62)
    print(classification_report(
        y_testA, metrics_A["y_pred"],
        target_names=CLASS_NAMES, digits=4,
    ))

    print("=" * 62)
    print("  CLASSIFICATION REPORT — FordB (Noisy)")
    print("=" * 62)
    print(classification_report(
        y_testB, metrics_B["y_pred"],
        target_names=CLASS_NAMES, digits=4,
    ))

    # ── Confusion matrices ─────────────────────────────────────────────────────
    cm_A = confusion_matrix(y_testA, metrics_A["y_pred"])
    cm_B = confusion_matrix(y_testB, metrics_B["y_pred"])

    print("  Confusion Matrix — FordA (rows=True, cols=Predicted):")
    print(f"    {cm_A}")
    print("  Confusion Matrix — FordB:")
    print(f"    {cm_B}")
    print(f"\n  [FordB] Row 0 = Faulty:  TN={cm_B[0,0]}  FP={cm_B[0,1]} ← missed faults")
    print(f"  [FordB] Row 1 = Healthy: FN={cm_B[1,0]}  TP={cm_B[1,1]} ← false alarms")

    # ── Domain shift analysis ──────────────────────────────────────────────────
    domain_shift_analysis(metrics_A, metrics_B)

    # ── Comparison table ───────────────────────────────────────────────────────
    print_comparison_table(metrics_A, metrics_B)

    # ── Save plots ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  SAVING PLOTS")
    print("=" * 62)
    plot_full_dashboard(
        cm_A, cm_B,
        y_testA, y_prob_A,
        y_testB, y_prob_B,
        metrics_A, metrics_B,
        OUTPUT_DIR,
    )
    plot_roc_curves(y_testA, y_prob_A, y_testB, y_prob_B, OUTPUT_DIR)
    plot_metric_comparison(metrics_A, metrics_B, OUTPUT_DIR)

    print("\n" + "=" * 62)
    print("  EVALUATION COMPLETE")
    print("=" * 62)


if __name__ == "__main__":
    main()
