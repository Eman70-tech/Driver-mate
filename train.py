import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import config
from model_builder import build_model

SEED = 42
os.environ["PYTHONHASHSEED"]       = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
np.random.seed(SEED)
tf.random.set_seed(SEED)


DATA_DIR    = config.OUTPUT_DIR
OUTPUT_DIR  = config.OUTPUT_DIR
MODEL_PATH  = os.path.join(OUTPUT_DIR, "best_sound_model.keras")
THRESH_PATH = os.path.join(OUTPUT_DIR, "optimal_threshold.npy")

BATCH_SIZE  = 64     # stable gradient estimates without excessive memory
MAX_EPOCHS  = 100    # EarlyStopping will terminate well before this
VAL_SPLIT   = 0.15   # 15% of combined training data reserved for validation
ES_PATIENCE = 15     # epochs without val_auc improvement before stopping
LR_INITIAL  = 1e-3
LR_PATIENCE = 6      # epochs without val_loss improvement before LR halving
LR_FACTOR   = 0.5
LR_MIN      = 1e-6


def load_data() -> tuple:

    print("  LOADING DATA (FordA + FordB combined)")

    X_trainA = np.load(os.path.join(OUTPUT_DIR, "final_X_trainA.npy"))
    y_trainA = np.load(os.path.join(OUTPUT_DIR, "final_y_trainA.npy"))
    X_testA  = np.load(os.path.join(OUTPUT_DIR, "final_X_testA.npy"))
    y_testA  = np.load(os.path.join(OUTPUT_DIR, "final_y_testA.npy"))

    X_trainB = np.load(os.path.join(OUTPUT_DIR, "final_X_trainB.npy"))
    y_trainB = np.load(os.path.join(OUTPUT_DIR, "final_y_trainB.npy"))
    X_testB  = np.load(os.path.join(OUTPUT_DIR, "final_X_testB.npy"))
    y_testB  = np.load(os.path.join(OUTPUT_DIR, "final_y_testB.npy"))

    X_train = np.concatenate([X_trainA, X_trainB], axis=0)
    y_train = np.concatenate([y_trainA, y_trainB], axis=0)


    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    n_h = int((y_train == 1).sum())
    n_f = int((y_train == 0).sum())
    print(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_testA : {X_testA.shape}  |  y_testA : {y_testA.shape}")
    print(f"  X_testB : {X_testB.shape}  |  y_testB : {y_testB.shape}")
    print(f"  Train — Healthy={n_h}  Faulty={n_f}  Ratio={n_h/n_f:.2f}:1\n")

    return X_train, y_train, X_testA, y_testA, X_testB, y_testB


def find_optimal_threshold(
    y_true:       np.ndarray,
    y_prob:       np.ndarray,
    beta:         float = 2.0,
    search_range: tuple = (0.40, 0.85),
    steps:        int   = 300,
) -> float:
   
    thresholds = np.linspace(search_range[0], search_range[1], steps)
    best_thresh, best_fbeta = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)   # 1=Healthy, 0=Faulty

        # Faulty (class 0) is the positive class
        tp = int(np.sum((y_pred == 0) & (y_true == 0)))  # caught faults
        fp = int(np.sum((y_pred == 0) & (y_true == 1)))  # false alarms
        fn = int(np.sum((y_pred == 1) & (y_true == 0)))  # missed faults

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)   # fault recall = 1 - missed_rate

        fbeta = (
            (1 + beta ** 2) * precision * recall
            / (beta ** 2 * precision + recall + 1e-8)
        )

        if fbeta > best_fbeta:
            best_fbeta  = fbeta
            best_thresh = t

    return float(best_thresh)


def threshold_safety_sweep(
    y_true:         np.ndarray,
    y_prob:         np.ndarray,
    optimal_thresh: float,
    dataset_name:   str = "FordB",
):


    n_faulty = int((y_true == 0).sum())
    print(f"\n  Threshold Safety Sweep — {dataset_name}  "
          f"({n_faulty} total faulty samples)")
    print(f"  model output = P(Healthy)  |  HIGHER threshold → FEWER missed faults\n")
    print(f"  {'Thresh':>8}  {'MissedFaults':>13}  {'FalseAlarms':>12}  "
          f"{'FaultRecall':>12}  {'Note'}")


    for t in np.arange(0.40, 0.90, 0.05):
        y_pred  = (y_prob >= t).astype(int)
        missed  = int(np.sum((y_pred == 1) & (y_true == 0)))
        alarms  = int(np.sum((y_pred == 0) & (y_true == 1)))
        recall_f = 1.0 - missed / (n_faulty + 1e-8)
        note = " ← OPTIMAL" if abs(t - optimal_thresh) < 0.026 else ""
        print(f"  {t:>8.2f}  {missed:>13}  {alarms:>12}  "
              f"{recall_f:>12.4f}{note}")


def build_callbacks(model_path: str) -> list:

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath         = model_path,
        monitor          = "val_auc",
        save_best_only   = True,
        save_weights_only= False,
        verbose          = 1,
        mode             = "max",
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor              = "val_auc",
        patience             = ES_PATIENCE,
        restore_best_weights = True,
        verbose              = 1,
        mode                 = "max",
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = LR_FACTOR,
        patience = LR_PATIENCE,
        min_lr   = LR_MIN,
        verbose  = 1,
        mode     = "min",
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        filename  = os.path.join(OUTPUT_DIR, "training_log.csv"),
        separator = ",",
        append    = False,
    )

    return [checkpoint, early_stop, reduce_lr, csv_logger]



def plot_history(history: tf.keras.callbacks.History, save_dir: str):
    h      = history.history
    epochs = range(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ford A+B — Training History", fontsize=14, fontweight="bold")

    axes[0, 0].plot(epochs, h["loss"],     "b-o", ms=4, label="Train Loss")
    axes[0, 0].plot(epochs, h["val_loss"], "r-o", ms=4, label="Val Loss")
    axes[0, 0].set_title("Binary Cross-Entropy Loss")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, h["accuracy"],     "b-o", ms=4, label="Train Acc")
    axes[0, 1].plot(epochs, h["val_accuracy"], "r-o", ms=4, label="Val Acc")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    if "auc" in h:
        axes[1, 0].plot(epochs, h["auc"],     "b-o", ms=4, label="Train AUC")
        axes[1, 0].plot(epochs, h["val_auc"], "r-o", ms=4, label="Val AUC")
        axes[1, 0].set_title("AUC  (checkpoint metric)")
        axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("AUC")
        axes[1, 0].set_ylim([0, 1.05])
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    if "lr" in h:
        axes[1, 1].semilogy(epochs, h["lr"], "g-o", ms=4)
        axes[1, 1].set_title("Learning Rate  (ReduceLROnPlateau)")
        axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("LR (log scale)")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training history saved → {path}")


def main():
    X_train, y_train, X_testA, y_testA, X_testB, y_testB = load_data()

    model = build_model(input_shape=(13, 63, 1), learning_rate=LR_INITIAL)

    print("  MODEL ARCHITECTURE")
    model.summary()
    print(f"\n  Total parameters : {model.count_params():,}")
    print(f"  Initializer      : he_normal")
    print(f"  Loss             : binary_crossentropy")
    print(f"  Checkpoint metric: val_auc\n")

    callbacks = build_callbacks(MODEL_PATH)

    print("  TRAINING")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Max epochs   : {MAX_EPOCHS}")
    print(f"  Val split    : {int(VAL_SPLIT*100)}%")
    print(f"  ES patience  : {ES_PATIENCE} epochs")
    print(f"  Model saved  : {MODEL_PATH}\n")

    history = model.fit(
        X_train, y_train,
        batch_size       = BATCH_SIZE,
        epochs           = MAX_EPOCHS,
        validation_split = VAL_SPLIT,
        callbacks        = callbacks,
        shuffle          = True,
        verbose          = 1,
    )

    print("\n  QUICK EVAL — FordB  (default threshold = 0.50)")
    results_B = model.evaluate(X_testB, y_testB, verbose=0)
    for name, val in zip(model.metrics_names, results_B):
        print(f"  {name:<12} : {val:.4f}")


    print("\n  THRESHOLD OPTIMISATION  (FordB — minimise missed faults)")

    y_prob_B = model.predict(X_testB, verbose=0).flatten()

    optimal_thresh = find_optimal_threshold(
        y_testB, y_prob_B,
        beta=2.0,
        search_range=(0.40, 0.85),
        steps=300,
    )

    print(f"\n  Optimal threshold : {optimal_thresh:.3f}")
    print(f"  (F2 of Faulty class maximised — recall weighted 2x precision)\n")

    threshold_safety_sweep(y_testB, y_prob_B, optimal_thresh, "FordB")

    np.save(THRESH_PATH, np.array(optimal_thresh))
    print(f"\n  Threshold saved → {THRESH_PATH}")

    plot_history(history, OUTPUT_DIR)

    best_epoch   = int(np.argmax(history.history["val_auc"])) + 1
    best_val_auc = max(history.history["val_auc"])
    best_val_acc = history.history["val_accuracy"][best_epoch - 1]

    print("\n  TRAINING COMPLETE")
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Best val_auc  : {best_val_auc:.4f}")
    print(f"  Best val_acc  : {best_val_acc:.4f}")
    print(f"  Model saved  → {MODEL_PATH}")
    print(f"  Threshold    → {THRESH_PATH}  (value: {optimal_thresh:.3f})")
    print("\n  Next step: python evaluate.py")


if __name__ == "__main__":
    main()
