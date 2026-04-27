"""
=============================================================
  plot_training_graphs.py
  Standalone training graph plotter — works for any model
  whose training log was saved as a CSV via CSVLogger.

  Usage:
      python plot_training_graphs.py \
          --csv  training_log.csv \
          --title "MobileNetV2 – Maize Disease Classification" \
          --out   training_history.png \
          --phase2_epoch 20          # optional: draws fine-tuning line

  If you did NOT use CSVLogger, you can also call save_graphs()
  directly from your training script (see bottom of this file).
=============================================================
"""

import argparse
import os
import csv
import matplotlib
matplotlib.use("Agg")           # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────
# CORE PLOTTING FUNCTION
# ─────────────────────────────────────────────────────────────
def save_graphs(
    acc,
    val_acc,
    loss,
    val_loss,
    title="Model Training History",
    out_path="training_history.png",
    phase2_start_epoch=None,   # int: first epoch of fine-tuning phase
    dpi=150,
):
    """
    Saves accuracy + loss curves to *out_path*.

    Parameters
    ----------
    acc, val_acc, loss, val_loss : list[float]
        Per-epoch metric values (combined across phases if two-phase training).
    title : str
        Figure super-title.
    out_path : str
        File path for the saved PNG.
    phase2_start_epoch : int or None
        If given, draws a dashed vertical line marking the start of fine-tuning.
    dpi : int
        Resolution of saved figure.
    """
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Accuracy ──────────────────────────────────────────────
    ax1.plot(epochs_range, acc,     label="Train Accuracy",      color="steelblue",   linewidth=2)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="darkorange",  linewidth=2)
    if phase2_start_epoch is not None:
        ax1.axvline(
            x=phase2_start_epoch,
            color="red", linestyle="--", alpha=0.7,
            label="Fine-tuning start"
        )
    ax1.set_title("Accuracy", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Annotate best val accuracy
    best_val = max(val_acc)
    best_ep  = val_acc.index(best_val) + 1
    ax1.annotate(
        f"Best: {best_val*100:.1f}%\n(epoch {best_ep})",
        xy=(best_ep, best_val),
        xytext=(best_ep + max(1, len(acc) // 10), best_val - 0.08),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=9,
        color="black",
    )

    # ── Loss ──────────────────────────────────────────────────
    ax2.plot(epochs_range, loss,     label="Train Loss",      color="steelblue",  linewidth=2)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", color="darkorange", linewidth=2)
    if phase2_start_epoch is not None:
        ax2.axvline(
            x=phase2_start_epoch,
            color="red", linestyle="--", alpha=0.7,
            label="Fine-tuning start"
        )
    ax2.set_title("Loss", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Training curves saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# CSV READER  (for standalone CLI use)
# ─────────────────────────────────────────────────────────────
def load_csv_log(csv_path):
    """Read a Keras CSVLogger file and return metric lists."""
    acc, val_acc, loss, val_loss = [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc.append(float(row["accuracy"]))
            val_acc.append(float(row["val_accuracy"]))
            loss.append(float(row["loss"]))
            val_loss.append(float(row["val_loss"]))
    return acc, val_acc, loss, val_loss


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training accuracy & loss from a Keras CSVLogger file."
    )
    parser.add_argument("--csv",           required=True,  help="Path to training_log.csv")
    parser.add_argument("--title",         default="Model Training History",
                        help="Figure title")
    parser.add_argument("--out",           default="training_history.png",
                        help="Output PNG path")
    parser.add_argument("--phase2_epoch",  type=int, default=None,
                        help="Epoch number where fine-tuning began (draws dashed line)")
    parser.add_argument("--dpi",           type=int, default=150)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    acc, val_acc, loss, val_loss = load_csv_log(args.csv)
    print(f"  Loaded {len(acc)} epochs from {args.csv}")

    save_graphs(
        acc=acc,
        val_acc=val_acc,
        loss=loss,
        val_loss=val_loss,
        title=args.title,
        out_path=args.out,
        phase2_start_epoch=args.phase2_epoch,
        dpi=args.dpi,
    )