"""Shared helpers for the TrashBusters evaluation notebooks.

Config constants + pure functions so the two notebooks stay thin and consistent.
Everything here reads ONLY from logged wandb artifacts (run summaries, epoch
history, and the per-sample confusion CSVs) -- no re-training, no re-inference.

Hard-label metrics come from the confusion artifacts of the "core" runs. AUROC and
ECE need per-sample probabilities, which live in the separate "softmax" runs
produced by scripts/log_prob_metrics.py; see build_probability_metrics().
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# --------------------------------------------------------------------- config
ENTITY = "trash-busters"
PROJECT = "trash-busters"
RUN_TAG = "core"  # only pull runs with this tag; set None for all runs
PROB_TAG = "softmax"  # separate runs holding per-sample probability artifacts

CLASS_NAMES_FULL = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

NORMAL_SPLIT = "test"
SHIFT_SPLIT = "custom_test"
SPLITS = {
    NORMAL_SPLIT: "Normal test",
    SHIFT_SPLIT: "Custom (domain shift)",
}

ROBUST_METRICS = ["macro_f1", "balanced_acc", "mcc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "balanced_acc": "Balanced acc",
    "mcc": "MCC",
}
SPLIT_COLORS = {NORMAL_SPLIT: "#2b8cbe", SHIFT_SPLIT: "#e34a33"}

# Per-model plot colours. Paste the hex wandb shows for each run so the notebook
# plots match the wandb UI. Leave a model as None to fall back to a stable
# default colour (still consistent for that model across every plot).
MODEL_COLORS: dict = {
    "baseline": "#a1a9ad",
    "mlp_mixer": "#fb8332",
    "convnext_tiny": "#5656e6",
    "deit": "#ef3435",
}
_MODEL_ORDER = ["baseline", "mlp_mixer", "convnext_tiny", "deit"]

HISTORY_KEYS = ["epoch", "train/loss", "train/acc", "val/loss", "val/acc", "val/f1_macro"]


# ---------------------------------------------------------------- io / figures
def apply_style() -> None:
    """Consistent matplotlib/pandas display settings for both notebooks."""
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def make_fig_dir(path: str = "figures") -> Path:
    fig_dir = Path(path)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def savefig(fig, name: str, fig_dir: Path, save: bool = True) -> None:
    if save:
        out = fig_dir / f"{name}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")


# ------------------------------------------------------------------ wandb pull
def fetch_runs(entity: str = ENTITY, project: str = PROJECT, tag: str | None = RUN_TAG):
    """Return the list of runs to evaluate, optionally filtered by tag."""
    api = wandb.Api()
    filters = {"tags": {"$in": [tag]}} if tag else None
    runs = list(api.runs(f"{entity}/{project}", filters=filters))
    if not runs:
        raise RuntimeError(
            f"No runs found in {entity}/{project}"
            + (f" tagged '{tag}'." if tag else ".")
            + " Check auth (`wandb login`), entity/project, and the tag."
        )
    return runs


def describe_runs(runs) -> None:
    print(f"Found {len(runs)} run(s):")
    for r in runs:
        print(
            f"  - {r.name:30s} model={r.config.get('model','?'):14s} "
            f"classes={r.config.get('num_classes','?')} state={r.state} tags={r.tags}"
        )


def model_label(run) -> str:
    return run.config.get("model") or run.name


def run_class_names(run) -> list[str]:
    """Class names for a run, respecting a smaller (e.g. no-trash) head."""
    ncls = run.config.get("num_classes")
    if isinstance(ncls, int) and 0 < ncls <= len(CLASS_NAMES_FULL):
        return CLASS_NAMES_FULL[:ncls]
    return CLASS_NAMES_FULL


def load_confusion_frame(run, split: str):
    """DataFrame [true_label, predicted_label] for `split`, or None.

    Two sources, tried in order (both are functions of the same confusion counts,
    so every downstream hard-label metric is identical either way):

    1. The per-sample `{name}_{split}_confusion` CSV artifact from engine.test().
       Matched by type + split suffix, so it survives a run being renamed.
    2. Fallback: the `{split}/confusion_matrix_table` wandb Table logged by
       `wandb.plot.confusion_matrix`. Only aggregated cell counts are stored, so
       we expand them back into per-sample rows.
    """
    frame = _confusion_from_artifact(run, split)
    if frame is None:
        frame = _confusion_from_table(run, split)
    return frame


def _confusion_from_artifact(run, split: str):
    suffix = f"_{split}_confusion"
    for art in run.logged_artifacts():
        if art.type != "confusion_data":
            continue
        name = art.name.split(":")[0]
        if not name.endswith(suffix):
            continue
        # "_test_confusion" also matches "_custom_test_confusion"; disambiguate.
        if split == NORMAL_SPLIT and name.endswith(f"_{SHIFT_SPLIT}_confusion"):
            continue
        local = Path(art.download())
        csvs = sorted(local.glob("*.csv"))
        if csvs:
            df = pd.read_csv(csvs[0]).rename(columns=str.lower)
            if {"true_label", "predicted_label"}.issubset(df.columns):
                return df[["true_label", "predicted_label"]]
    return None


def _confusion_from_table(run, split: str):
    """Reconstruct per-sample rows from the `{split}/confusion_matrix_table`.

    The table (from wandb.plot.confusion_matrix) has columns
    ``Actual, Predicted, nPredictions`` with class *names*; we map names -> ids
    and repeat each (true, pred) pair by its count.
    """
    ref = run.summary.get(f"{split}/confusion_matrix_table")
    path = ref.get("path") if hasattr(ref, "get") else None
    if not path:
        return None

    tmp = Path(tempfile.mkdtemp(prefix="wandb_cm_"))
    try:
        run.file(path).download(root=str(tmp), replace=True)
        with open(tmp / path) as fh:
            tbl = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, notebook prints note
        print(f"note: could not load {split} confusion table for {run.name}: {exc}")
        return None

    cols = [c.lower() for c in tbl.get("columns", [])]
    data = tbl.get("data", [])
    idx = {name: i for i, name in enumerate(cols)}
    a = idx.get("actual", 0)
    p = idx.get("predicted", 1)
    n = idx.get("npredictions", 2)

    name_to_id = {name: i for i, name in enumerate(CLASS_NAMES_FULL)}
    trues, preds = [], []
    for row in data:
        ti, pi = name_to_id.get(row[a]), name_to_id.get(row[p])
        count = int(row[n])
        if ti is None or pi is None or count <= 0:
            continue
        trues.extend([ti] * count)
        preds.extend([pi] * count)

    if not trues:
        return None
    return pd.DataFrame({"true_label": trues, "predicted_label": preds})


def get_history(run, keys: list[str] = HISTORY_KEYS):
    """Per-epoch DataFrame for a run (partial rows dropped)."""
    try:
        h = run.history(keys=keys, pandas=True)
    except Exception:
        h = pd.DataFrame()
    if h.empty:
        h = pd.DataFrame(run.scan_history())
    if "epoch" in h.columns:
        h = h.dropna(subset=["epoch"]).sort_values("epoch")
    return h


# --------------------------------------------------------------------- metrics
def point_metrics(y_true, y_pred, labels) -> dict:
    """Hard-label metrics computable from argmax predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def build_metrics(runs):
    """Compute core metrics for every run x split from the confusion CSVs.

    Returns ``(metrics_long, confusion_frames)`` where ``confusion_frames`` maps
    ``(run.name, split) -> DataFrame | None`` for reuse by the class-level notebook.
    Falls back to logged scalars for accuracy/macro_f1 when a CSV is missing.
    """
    confusion_frames: dict = {}
    rows = []
    for run in runs:
        names = run_class_names(run)
        labels = list(range(len(names)))
        for split in SPLITS:
            df = load_confusion_frame(run, split)
            confusion_frames[(run.name, split)] = df
            row = {
                "run": run.name,
                "model": model_label(run),
                "split": split,
                "split_label": SPLITS[split],
            }
            if df is not None:
                row.update(point_metrics(df["true_label"], df["predicted_label"], labels))
                row["n"] = len(df)
            else:
                s = run.summary
                row["accuracy"] = s.get(f"{split}/acc", np.nan)
                row["macro_f1"] = s.get(f"{split}/f1_macro", np.nan)
                row["balanced_acc"] = np.nan
                row["mcc"] = np.nan
                row["n"] = np.nan
                print(f"note: no confusion CSV for {run.name}/{split}; used logged scalars")
            rows.append(row)
    return pd.DataFrame(rows), confusion_frames


def gap_table(metrics_long: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Generalization gap (normal - shift) per model for one metric."""
    piv = metrics_long.pivot_table(index="model", columns="split", values=metric)
    piv = piv.rename(columns=SPLITS)
    n, c = SPLITS[NORMAL_SPLIT], SPLITS[SHIFT_SPLIT]
    piv["gap"] = piv[n] - piv[c]
    piv["rel_%"] = 100 * piv["gap"] / piv[n].abs()
    return piv.sort_values("gap", ascending=False)


# ----------------------------------------------------------------- plot helper
def model_palette(models) -> list:
    """Colour list aligned to `models`.

    Uses MODEL_COLORS where set, else a stable per-model fallback -- so a given
    model gets the same colour in every plot, whether or not you've filled in the
    wandb hex values yet.
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colours = []
    for m in models:
        colour = MODEL_COLORS.get(m)
        if not colour:
            i = _MODEL_ORDER.index(m) if m in _MODEL_ORDER else abs(hash(m))
            colour = cycle[i % len(cycle)]
        colours.append(colour)
    return colours


def plot_confusion(cm, class_names, title, ax):
    """Row-normalised confusion heatmap (diagonal = per-class recall)."""
    cmn = cm.astype(float)
    rs = cmn.sum(axis=1, keepdims=True)
    cmn = np.divide(cmn, rs, out=np.zeros_like(cmn), where=rs != 0)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    ax.grid(False)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            ax.text(
                j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if cmn[i, j] > 0.5 else "black",
            )
    return im


# ------------------------------------------------- probability metrics (AUROC/ECE)
def fetch_softmax_runs(entity: str = ENTITY, project: str = PROJECT, tag: str = PROB_TAG):
    """Runs holding per-sample probability artifacts (from log_prob_metrics.py).

    Returns ``[]`` if none exist yet (unlike ``fetch_runs``, which raises), so the
    notebooks can simply skip the AUROC/ECE section until those runs are created.
    """
    api = wandb.Api()
    return list(api.runs(f"{entity}/{project}", filters={"tags": {"$in": [tag]}}))


def load_probability_frame(run, split: str):
    """Return ``(probs [N, C], labels [N])`` from the `{split}_probs` artifact.

    Matched by artifact type + split suffix (rename-proof, like the confusion
    loader). Returns None if the run has no probabilities for this split.
    """
    suffix = f"_{split}_probs"
    for art in run.logged_artifacts():
        if art.type != "probabilities":
            continue
        name = art.name.split(":")[0]
        if not name.endswith(suffix):
            continue
        # "_test_probs" also matches "_custom_test_probs"; disambiguate.
        if split == NORMAL_SPLIT and name.endswith(f"_{SHIFT_SPLIT}_probs"):
            continue
        csvs = sorted(Path(art.download()).glob("*.csv"))
        if csvs:
            df = pd.read_csv(csvs[0])
            prob_cols = [c for c in df.columns if c.startswith("p_")]
            return df[prob_cols].to_numpy(), df["true_label"].to_numpy()
    return None


def auroc_scores(probs, labels, class_names):
    """One-vs-rest AUROC per class + macro average.

    Classes absent from ``labels`` (possible on the imbalanced shift set) are
    undefined -> NaN; the macro is averaged over the rest.
    """
    labels = np.asarray(labels)
    n_classes = probs.shape[1]
    binarized = label_binarize(labels, classes=list(range(n_classes)))
    per_class = {}
    for c in range(n_classes):
        y_c = binarized[:, c]
        if y_c.min() == y_c.max():  # only one class present -> AUROC undefined
            per_class[class_names[c]] = float("nan")
        else:
            per_class[class_names[c]] = float(roc_auc_score(y_c, probs[:, c]))
    valid = [v for v in per_class.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, per_class


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins (lower = better calibrated)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if i == 0:
            in_bin |= confidence <= edges[0]
        if in_bin.sum():
            ece += (in_bin.sum() / n) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def build_probability_metrics(runs, n_bins: int = 15) -> pd.DataFrame:
    """AUROC (macro OvR + per-class) and ECE per softmax run x split.

    Long DataFrame, one row per run x split, carrying ``source_run_name`` so it
    merges back onto ``build_metrics()`` output by model / source run.
    """
    rows = []
    for run in runs:
        class_names = run_class_names(run)
        for split in SPLITS:
            row = {
                "run": run.name,
                "model": model_label(run),
                "source_run_name": run.config.get("source_run_name", run.name),
                "split": split,
                "split_label": SPLITS[split],
            }
            data = load_probability_frame(run, split)
            if data is not None:
                probs, labels = data
                macro, per_class = auroc_scores(probs, labels, class_names)
                row["auroc_macro_ovr"] = macro
                row["ece"] = expected_calibration_error(probs, labels, n_bins)
                for cls, value in per_class.items():
                    row[f"auroc_{cls}"] = value
            else:
                row["auroc_macro_ovr"] = np.nan
                row["ece"] = np.nan
                print(f"note: no {split} probability artifact for {run.name}")
            rows.append(row)
    return pd.DataFrame(rows)
