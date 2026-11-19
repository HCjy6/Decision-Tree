"""Reproducible clinic no-show decision-tree assignment analysis.

Run from Cursor: python decision_tree_analysis_revised.py
Input: clinic_appointments.csv beside this script.
Outputs: analysis_results_revised.md plus PNG charts in outputs_revised/.

This script fits one fixed model on the training partition, then scores the test partition.
An earlier exploratory run reused the original test split for model comparison.
This revised split of the same CSV is an internal evaluation.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "clinic_appointments.csv"
OUTPUT_DIR = ROOT / "outputs_revised"
OUTPUT_DIR.mkdir(exist_ok=True)
FEATURES = [
    "lead_time_days", "prior_no_shows", "age", "distance_miles",
    "sms_reminder", "new_patient", "appointment_hour",
]
TARGET = "no_show"
SEED = 2026
TEST_FRACTION = 0.30
MAX_DEPTH = 2
MIN_LEAF_SIZE = 28
ANNUAL_VOLUME = 10_000  # Hypothetical volume; not provided in the dataset.


def decision_for_appointment(model, appointment):
    """Turn the prediction for a one-row appointment frame into a clinic action."""
    prediction = int(model.predict(appointment)[0])
    return "overbook" if prediction == 1 else "hold"


def summarize(y_true, y_pred):
    tn, fp, fn, tp = map(int, confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    total = -60 * fp - 120 * fn  # given assignment cost table
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Baseline predicts no positives; use 0.0 convention for undefined precision.
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "total_value": total,
        "per_appointment": total / len(y_true),
    }


def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            "Missing clinic_appointments.csv. Put it in the same folder as this script."
        )
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    if list(df.columns) != required:
        raise ValueError(f"Unexpected CSV columns. Expected: {required}; got: {list(df.columns)}")
    if df[required].isnull().any().any():
        raise ValueError("Dataset has missing values; inspect before fitting.")
    if set(df[TARGET].unique()) != {"no-show", "checked-in"}:
        raise ValueError("Outcome labels are not exactly 'no-show' and 'checked-in'.")
    X = df[FEATURES]
    y = df[TARGET].eq("no-show").astype(int)

    # Hold out 30% while maintaining approximately equal no-show prevalence.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=y
    )

    # Two splits per path keep the rules short. A leaf must contain at least
    # 28 appointments, one tenth of the 280 training rows. These settings are
    # fixed before fitting and do not depend on the revised test scores.
    model = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF_SIZE, random_state=SEED
    )
    model.fit(X_train, y_train)

    baseline_class = int(y_train.mode().iloc[0])  # 0 (checked-in), majority class
    baseline_predictions = np.full(len(y_test), baseline_class, dtype=int)
    v1_predictions = model.predict(X_test)
    baseline = summarize(y_test, baseline_predictions)
    v1 = summarize(y_test, v1_predictions)
    train_accuracy = float(model.score(X_train, y_train))
    improvement_total = v1["total_value"] - baseline["total_value"]
    improvement_per_appointment = v1["per_appointment"] - baseline["per_appointment"]
    annual_improvement = improvement_per_appointment * ANNUAL_VOLUME
    annual_baseline = baseline["per_appointment"] * ANNUAL_VOLUME
    annual_v1 = v1["per_appointment"] * ANNUAL_VOLUME

    upcoming = {
        "D": {
            "lead_time_days": 21, "prior_no_shows": 1, "age": 27,
            "distance_miles": 9.5, "sms_reminder": 0, "new_patient": 1,
            "appointment_hour": 15,
        },
        "E": {
            "lead_time_days": 3, "prior_no_shows": 0, "age": 61,
            "distance_miles": 2.0, "sms_reminder": 1, "new_patient": 0,
            "appointment_hour": 9,
        },
    }
    decisions = {}
    for key, appointment in upcoming.items():
        frame = pd.DataFrame([appointment], columns=FEATURES)
        action = decision_for_appointment(model, frame)
        leaf_id = int(model.apply(frame)[0])
        decisions[key] = (action, leaf_id)

    # Figure 1: allowed Python-generated tree plot for user-created slides.
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    plot_tree(
        model, feature_names=FEATURES,
        class_names=["checked-in", "no-show"], filled=False,
        rounded=True, fontsize=10, proportion=False, ax=ax
    )
    ax.set_title(f"Clinic decision tree (v1) · {len(y_train)} training appointments", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "decision_tree.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: evaluation metrics; no custom color scheme.
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bw = 0.35
    b1 = ax.bar(x - bw/2, [baseline[m] * 100 for m in ("accuracy", "precision", "recall")], bw, label="Current policy")
    b2 = ax.bar(x + bw/2, [v1[m] * 100 for m in ("accuracy", "precision", "recall")], bw, label="Decision tree v1")
    ax.set_xticks(x, ["Accuracy", "No-show precision", "No-show recall"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Revised test performance ({len(y_test)} appointments)")
    ax.legend(loc="upper right")
    ax.bar_label(b1, fmt="%.1f%%", padding=3)
    ax.bar_label(b2, fmt="%.1f%%", padding=3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: direct modeled loss comparison.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(["Current policy", "Decision tree v1"],
                  [-baseline["per_appointment"], -v1["per_appointment"]])
    ax.bar_label(bars, labels=[f"${-baseline['per_appointment']:,.2f}", f"${-v1['per_appointment']:,.2f}"], padding=4)
    ax.set_ylabel("Average loss per appointment (US$)")
    ax.set_title("Test-set estimated cost: lower is better")
    ax.set_ylim(0, 32)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loss_per_appointment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    actual_splits = [FEATURES[int(i)] for i in model.tree_.feature if i >= 0]
    unique_splits = list(dict.fromkeys(actual_splits))
    txt_tree = export_text(model, feature_names=FEATURES)

    lines = [
        "# Revised clinic appointment no-show analysis", "",
        "## Data and design", "",
        f"- Source: `clinic_appointments.csv`; {len(df)} rows and {len(df.columns)} columns.",
        f"- Total no-shows: {int(y.sum())} / {len(y)} = {y.mean():.2%}.",
        f"- Training: {len(y_train)} appointments, {int(y_train.sum())} no-shows ({y_train.mean():.2%}).",
        f"- Test (held out): {len(y_test)} appointments, {int(y_test.sum())} no-shows ({y_test.mean():.2%}).",
        f"- Split: `train_test_split(test_size={TEST_FRACTION:.2f}, stratify=y, random_state={SEED})`.",
        "- The prior analysis compared settings on its original 30% test partition. This is a revised internal test on a different split of the same CSV. An independent validation requires new appointments.",
        "- Positive class: no-show (1); negative class: checked-in (0).",
        "- Baseline: always predict checked-in (hold).",
        f"- v1: one unweighted scikit-learn decision tree with max_depth={MAX_DEPTH}, min_samples_leaf={MIN_LEAF_SIZE}, and random_state={SEED}. Actual depth: {model.get_depth()}.",
        f"- Candidate features: {', '.join(FEATURES)}.",
        f"- Features actually used in splits: {', '.join(unique_splits)}.",
        f"- Root split: `{FEATURES[model.tree_.feature[0]]}` at {model.tree_.threshold[0]:.2f}.",
        "- The model specification was fixed before scoring this split. Depth two limits rule length, and each leaf contains at least 10% of the training data.", "",
        "## Test metrics", "",
        "| Metric | Baseline | Decision tree v1 |", "|---|---:|---:|",
    ]
    for m in ("TN", "FP", "FN", "TP"):
        lines.append(f"| {m} | {baseline[m]} | {v1[m]} |")
    for m in ("accuracy", "precision", "recall"):
        lines.append(f"| {m.title()} | {baseline[m]:.1%} | {v1[m]:.1%} |")
    lines += [
        f"| Total test value | ${baseline['total_value']:,.2f} | ${v1['total_value']:,.2f} |",
        f"| Value per appointment | ${baseline['per_appointment']:,.2f} | ${v1['per_appointment']:,.2f} |",
        "",
        "Baseline precision is undefined because the policy never predicts a no-show. The table and chart display 0.0% using the `zero_division=0` convention.",
        f"v1 accuracy was {train_accuracy:.1%} on training appointments and {v1['accuracy']:.1%} on test appointments.", "",
        "## Costs and business decision", "",
        "- TN (hold and checked-in) = $0; FP (overbook and checked-in) = -$60; FN (hold and no-show) = -$120; TP (overbook and no-show) = $0.",
        "- Total value = -$60 × FP - $120 × FN.",
        f"- Baseline test value = -$60 × {baseline['FP']} - $120 × {baseline['FN']} = ${baseline['total_value']:,.0f}.",
        f"- v1 test value = -$60 × {v1['FP']} - $120 × {v1['FN']} = ${v1['total_value']:,.0f}.",
        f"- Test-set improvement = ${improvement_total:,.0f} total / ${improvement_per_appointment:,.2f} per appointment.",
        f"- Annual volume assumption (not supplied by data): {ANNUAL_VOLUME:,} appointments per year.",
        f"- Annualized model value if test sample rates generalize: baseline ${annual_baseline:,.0f}; v1 ${annual_v1:,.0f}; potential improvement ${annual_improvement:,.0f} per year.",
        "- These are modeled loss reductions based on the provided simplified cost table, not verified realized profits.", "",
        "## Upcoming appointments", "",
        "Decision rule: `overbook` if the fitted classifier predicts no-show, otherwise `hold`.", "",
    ]
    for k,(action,leaf) in decisions.items():
        pred = "no-show" if action=="overbook" else "checked-in"
        lines.append(f"- {k}: leaf node ID {leaf}; predicted `{pred}` → **{action}**. Input: {upcoming[k]}.")
    lines += [
        "", "## Tree rules (0 = checked-in, 1 = no-show)", "", "```text", txt_tree.rstrip(), "```", "",
        "## Interpretation and caveats", "",
        f"- The root uses `{FEATURES[model.tree_.feature[0]]}`. The branches in the tree rules above show when the other inputs affect the decision.",
        "- These are correlations in this observational dataset, not proof that sending a reminder causes higher attendance.",
        "- The sample is small and contains only 120 test appointments. A real pilot should validate the prediction and cost assumptions before broad rollout.",
        "- The assignment assumes a second patient is available, the extra patient attends, and no further capacity costs arise beyond the given table.",
        "- Slides must be written by the student per the assignment. Python-generated charts can be inserted into student-created slides.",
        "", "## Chart files", "", "- `outputs_revised/decision_tree.png`", "- `outputs_revised/test_metrics.png`", "- `outputs_revised/loss_per_appointment.png`", "",
    ]
    out = ROOT / "analysis_results_revised.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:43]))
    print(f"\nD/E decisions: {decisions}")
    print(f"\nSaved {out} and PNG charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
