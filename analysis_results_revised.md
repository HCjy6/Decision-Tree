# Revised clinic appointment no-show analysis

## Data and design

- Source: `clinic_appointments.csv`; 400 rows and 8 columns.
- Total no-shows: 91 / 400 = 22.75%.
- Training: 280 appointments, 64 no-shows (22.86%).
- Test (held out): 120 appointments, 27 no-shows (22.50%).
- Split: `train_test_split(test_size=0.30, stratify=y, random_state=2026)`.
- The prior analysis compared settings on its original 30% test partition. This is a revised internal test on a different split of the same CSV. An independent validation requires new appointments.
- Positive class: no-show (1); negative class: checked-in (0).
- Baseline: always predict checked-in (hold).
- v1: one unweighted scikit-learn decision tree with max_depth=2, min_samples_leaf=28, and random_state=2026. Actual depth: 2.
- Candidate features: lead_time_days, prior_no_shows, age, distance_miles, sms_reminder, new_patient, appointment_hour.
- Features actually used in splits: sms_reminder, prior_no_shows.
- Root split: `sms_reminder` at 0.50.
- The model specification was fixed before scoring this split. Depth two limits rule length, and each leaf contains at least 10% of the training data.

## Test metrics

| Metric | Baseline | Decision tree v1 |
|---|---:|---:|
| TN | 93 | 82 |
| FP | 0 | 11 |
| FN | 27 | 12 |
| TP | 0 | 15 |
| Accuracy | 77.5% | 80.8% |
| Precision | 0.0% | 57.7% |
| Recall | 0.0% | 55.6% |
| Total test value | $-3,240.00 | $-2,100.00 |
| Value per appointment | $-27.00 | $-17.50 |

Baseline precision is undefined because the policy never predicts a no-show. The table and chart display 0.0% using the `zero_division=0` convention.
v1 accuracy was 85.7% on training appointments and 80.8% on test appointments.

## Costs and business decision

- TN (hold and checked-in) = $0; FP (overbook and checked-in) = -$60; FN (hold and no-show) = -$120; TP (overbook and no-show) = $0.
- Total value = -$60 × FP - $120 × FN.
- Baseline test value = -$60 × 0 - $120 × 27 = $-3,240.
- v1 test value = -$60 × 11 - $120 × 12 = $-2,100.
- Test-set improvement = $1,140 total / $9.50 per appointment.
- Annual volume assumption (not supplied by data): 10,000 appointments per year.
- Annualized model value if test sample rates generalize: baseline $-270,000; v1 $-175,000; potential improvement $95,000 per year.
- These are modeled loss reductions based on the provided simplified cost table, not verified realized profits.

## Upcoming appointments

Decision rule: `overbook` if the fitted classifier predicts no-show, otherwise `hold`.

- D: leaf node ID 3; predicted `no-show` → **overbook**. Input: {'lead_time_days': 21, 'prior_no_shows': 1, 'age': 27, 'distance_miles': 9.5, 'sms_reminder': 0, 'new_patient': 1, 'appointment_hour': 15}.
- E: leaf node ID 5; predicted `checked-in` → **hold**. Input: {'lead_time_days': 3, 'prior_no_shows': 0, 'age': 61, 'distance_miles': 2.0, 'sms_reminder': 1, 'new_patient': 0, 'appointment_hour': 9}.

## Tree rules (0 = checked-in, 1 = no-show)

```text
|--- sms_reminder <= 0.50
|   |--- prior_no_shows <= 0.50
|   |   |--- class: 0
|   |--- prior_no_shows >  0.50
|   |   |--- class: 1
|--- sms_reminder >  0.50
|   |--- prior_no_shows <= 1.50
|   |   |--- class: 0
|   |--- prior_no_shows >  1.50
|   |   |--- class: 0
```

## Interpretation and caveats

- The root uses `sms_reminder`. The branches in the tree rules above show when the other inputs affect the decision.
- These are correlations in this observational dataset, not proof that sending a reminder causes higher attendance.
- The sample is small and contains only 120 test appointments. A real pilot should validate the prediction and cost assumptions before broad rollout.
- The assignment assumes a second patient is available, the extra patient attends, and no further capacity costs arise beyond the given table.
- Slides must be written by the student per the assignment. Python-generated charts can be inserted into student-created slides.

## Chart files

- `outputs_revised/decision_tree.png`
- `outputs_revised/test_metrics.png`
- `outputs_revised/loss_per_appointment.png`
