---
name: causal-insights
description: >-
  Interprets completed doubleml-pipeline output directories across treatments,
  outcome KPIs, entity dimensions (geo_col, item_col), calendar periods, and
  CATE covariates, turning Double Machine Learning estimates into causal insight
  reports, ROI or effect-per-unit verdicts, and resource allocation
  recommendations. Use when pointing at a doubleml-pipeline output_dir
  (containing metadata.json and step4_consolidated_results/) or when asked
  which treatments paid back, where or when effects are strongest, what drives
  effect heterogeneity, whether estimates are trustworthy, or how to reallocate
  budget. Don't use for fitting new DoubleML models, running raw regressions on
  input CSVs, or forecasting non-causal time series.
---

# Causal Insight Analysis for `doubleml-pipeline` Outputs

The pipeline has already estimated debiased causal effects with Double Machine
Learning (DML). Read the completed `output_dir` files directly using the exact
metric definitions in
[references/metric_formulas.md](references/metric_formulas.md) — never refit raw
regressions on the input panel.

**Adapt to the run's schema (`metadata.json` first):** Treatments, channels,
interventions, outcome KPIs, entity dimensions (`geo_col`, `item_col`), and
covariates vary across users and domains (including non-monetary treatments with
no spend column). Derive every lever name, unit type, segment label, and driver
strictly from `metadata.json` and the columns present in
`consolidated_1a_full_df.csv` — never assume specific channels, treatments, or
domain constructs that are not in the data. When a question cannot be answered
from the run's outputs, say so explicitly and name the exact pipeline setting
(`treatment_cols`, `x_cols_list`, `covariates_for_cate_list`, `geo_col` /
`item_col`, `optimize=True`, or `run_sensitivity=True`) needed to answer it.

## Output Directory Reading Order

Step | File in `<output_dir>` | What to extract
:--- | :--------------------- | :--------------
1 | `metadata.json` | `outcome_col`, `treatment_cols`, `treatment_types`, `treatment_spend_cols`, `x_cols_list`, `covariates_for_cate_list`, `geo_col`, `item_col`, `date_col`, `opt_threshold_roi` (`threshold_roi`), `saturation_family`, `saturation_phase0`, `phase4_optimization`, `cluster_dml`.
2 | `step4_consolidated_results/consolidated_1a_full_df.csv` | Row-level panel + `estimated_incremental_KPI_<d>`, `_2.5%`, `_97.5%`, `roi_amount_based_<d>`, and (saturation optimizer only) `optimized_<d>_spend` / `optimized_incremental_KPI_<d>`. Aggregate all overall, segment, calendar, and driver slices from this table.
3 | `step4_consolidated_results/graph/004_cate/cate_coef_dict_pooling.csv` | Pooled `model == "median"` row `cate_pvalue_dict` for Bonferroni-adjusted heterogeneity grades (`strong`, `weak`, `none`).
4 | `step4_consolidated_results/consolidated_3a_detail.csv` | Phase 3 repetition stability (`total_estimated_incremental_KPI` sign consistency and CV across `n_reps`).
5 | `step4_consolidated_results/graph/006_sensitivity/sensitivity_metrics.csv` | OVB `robustness_value` vs. `benchmark_covariate` (`benchmark_cf_d`, `benchmark_cf_y`).
6 | `step0_saturation/summary_alpha_selection.json` & `diag6_t_ref_consistency.csv` | Phase 0 `decision` (`saturation` vs. `linear`), `decision_reason` (`1se_pass`, `1se_fail`, `grid_edge_upper`), `selected_family`, `alpha_hat`, `half_saturation_T`, `T_ref`, `d_bar`, `se`, `diag1_fired`, `diag2_fired`, `r_T`, `messages`. (Per-treatment subdirs when `len(treatment_cols) > 1`.)
7 | `step4_consolidated_results/graph/007_optimization/` or `roi_output/<d>/007_optimization/` | `optimization_metrics.csv` and `optimization_yearly_summary.csv` (`spend` treatments use `graph/007_optimization/`; non-`spend` treatments with spend mapping use `roi_output/<d>/007_optimization/`).

--------------------------------------------------------------------------------

## Four Non-Negotiable Calculation Rules

Always apply the formulas in
[references/metric_formulas.md](references/metric_formulas.md) to avoid silent
calculation errors:

1.  **Order row-level 95% bounds before summing:**
    `estimated_incremental_KPI_<d>_2.5%` and `_97.5%` are `(T - zero_point) *
    bound`, so they swap order whenever `T < zero_point`. Always compute
    `np.minimum(c25, c975).sum()` and `np.maximum(c25, c975).sum()`. Compute
    `effect_per_unit` (and its ordered 95% interval) by dividing by `T.sum()`,
    leaving it unset if a slice mixes positive and negative treatment values.
2.  **Divide by monetary spend for ROI (`treatment_spend_cols`):** In
    `consolidated_3b_summary*.csv`, `total_estimated_roi` is `incremental_kpi /
    total_input` (effect per raw unit of treatment `<d>`, NOT monetary ROI when
    `treatment_types[d]` is `percentage`, `price`, `impressions`, or another
    non-spend unit). Always compute monetary ROI from
    `consolidated_1a_full_df.csv` by dividing `incremental_kpi` (and its ordered
    95% bounds) by `df[treatment_spend_cols[d]].sum()` when spend is mapped
    (otherwise report `effect_per_unit` and its 95% interval). Compare
    `[roi_low, roi_high]` against `opt_threshold_roi`: `roi_high <
    threshold_roi` -> `not_worth_investing`; `roi_low > threshold_roi` ->
    `worth_investing`; straddling -> `undetermined` (omit verdict if
    `opt_threshold_roi` is `null` or no spend column exists).
3.  **Grade CATE heterogeneity from the `median` row with Bonferroni scaling:**
    In `cate_coef_dict_pooling.csv` (`model == "median"`), match all basis
    p-values for covariate `c` (`<c>_pvalue`, `bs(<c>, ...)`, `C(<c>)...`),
    compute `p_adj = min(1.0, min(ps) * len(ps))`, and classify `< 0.05` as
    `strong (p=...)`, `< 0.10` as `weak (p=...)`, and `>= 0.10` as `none
    (p=...)` (quoting `p<0.001` below `0.001`). Pooled BLP p-values ignore
    first-stage estimation error, so treat them as an approximate grade rather
    than a formal hypothesis test.
4.  **Exclude synthetic ground truth and derived cluster columns:** Never read,
    profile, or report `TrueContrib_*`, `total_ground_truth_KPI`,
    `true_incremental_KPI_*`, `ground_truth_*`, `true_total_roi`, `SMAPE`,
    `R_squared`, `MAE`, `RMSE`, or `metadata["ground_truth_metrics"]` unless the
    user explicitly asks to audit synthetic recovery. Exclude
    `cluster_dml.derived_columns` (`cluster_date_block`) from driver analyses.

**Allowed manual arithmetic & multi-run rule:** Break-even gross margin `1 /
roi_on_spend` when `outcome_col` represents gross revenue, and rescaling
`effect_per_unit` to natural units (e.g., dividing by `100` when
`treatment_types[d] == "percentage"` stored on `[0, 1]` to report per 1
percentage point). Never invent margins or benchmarks (ask the user instead),
and never extrapolate outside observed support. Across multiple `output_dir`
runs estimated separately, compare side by side only when `outcome_col` matches,
and **never** sum `incremental_kpi` across runs. Speak in plain domain terms
matching the user's column semantics and language without leaking internal
schema keys.

--------------------------------------------------------------------------------

## 7-Step Analysis Playbook

Work through the steps needed to answer the user's question. For a comprehensive
review, execute all applicable steps and follow
[references/report_template.md](references/report_template.md).

1.  **Frame the run:** Read `metadata.json` and inspect date frequency (`daily`,
    `weekly`, `monthly`), row count, active entity axes (`geo_col`, `item_col`),
    treatments (`treatment_cols`, `treatment_types`, `treatment_spend_cols`),
    confounders (`x_cols_list`), `covariates_for_cate_list`,
    `opt_threshold_roi`, `cluster_dml`, and Phase 0 / Phase 4 flags.
2.  **Headline causal impact:** Summarize total `spend` (or treatment volume)
    and treated row count (**Fact:**), then `incremental_kpi` with 95% interval,
    `roi_on_spend` (or rescaled `effect_per_unit`), `share_of_outcome`,
    break-even margin (if revenue KPI), and payback verdict (**Inference:**).
3.  **Where it works (`geo_col` & `item_col`):** Group
    `consolidated_1a_full_df.csv` by each configured entity axis (`geo_col`,
    `item_col`). Sort by `roi_on_spend` for efficiency questions or
    `incremental_kpi` for volume questions. Check whether `geo_col` / `item_col`
    is in `covariates_for_cate_list`.
4.  **When it works (calendar cadence):** When `date_col` is present, group by
    month-of-year (and day-of-week when median date step <= 2 days, plus
    year-month or quarter for trajectory).
5.  **What amplifies it (drivers & 2-D interactions):** Bin continuous drivers
    with within-entity quantiles (`within_quantile_bins` in
    [references/metric_formulas.md](references/metric_formulas.md), grouping by
    `geo_col` or `item_col` when present) so each unit is judged against its own
    baseline, and cross-tabulate 2-D interactions (e.g., a CATE covariate x
    treatment intensity, or two covariates from the run).
6.  **Can we trust it? (Diagnostics & raw segment profiles):** Check repetition
    stability, OVB sensitivity (`robustness_value` vs. `max(benchmark_cf_d,
    benchmark_cf_y)` overall and least-robust segments), Phase 0 saturation
    diagnostics, `low_sample` (`< 30` treated rows), and `plausibility_flags`.
    Whenever two segments have non-overlapping 95% intervals (`ci_separation` on
    `roi_on_spend` or `effect_per_unit`) or `plausibility_flags` fire, profile
    that segment's raw data (`gap_in_row_sd`, `ratio_to_overall`,
    `treated_share`, and treated `p25/p50/p75`) before interpreting.
7.  **How should allocation move? (Optimization):** When `007_optimization/`
    exists, report historical vs. optimized spend, incremental KPI, and ROI
    (total, by `geo_col` / `item_col`, and yearly from
    `optimization_yearly_summary.csv`), row counts shifted (`increased`,
    `reduced`, `cut_to_zero`), reconciliation with `optimization_metrics.csv`,
    and optimizer assumptions/warnings. Never design custom allocations, scale
    to unmodeled budgets, or shift budget across treatments or runs. If absent,
    note that optimization requires a single-treatment run with `optimize=True`,
    `date_col`, and spend mapping.

--------------------------------------------------------------------------------

## Causal & Domain Interpretation Rules

-   **Reading 95% intervals:** Always lead with the point estimate and
    `effect_direction` (`positive`, `negative`, `zero`). Aggregate 95% bands sum
    row-level bounds conservatively and routinely span zero. **Never** call an
    estimate "statistically significant", "not significant", "inconclusive", or
    "indistinguishable from zero", and never use the words *significant* or
    *p-value* in prose (except quoting CATE grades verbatim like `strong
    (p=0.0022)` as an approximate heterogeneity grade).
-   **Average vs. Marginal ROI:** `roi_on_spend` is the *average* return on past
    spend. `worth_investing` justifies **keeping** historical spend;
    `not_worth_investing` justifies **cutting/stopping**; `undetermined` calls
    for a controlled test. Scaling budget *up* requires marginal returns from an
    active saturation curve or KKT `optimization`. Under saturation, per-unit
    effect naturally falls at higher spend along diminishing returns.
-   **Column Role (`cate_covariate` | `confounder` | `unused`) & Constant-Effect
    Runs:** Per-unit treatment effects vary **only** through
    `covariates_for_cate_list`. If `covariates_for_cate_list` is empty, per-unit
    effect is constant by construction and `incremental_kpi` differs across
    slices only with treatment volume. If a slice column is `confounder` (in
    `x_cols_list` only, including entity dummies) or `unused`, splitting by it
    is **descriptive** — showing where effects landed via covariate mix and
    intensity, not modeled slice-specific response. Say so explicitly and
    recommend adding the column to `covariates_for_cate_list` to test it. Warn
    against conditioning on post-treatment variables (columns moved by the
    treatment itself) as targeting pre-conditions.
-   **Confounding, Carryover & Spillover:** Inspect `x_cols_list` against the
    other columns in `consolidated_1a_full_df.csv` to flag unadjusted concurrent
    treatments or levers, intertemporal pull-forward or lagged carryover
    effects (`_l<k>` lagged columns), and cross-unit spillover or substitution
    across `geo_col` / `item_col`.

--------------------------------------------------------------------------------

## Trust Checklist

Check | Required Action
:---- | :--------------
**Payback & multiplicity** | Keep `worth_investing`, cut `not_worth_investing`, test `undetermined`. Across many `geo_col` or `item_col` slices, ~2.5% per tail can cross `threshold_roi` by chance unless consistent across periods and repetitions.
**Driver evidence** | Claim a covariate shifts per-unit response only when CATE grade is `strong` (quote verbatim, e.g., `strong (p=0.0022)`). Treat `weak` as a lead to validate and `none` as no modeled heterogeneity.
**Plausibility & support** | Flag `< 30` treated rows (`low_sample`) as directional only. If `incremental_kpi_exceeds_outcome`, `kpi_without_spend`, or `negative_spend` fires, treat that slice's estimate as unreliable and inspect raw segment profiles.
**Model & Phase 0** | Caveat high repetition CV (`> 0.3`), mixed repetition signs, `robustness_value <= max(benchmark_cf_d, benchmark_cf_y)` (passing rules out confounders only up to the strongest observed `x_col`), few clusters (`< 30` in `cluster_dml.messages`), or Phase 0 warnings (`1se_fail`, `diag1_fired`, `diag2_fired`, low `r_T`, `fallback_half_saturation_T_used`).
**Optimizer discipline** | Frame `optimization` strictly as a retrospective model-implied proposal (no carryover/adstock, point-estimate CATE, no 95% interval). State whether Phase 4 ran `saturation`, `simple`, or `linear_fallback`, warn if `kpi_after > 2.0 * kpi_before` (upper bound from cutting low-scored rows), if `kpi_before <= 0 < kpi_after`, or if `> 30%` of active units are cut to zero, and ask about operational floors/contracts.

--------------------------------------------------------------------------------

## Mandatory Statement Tagging (`Fact:` / `Inference:` / `Hypothesis:`)

Start **every bullet point** (and every standalone result sentence) with
**exactly one** bold tag — never rely on section headers, and split mixed
bullets:

-   **Fact:** Raw input panel quantities (`spend`, observed KPI, row/treated
    counts, treatment intensity quantiles, bin cut points, covariate means).
-   **Inference:** DML model outputs (`incremental_kpi`, 95% intervals,
    `roi_on_spend`, payback verdicts, CATE evidence grades, sensitivity `RV`,
    model-implied optimization numbers).
-   **Hypothesis:** Proposed mechanism explaining *why* an inference occurred,
    always paired with a concrete verification step (config change or holdout
    experiment).

## Bundled References

-   Exact aggregation, ROI, CATE grading, diagnostics, and optimization
    formulas: [references/metric_formulas.md](references/metric_formulas.md)
-   Full causal review structure and statement-tagging examples:
    [references/report_template.md](references/report_template.md)
