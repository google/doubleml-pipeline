# Report Template and Statement Tagging

Use this structure for a full causal review of a `doubleml-pipeline` output
directory. Shorten it for targeted follow-up questions, while preserving strict
statement tagging everywhere.

## Table of Contents

-   [Three-Way Statement Tagging (Mandatory)](#three-way-statement-tagging-mandatory)
-   [Full Review Structure](#full-review-structure)
-   [Example Bullet Sequence](#example-bullet-sequence)

## Three-Way Statement Tagging (Mandatory)

Keep raw observations, model estimates, and explanatory mechanisms strictly
separate. Start **every bullet point** (and every result-stating sentence
outside bullets) with exactly **one** bold tag:

Tag | What belongs here | Source in `<output_dir>`
:-- | :---------------- | :-----------------------
**Fact:** | Raw values recorded in the input panel before causal modeling: spend, observed outcome KPI, row counts, active treatment share, treatment intensity/spend quantiles, bin cut points, covariate means. | `metadata.json` and raw input columns in `consolidated_1a_full_df.csv`.
**Inference:** | Causal quantities estimated by Double Machine Learning under model assumptions: `incremental_kpi`, 95% intervals, `effect_per_unit`, `roi_on_spend`, `roi_vs_threshold` verdicts, CATE evidence grades, sensitivity `robustness_value`, and model-implied optimization allocations. | DML estimate columns in `consolidated_1a_full_df.csv`, `cate_coef_dict_pooling.csv`, `consolidated_3a_detail.csv`, `sensitivity_metrics.csv`, `step0_saturation/`, and `007_optimization/`.
**Hypothesis:** | Proposed domain or behavioral mechanism explaining *why* an inference looks the way it does, paired with a concrete verification step (adding a column to `covariates_for_cate_list`, running a controlled holdout experiment, or collecting missing confounders). | Analyst reasoning anchored to facts and inferences.

Formatting rules:

-   **Never** rely on section headings to carry the tag; tag every individual
    bullet.
-   **One kind per bullet:** if a sentence mixes spend (Fact) and ROI
    (Inference), split it into two bullets.
-   **Tables:** include a `Statement` column (`Fact` / `Inference`) or label
    each metric column/row explicitly so raw facts and DML inferences are never
    conflated.
-   **Actions:** state the recommended action first, followed immediately by
    `**Inference:** ...` for the causal estimate supporting it and
    `**Hypothesis:** ...` if an unmodeled mechanism is assumed.

--------------------------------------------------------------------------------

## Full Review Structure

1.  **Executive Summary**
    -   Three concise bullets: headline incremental KPI and ROI verdict
        (**Inference:**), largest segment/timing/driver contrast
        (**Inference:**), and primary action with supporting tag.
2.  **Analysis Setup**
    -   **Fact:** bullets covering outcome KPI, panel dimensions (row count,
        sampling frequency, date period, `geo` and `item` counts), treatments
        (`type`, `spend_column`, `saturation_family`), confounders adjusted for,
        CATE covariates, Cluster DML keys, and configured `threshold_roi`.
3.  **Overall Treatment Impact**
    -   Per-treatment comparison separating historical spend and treated rows
        (**Fact:**) from `incremental_kpi` (with 95% uncertainty interval),
        `roi_on_spend` (or rescaled `effect_per_unit`), `share_of_outcome`,
        break-even gross margin (`1 / roi_on_spend` for revenue KPIs), and
        payback verdict (**Inference:**).
4.  **Where It Works (Entity Segments: `geo_col` & `item_col`)**
    -   Breakdown across each configured entity dimension (`geo_col` and
        `item_col`, sorted by `roi_on_spend` for efficiency questions or
        `incremental_kpi` for volume questions; omit dimensions that are `null`
        in `metadata.json`).
    -   Highlight `worth_investing` vs. `not_worth_investing` segments, flag
        `low_sample` (`< 30` treated rows) and `plausibility_flags`, and
        summarize raw covariate profiles (**Fact:**) for extreme or separated
        segments.
    -   If `geo_col` or `item_col` is not in `covariates_for_cate_list`, state
        explicitly that segment differences reflect covariate mix and treatment
        intensity rather than modeled segment-specific response.
5.  **When It Works (Calendar & Temporal Rhythm)**
    -   Breakdown by month-of-year (and day-of-week on daily panels, plus
        year-month or quarter trajectory) when `date_col` is present.
    -   Pair temporal/calendar ROI patterns (**Inference:**) with a testable
        domain **Hypothesis:**.
6.  **What Amplifies It (Drivers & Interaction Conditions)**
    -   Breakdown by within-entity driver quantiles and 2-D interaction tables
        (e.g., CATE covariate x treatment intensity, or two covariates from the
        run).
    -   Report bin cut points and spend per bin (**Fact:**), then ROI/effect per
        bin and CATE heterogeneity evidence (**Inference:**).
    -   Propose an activation mechanism (**Hypothesis:**) *only* when the driver
        is in `covariates_for_cate_list` with `strong` evidence; otherwise label
        the split descriptive and recommend adding the driver to
        `covariates_for_cate_list`.
7.  **Budget Optimization (When Available)**
    -   Summarize `007_optimization/` (overall, by `geo_col` / `item_col`, and
        yearly from `optimization_yearly_summary.csv`): historical vs. optimized
        spend, row counts shifted (`increased`, `reduced`, `cut_to_zero`),
        model-implied KPI change (**Inference:**), and optimizer assumptions and
        warnings.
8.  **Recommended Actions**
    -   Up to five concrete actions (`cut`, `keep`, `reallocate` via optimizer,
        or `validate via experiment`), each citing its supporting **Inference:**
        and any **Hypothesis:**.
9.  **Trust Assessment & Next Step**
    -   Report repetition stability (`share_positive`, CV), OVB sensitivity
        (`robustness_value` vs. `benchmark_covariate` and unadjusted concurrent
        levers), Phase 0 saturation diagnostics (when saturation ran), missing
        margin inputs (when `outcome_col` is gross revenue), and the single
        highest-value next pipeline run or holdout experiment.

--------------------------------------------------------------------------------

## Example Bullet Sequence

Replace `treatment_A`, `driver_X`, and entity labels with the actual names from
`metadata.json`:

> -   **Fact:** Entities in the top within-group quartile of `driver_X` received
>     active `treatment_A` on 1,370 unit-periods with total spend of 17.2M,
>     compared with 3.8M in the bottom quartile.
> -   **Inference:** The DML model estimates a `treatment_A` ROI on spend of
>     0.80 (95% interval 0.30 to 1.30) in top-quartile `driver_X` periods versus
>     0.17 in bottom-quartile periods.
> -   **Inference:** `driver_X` is a modeled CATE covariate with strong
>     heterogeneity evidence (`strong (p=0.003)`), so the model explicitly
>     estimates higher per-unit treatment response at higher `driver_X` levels.
> -   **Hypothesis:** Higher baseline `driver_X` amplifies conversion of
>     `treatment_A`; validate with a controlled holdout test comparing
>     `treatment_A` across high versus baseline `driver_X` segments.
