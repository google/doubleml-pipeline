# Report template

Use this structure for a full review of a run. Shorten it for quick
follow-up questions, but keep the three kinds of statements apart everywhere.

Kind           | Meaning | Source
-------------- | ------- | ------
**Fact**       | What the input data recorded: spend, sales, how often and how deeply a segment was treated, covariate values. | `get_segment_profile`; the raw counts, spend and outcome totals in any tool.
**Inference**  | What the DML model estimated from those facts: incremental KPI, ROI and its interval, `roi_vs_threshold` verdicts, CATE evidence. Valid only under the model's assumptions, so it travels with its reliability signals. | `get_treatment_effects`, `get_effects_by`, `get_model_diagnostics`, `get_budget_optimization`.
**Hypothesis** | Your explanation of why an inference looks the way it does, with the test that would verify it (a covariate to add and re-run, a holdout, data to collect). | Your reasoning.

Never present an inference as a fact, or a hypothesis as either.

1.  **Result summary.** Three lines: the headline ROI with its verdict
    (inference), the biggest where/when difference (inference), and the
    recommended action.
2.  **Analysis setup.** Outcome, treatment(s) with spend column, confounders
    adjusted for, CATE covariates, period, data frequency, rows, stores and
    products (facts). State the ROI threshold.
3.  **Overall.** Spend (fact); incremental KPI with its interval, ROI and the
    `roi_vs_threshold` verdict (inference), per treatment.
4.  **Where it works.** Table of stores or regions (then products) with spend
    (fact), ROI, interval and verdict (inference). Name the segments that are
    worth and not worth investing in. Flag `low_sample` segments and any
    `plausibility_flags`.
    *Hypothesis:* why those segments differ, backed by the facts from
    `get_segment_profile` where the gap is large.
5.  **When it works.** Month-of-year table (and day-of-week for daily data)
    with spend (fact), ROI and interval (inference).
    *Hypothesis:* the seasonal or weekly reason.
6.  **What amplifies it.** ROI by driver quartile (within market), and the
    driver x depth table when relevant. The bin ranges and spend per bin are
    facts; the ROI per bin and whether the driver is a modeled CATE covariate
    with strong evidence are inferences. If the driver is not a CATE
    covariate, say that the split is descriptive and skip the mechanism.
    *Hypothesis:* only for a modeled driver with strong evidence, the
    mechanism (for example media creates demand that the promotion converts)
    and the test.
7.  **Recommended actions.** Up to five, each with the segment or period, the
    direction, the supporting inference and its verdict. Only
    `not_worth_investing` / `worth_investing` verdicts justify stopping or
    scaling spend; everything else becomes a test.
8.  **Confidence and caveats.** Repetition stability, sensitivity to
    confounding, low-support results, plausibility flags, missing margins and
    confounders that were not adjusted for.
9.  **Next step.** The analysis or test that would most reduce the remaining
    uncertainty.

Example of the split:

> **Fact:** In the top media-spend quartile of their own market, stores ran
> promotions on 1,370 days and spent $17.2M; in the bottom quartile, $3.8M.
> **Inference:** The model estimates a promo ROI of 0.80 (interval 0.3 to
> 1.3) on the high-media days and 0.17 on the low-media days. Media spend is
> a CATE covariate of this run with strong evidence (p=0.004), so the model
> itself lets the effect rise with media. Had it not been a CATE covariate,
> the gap would only reflect the mix of modeled covariates and intensity in
> each bin and would be reported as descriptive.
> **Hypothesis:** Media builds demand that the promotion converts. To verify,
> run the promotion in matched markets with and without media.
