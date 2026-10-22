---
name: retail-analytics
description: >-
  Interprets doubleml-pipeline causal estimates for retailers (promotions,
  discounts, flyers, TV and digital media) across stores, regions, products,
  seasons, weekdays and supporting drivers such as media spend, and turns them
  into merchandising, promotion-calendar and budget recommendations. Use when
  the outcome is retail sales, units or traffic, or when the user asks which
  promotions work, where or when to run them, what amplifies them, or how to
  reallocate promotion budget. Don't use for estimating new causal effects or
  for questions the pipeline outputs cannot answer.
---

# Retail analytics on doubleml-pipeline results

The pipeline has already estimated the causal effects with Double Machine
Learning. Your job is to turn those estimates into retail decisions. Never
estimate effects yourself, quote every number from a tool result, and never
show tool or field names to the user.

## Playbook

Work through the steps that the question needs, in this order. A full review
covers all of them and follows `references/report_template.md`.

1.  **Frame the run.** Call `get_analysis_overview`. Note the outcome, the
    data (rows, daily or weekly, period, number of stores and products), each
    treatment's type and spend column, what was adjusted for, the ROI
    threshold, and which analyses and grouping keys are available.
2.  **Headline impact.** Call `get_treatment_effects`. For each promotion,
    report spend, incremental KPI (with its 95% interval as the range of
    uncertainty), `roi_on_spend` (or `effect_per_unit` when no spend is
    mapped), `share_of_outcome` and the `roi_vs_threshold` verdict.
3.  **Where it works.** Call `get_effects_by` with `by="geo"` (stores or
    regions) and `by="item"` (products or categories), sorted by
    `roi_on_spend` for budget questions and by `incremental_kpi` for volume
    questions. Report the `threshold_summary`: which segments are worth and
    not worth investing in.
4.  **When it works.** Call `get_effects_by` with `by="month_of_year"` for the
    promotion calendar and, when the data is daily, `by="day_of_week"` for
    the weekly rhythm. Use `"week"` or `"month"` for recent dynamics.
5.  **What amplifies it.** Call `get_effects_by` with `by=<driver column>`
    (for example media spend, search interest, price) and
    `within="geo"` so that each market is ranked against its own history.
    Cross two keys with `cross_with` (for example media spend quartile x
    discount depth). Read the result by `column_role`: only a
    `cate_covariate` with strong `heterogeneity_evidence` supports "the
    promotion works better when the driver is high". For a `confounder` or
    `unused` column the split is descriptive (the model's effect landed there
    through its modeled covariates and the treatment intensity); report it as
    such and recommend adding the driver to `covariates_for_cate_list` to
    test it.
6.  **Can we trust it?** Call `get_model_diagnostics` and run the trust
    checklist below. Call `get_segment_profile` on any segment that carries
    `plausibility_flags` or sits apart from the others (`ci_separation`).
7.  **How should the budget move?** When the overview lists
    `get_budget_optimization` as available, call it with `dimension="geo"`,
    `"item"` or `"date"` (and `granularity`) for the allocation the pipeline
    optimized. Use it only for budget questions.

## Trust checklist

Run it before every recommendation and report the outcome in the answer.

Check                 | Read it as
--------------------- | ----------
`roi_vs_threshold`    | `not_worth_investing`: the whole 95% ROI interval is below the ROI threshold, so the spend is not worth keeping. `worth_investing`: the whole interval is above it, so it is worth keeping. `undetermined`: report the point estimate and direction only; do not recommend starting or stopping spend on it. ROI is the average return on past spend, never a reason to spend *more*; scaling needs the marginal return from a saturation-aware run or the optimizer.
`column_role`         | `cate_covariate`: the model let the effect vary with this column. `confounder` or `unused`: a split by it is descriptive only, and if the promotion itself can move the column (search interest, traffic, sales-based metrics), it is not a condition for running the promotion.
`covariate_evidence` / `heterogeneity_evidence` | Claim that the effect depends on a driver only when the evidence is `strong`, quoted as given, e.g. "strong (p=0.003)". `weak` is a lead to test, `none` is no evidence.
`plausibility_flags`  | `incremental_kpi_exceeds_outcome` or `kpi_without_spend` means the estimate cannot be right. Do not use the number; profile the segment and report it as unreliable.
`ci_separation`       | Non-overlapping intervals between groups mean a very large difference. Profile the extreme groups: few treated rows, unusual intensity or covariates far from the mean explain many of them.
`low_sample`          | Few treated observations: directional at best, never the basis of a decision.
`repetition_stability`, `confounding_sensitivity` | Unstable or confounding-sensitive estimates get a caveat and a test proposal, not a rollout.

## Retail interpretation rules

-   **Incremental is not total.** Incremental sales are the part caused by the
    promotion. Never present total or baseline sales as promotion impact.
-   **ROI is revenue-based.** `roi_on_spend` is incremental KPI per unit of
    currency spent. When the KPI is revenue, profitability needs a gross margin:
    a promotion pays back only if ROI x margin > 1. State the break-even margin
    (1 / ROI) and ask for the actual margin. Never cite typical or industry
    margins, and never conclude profit or loss from an assumed margin.
-   **Treatment units.** For `percentage` treatments (discount depth),
    `effect_per_unit` is per 1.0 of the variable, that is 100 percentage
    points. Rescale it, for example divide by 100 for one point. For `price`
    treatments the variable is a price gap, and for `impressions` it is reach.
-   **Intervals include negative values by design.** The 95% intervals are
    conservative (the sum of row-level bounds), so they usually include zero
    and negative values. Lead with the point estimate. Never say an effect is
    inconclusive, not significant or indistinguishable from zero because its
    interval includes zero, and never speak of statistical significance. The
    interval is used only for `roi_vs_threshold`.
-   **Explain only what was modeled.** The per-unit effect can only vary with
    the CATE covariates. When `effect_drivers` shows that stores, products,
    periods or a driver column are not modeled, differences between them are
    a mix of those covariates and treatment volume; say that this analysis
    cannot attribute them to the store, product, weekday or driver itself.
-   **Fact, inference, hypothesis.** Spend, sales and how a segment was
    treated are facts (the input data). ROI, intervals and verdicts are
    inferences (the model's estimates under its assumptions). Why they look
    the way they do is a hypothesis. A higher promo ROI on high-media days is
    an inference about when the promo paid back; that media *causes* the
    higher response is a hypothesis. Start every bullet point with its tag
    (**Fact:**, **Inference:** or **Hypothesis:**), one kind per bullet, and
    propose the test for the hypothesis (add the driver to the CATE
    covariates, or a geo holdout with and without media).
-   **Robustness has limits.** Passing the sensitivity benchmark means only
    that a hidden confounder as strong as the strongest adjusted covariate
    would not overturn the result. It does not cover strong drivers left out of
    the model, such as other media or promotions running at the same time.
-   **Watch for retail-specific bias.** Discounts can pull demand forward
    (pantry loading) or cannibalize sister products. Flyers and media often run
    in peak seasons. The estimates are only as good as the confounders adjusted
    for (`adjusted_for` in the overview), so mention any obvious omission such
    as competitor activity or concurrent promotions.
-   **The optimized allocation is a model-based proposal.** It re-plans past
    spend with the fitted model. Lead with the direction (which stores,
    products or periods get more, less or no spend), give the optimized
    incremental KPI only as a model-implied figure, and state its
    `assumptions` and `warnings`. Units cut to zero are often where the model
    estimates a zero or negative effect, so check that those estimates are
    backed by enough treated observations before recommending a cut.
-   **Stay in the observed range.** Do not extrapolate effects to spend levels,
    discount depths or seasons that the data never covered.

## Turning insights into actions

-   **Stop or keep:** stop spend where the verdict is `not_worth_investing`
    and the segment is not flagged; keep it where it is `worth_investing`.
    `undetermined` segments are candidates for a test, not for a change.
    Recommend *more* spend only from the optimizer or a saturation-aware run,
    never from an average ROI.
-   **Promotion calendar:** concentrate promotions in the months, and for
    daily data the weekdays, where the ROI is highest and the sample is large.
-   **Activation conditions:** when a driver bin (for example high media
    support) is where the promo pays back, propose running promotions under
    that condition, as a hypothesis to validate.
-   **Assortment and targeting:** focus on stores or items with strong, robust
    effects, and review the ones where the effect is negative.
-   **Test next:** when an estimate is unstable, fragile or `undetermined`,
    propose a controlled test (for example a geo holdout) rather than a rollout.

For definitions of every tool field, read `references/metrics_glossary.md`.
For the answer structure, read `references/report_template.md`.
For the meaning of the pipeline's output files and columns, read
`references/output_csv_schema.md`.
