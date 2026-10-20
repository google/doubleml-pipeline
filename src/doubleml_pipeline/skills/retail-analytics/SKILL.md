---
name: retail-analytics
description: >-
  Interprets doubleml-pipeline causal estimates for retailers (promotions,
  discounts, flyers, TV and digital media) across stores, regions, products and
  seasons, and turns them into merchandising, promotion-calendar and budget
  recommendations. Use when the outcome is retail sales, units or traffic, or
  when the user asks which promotions work, where or when to run them, or how
  to reallocate promotion budget. Don't use for estimating new causal effects
  or for questions the pipeline outputs cannot answer.
---

# Retail analytics on doubleml-pipeline results

The pipeline has already estimated the causal effects with Double Machine
Learning. Your job is to turn those estimates into retail decisions. Never
estimate effects yourself, quote every number from a tool result, and never
show tool or field names to the user.

## Playbook

Work through the steps that the question needs, in this order.

1.  **Frame the run.** Call `get_analysis_overview`. Note the outcome (for
    example sales or units), each treatment's type and spend column, what was
    adjusted for, and which analyses are available.
2.  **Headline impact.** Call `get_treatment_effects`. For each promotion,
    report the incremental KPI (with its 95% interval as the range of
    uncertainty), `roi_on_spend` (or `effect_per_unit` when no spend is mapped) and
    `share_of_outcome`.
3.  **Where it works.** Call `get_segment_effects` with `dimension="geo"`
    (stores or regions) and `dimension="item"` (products or categories). Rank
    by `roi_on_spend` for budget questions and by `incremental_kpi` for volume
    questions.
4.  **When it works.** Call `get_effect_trend` with
    `granularity="month_of_year"` for the seasonal promotion calendar, or with
    `"week"` or `"month"` for recent dynamics.
5.  **Why it varies.** Call `get_effect_heterogeneity` to see which drivers
    (for example demand level, holidays or price) change the per-unit effect.
6.  **Can we trust it?** Call `get_robustness_checks` before recommending any
    action.
7.  **How should the budget move?** When the overview lists
    `get_budget_optimization` as available, call it with `dimension="geo"`,
    `"item"` or `"date"` (and `granularity`) for the allocation the pipeline
    optimized. Use it only for budget questions.

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
    interval includes zero. Judge reliability from repetition stability,
    sensitivity to confounding and the number of treated observations.
-   **Low support means low reliability.** Treat segments, bins or periods with
    few `treated_observations` as directional at best, even when their point
    estimate is extreme. Prefer insights about timing or drivers that are
    backed by many treated observations.
-   **Heterogeneity needs evidence.** Claim that the effect depends on a driver
    only when its `heterogeneity_pvalue` is below 0.05 and the bins tell a
    consistent story.
-   **Explain only what was modeled.** The per-unit effect can only vary with
    the CATE covariates. When `effect_drivers` shows that stores, products or
    periods are not modeled, say that this analysis cannot tell whether the
    effect differs between them beyond those drivers. Do not call a narrow or
    wide spread a store or product effect.
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

-   **Reallocate:** shift spend from segments or periods with low ROI to those
    with high, stable ROI. When the run was optimized, base the proposal on the
    optimized allocation instead of your own split.
-   **Promotion calendar:** concentrate promotions in the months or demand
    conditions where the per-unit effect is highest.
-   **Assortment and targeting:** focus on stores or items with strong, robust
    effects, and review the ones where the effect is negative.
-   **Test next:** when an estimate is unstable or fragile, propose a
    controlled test (for example a geo holdout) rather than a rollout.

For definitions of every tool field, read `references/metrics_glossary.md`.
For the answer structure, read `references/report_template.md`.
