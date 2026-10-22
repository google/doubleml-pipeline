# Metrics glossary

Fields returned by the doubleml-pipeline analysis tools.

## Effect metrics

These appear in `get_treatment_effects` and in every group or cell of
`get_effects_by`.

Field                                    | Meaning
---------------------------------------- | -------
`incremental_kpi`                        | Outcome caused by the treatment: the estimated outcome minus the outcome without the treatment.
`incremental_kpi_low` / `_high`          | 95% interval: the sum of the row-level bounds. It is conservative, so it is wider than an exact interval for the aggregate and is expected to include negative values. It is a range of uncertainty, not a significance test: an interval that includes zero does not mean there is no effect.
`effect_direction`                       | Sign of the point estimate `incremental_kpi`: `positive`, `negative` or `zero`.
`treatment_input`                        | Sum of the treatment variable, in its own unit (spend, discount rate, impressions and so on).
`effect_per_unit` (`_low` / `_high`)     | `incremental_kpi / treatment_input`: the treatment-weighted average effect per unit of the treatment, with its 95% interval. Null where a signed treatment (price gaps) has both signs in the group. Under a `saturation_family` it falls with intensity by construction.
`spend`, `roi_on_spend` (`_low` / `_high`) | Mapped spend and `incremental_kpi / spend` (return per unit of currency, for example ROAS for revenue KPIs), with its 95% interval.
`share_of_outcome`                       | `incremental_kpi / outcome`: the share of the outcome that the treatment caused.
`share_of_total_incremental_kpi`         | The treatment's share of the incremental KPI of all treatments.
`observations` / `treated_observations`  | Rows in the group, and rows where the treatment was non-zero.

Lagged treatments (`is_lagged`, names ending in `_l<k>`) are carry-over
effects of the base treatment k periods later.

## Reliability signals

These accompany every effect row.

Field                 | Meaning
--------------------- | -------
`roi_vs_threshold`    | The ROI interval compared with the run's ROI threshold (`threshold_roi`, from `opt_threshold_roi`). `not_worth_investing`: `roi_on_spend_high` is below the threshold, so even the optimistic case does not pay back. `worth_investing`: `roi_on_spend_low` is above it. `undetermined`: the interval straddles the threshold. Absent without a spend column or a configured threshold. With many groups, expect a few verdicts to be wrong by chance.
`low_sample`          | Fewer than 30 treated rows (a heuristic; the interval carries the precision). Directional at best.
`plausibility_flags`  | `incremental_kpi_exceeds_outcome`: a positive effect larger than the outcome itself, which is impossible. `kpi_without_spend`: a spend treatment with an effect where nothing was spent. `negative_spend`. Any flag means the estimate should not be used; profile the segment instead.
`threshold_summary`   | In `get_effects_by`: the groups per verdict.
`ci_separation`       | In `get_effects_by`: pairs of groups whose per-unit intervals do not overlap (`separated_pairs`) and the pair with the largest gap. A sign of a very large difference that deserves a `get_segment_profile` check.
`low_sample_groups`   | In `get_effects_by`: the groups flagged `low_sample`.

## Grouping (`get_effects_by`)

Field                       | Meaning
--------------------------- | -------
`grouping.kind`, `grouping.binning` | `segment` (store, region, product), `period` (calendar) or `bin` (an input column, binned by `quantiles`, kept as `distinct_values` when it has few, or pooled into `categories` with `(other)`).
`grouping.cut_points`       | For numeric columns: the value range of each quantile bin `Q1..Qn`. With `within`, one set of ranges per segment: `Q4` is the top quartile *of that segment's own history*.
`data_period`, `filter`     | First and last date in the data, and the `start` / `end` window applied to this call, if any.
`window`                    | The page shown: `page` of `pages`, `groups_total`, `groups_shown`, and the `first` and `last` group on the page. Summaries (`threshold_summary`, `ci_separation`, `spread`, ...) always cover all groups in the window, not only the page.
`spread`                    | Min, median and max of the per-unit metric across all groups.
`highest_period` / `lowest_period` | For periods: the period with the largest and smallest incremental KPI.
`effect_drivers`            | The CATE covariates per treatment and whether the grouping column is one of them (`dimension_modeled`). If not, per-unit differences across groups reflect only their mix of those covariates and treatment volume; the analysis cannot attribute them to the group itself.
`column_role`               | How the model used the grouping column: `cate_covariate` (the effect was allowed to vary with it), `confounder` (adjusted for only) or `unused`. Splits by the latter two are descriptive.
`heterogeneity_evidence`    | When the grouping column is a CATE covariate: `strong (p=...)`, `weak (p=...)` or `none (p=...)`, the Bonferroni-adjusted p-value of its CATE terms in the pooled fit. It ignores first-stage estimation error, so read it as a grade. Only `strong` supports saying that the effect depends on it.
`cells`, `row_groups`, `col_groups` | For `cross_with`: one row per combination of the two keys, in presentation order.

## Model diagnostics (`get_model_diagnostics`)

Field                            | Meaning
-------------------------------- | -------
`share_positive`                 | Share of the cross-fitting repetitions with a positive total effect. Values near 0 or 1 mean a stable sign.
`coefficient_of_variation`       | Standard deviation divided by the mean of the total effect across repetitions. Above about 0.3 means unstable.
`robustness_value`               | Partial R2 that an unobserved confounder would need with both the treatment and the outcome to explain the effect away. Higher is more robust.
`benchmark_cf_d` / `_cf_y`       | Partial R2 of the strongest observed covariate with the treatment and the outcome.
`robust_to_benchmark_confounder` | True if a hidden confounder as strong as that covariate could not explain the effect away. It says nothing about stronger drivers that were not adjusted for.
`cate_model.structure`, `spline_df`, `spline_degree` | How the per-unit effect was allowed to vary with the CATE covariates.
`cate_model.covariate_evidence`  | Per CATE covariate: `strong (p=...)`, `weak (p=...)` or `none (p=...)`.
`cate_model.terms`               | The basis terms with coefficient, standard error and p-value, for readers who want the fitted equation.
`cate_model.effect_per_unit_distribution` | Quantiles of the row-level per-unit effect over treated rows and the share of treated rows with a negative effect.

## Segment profile (`get_segment_profile`)

Field                        | Meaning
---------------------------- | -------
`share_of_observations`, `outcome.share_of_total`, `treatments[].share_of_spend` | How big the segment is in rows, outcome and spend.
`treatments[].treated_share`, `mean_when_treated`, `p25/p50/p75_when_treated` | How often and how intensely the segment was treated, next to the overall figures.
`covariates[].ratio_to_overall`, `gap_in_row_sd` | Segment mean over overall mean, and the gap in row-level standard deviations of the column (a scale-free size of the difference).

## Budget optimization

These appear in `get_budget_optimization` (runs with `optimize=True`). An
allocation unit is one geo x item x date row of the optimization period.

Field                                         | Meaning
--------------------------------------------- | -------
`settings.optimizer`                          | `simple`: rows with ROI below `threshold_roi` lose all their spend, and the freed budget is not reallocated. `saturation`: the budget is redistributed along the estimated diminishing-returns curve. `linear_fallback`: saturation was configured but not identified, so the simple rule was used.
`spend_before` / `spend_after`                | Actual and optimized spend.
`incremental_kpi_before` / `_after`           | Estimated incremental KPI at the actual spend, and model-implied incremental KPI at the optimized spend. The latter is not observed and has no interval.
`roi_before` / `roi_after`                    | Incremental KPI per unit of spend before and after.
`spend_change`, `spend_change_pct`            | Optimized minus actual spend, and its share of actual spend.
`units_spend_increased` / `_reduced` / `_cut_to_zero` | Allocation units whose spend goes up, goes down but stays positive, or goes to zero.
`assumptions`                                 | What the optimized allocation assumes. State them with any recommendation.
`warnings`                                    | Results that need care, for example a model-implied gain far above the current estimate.
`optimizer_statistics`                        | Saturation-aware optimizer only: budget `K`, spend allocated `K_out`, `unspent_budget`, `marginal_roi_at_optimum` and counts of units excluded for a non-positive effect or capped.
