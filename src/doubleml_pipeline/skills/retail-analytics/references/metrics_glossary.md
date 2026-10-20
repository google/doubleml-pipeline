# Metrics glossary

Fields returned by the doubleml-pipeline analysis tools.

## Effect metrics

These appear in `get_treatment_effects`, `get_segment_effects`,
`get_effect_trend` and `get_effect_heterogeneity`.

Field                                    | Meaning
---------------------------------------- | -------
`incremental_kpi`                        | Outcome caused by the treatment: the estimated outcome minus the outcome without the treatment.
`incremental_kpi_low` / `_high`          | 95% interval: the sum of the row-level bounds. It is conservative, so it is wider than an exact interval for the aggregate and is expected to include negative values. It is a range of uncertainty, not a significance test: an interval that includes zero does not mean there is no effect.
`effect_direction`                       | Sign of the point estimate `incremental_kpi`: `positive`, `negative` or `zero`.
`treatment_input`                        | Sum of the treatment variable, in its own unit (spend, discount rate, impressions and so on).
`effect_per_unit` (`_low` / `_high`)     | `incremental_kpi / treatment_input`: the average per-unit effect (CATE), weighted by treatment, with its 95% interval.
`spend`, `roi_on_spend` (`_low` / `_high`) | Mapped spend and `incremental_kpi / spend` (return per unit of currency, for example ROAS for revenue KPIs), with its 95% interval.
`share_of_outcome`                       | `incremental_kpi / outcome`: the share of the outcome that the treatment caused.
`share_of_total_incremental_kpi`         | The treatment's share of the incremental KPI of all treatments.
`observations` / `treated_observations`  | Rows in the group, and rows where the treatment was non-zero. Few treated rows mean low reliability.

Lagged treatments (`is_lagged`, names ending in `_l<k>`) are carry-over
effects of the base treatment k periods later.

## Robustness metrics

These appear in `get_robustness_checks`.

Field                            | Meaning
-------------------------------- | -------
`share_positive`                 | Share of the cross-fitting repetitions with a positive total effect. Values near 0 or 1 mean a stable sign.
`coefficient_of_variation`       | Standard deviation divided by the mean of the total effect across repetitions. Above about 0.3 means unstable.
`robustness_value`               | Partial R2 that an unobserved confounder would need with both the treatment and the outcome to explain the effect away. Higher is more robust.
`benchmark_cf_d` / `_cf_y`       | Partial R2 of the strongest observed covariate with the treatment and the outcome.
`robust_to_benchmark_confounder` | True if a hidden confounder as strong as that covariate could not explain the effect away. It says nothing about stronger drivers that were not adjusted for.

## What the model can explain

`effect_drivers` (in segment and trend results) lists the CATE covariates per
treatment. The per-unit effect varies only with them. If `dimension_modeled` is
false, differences in `effect_per_unit` or `roi_on_spend` across stores,
products or periods only reflect their mix of those covariates (or nothing, for
a constant effect). Any other difference is not explained by this analysis.

## Heterogeneity

`heterogeneity_pvalue` is the Bonferroni-adjusted p-value of the CATE basis
terms of a covariate. Below 0.05 means the per-unit effect significantly
changes with the covariate. Read the direction and shape from the bins.
