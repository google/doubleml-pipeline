# doubleml-pipeline slides

The original slide deck of the user guide, one image per slide, in
presentation order. The searchable, maintained version of this content is the
[User Guide](guide.md); use this page when you need the pictures for a talk.

> [!WARNING]
> The deck predates several features and a few names have changed since it
> was made. Where the slides and the [User Guide](guide.md) disagree, the guide
> and the [API Reference](api_reference.md) are right. In particular:
>
> *   `n_jobs=-1` uses all **CPU** cores; the pipeline does not use a GPU.
> *   `treatment_types` accepts `"spend"` (default), `"percentage"`,
>     `"impressions"` and `"price"`.
> *   `simple_optimization` defaults to `None` and a saturation-aware
>     optimizer exists (see [Saturation DML](saturation.md)); the default
>     `opt_threshold_roi` is `1.0`.
> *   Phase 0 (saturation) and the HTML report did not exist
>     yet; the Phase 4 folder is `step4_consolidated_results/` and the ROI
>     folders are `003_roi_by_month`, `005_roi_by_item` and
>     `007_optimization`.
> *   The Cluster DML keys are chosen automatically from `item_col`,
>     `geo_col` and a date block (see
>     [Advanced configuration](advanced.md#cluster-dml-keys-chosen-automatically-every-run)).
> *   Parameter names on the slides with typos read `standardize_cols` and
>     `population_median_normalize_cols`.

## Overview and inputs

![Overview of doubleml-pipeline](../img/guide_01.png)

![Input of the pipeline](../img/guide_02.png)

## 1. Data granularity

![Data granularity: code setup](../img/guide_03.png)

![Setting date, population, geo and item columns](../img/guide_04.png)

![How to create lag variables when necessary](../img/guide_05.png)

## 2. Variable selection

![Variable selection: code setup](../img/guide_06.png)

![Variable selection example with 28 confounders and their VIF](../img/guide_07.png)

![Impact of excessive predictors on the estimate](../img/guide_39.png)

![Multicollinearity reduction: dropping the high-VIF columns](../img/guide_08.png)

![Unstable estimation after the VIF reduction](../img/guide_41.png)

![Reduced predictor setup](../img/guide_42.png)

![Successful estimation result](../img/guide_40.png)

![Confounders versus predictors in the causal graph](../img/guide_09.png)

![Identifying the back-door paths](../img/guide_10.png)

## 3. Normalization

![Normalization: code setup](../img/guide_11.png)

![Guideline on normalization in doubleml-pipeline](../img/guide_12.png)

![Variable normalization assignment](../img/guide_13.png)

## 4. Covariates for CATE and 5. CATE structure

![Objective of selecting covariates X for CATE estimation](../img/guide_14.png)

![Concept of B-spline regression](../img/guide_15.png)

![Notes 1: number of covariates](../img/guide_16.png)

![Notes 2: number of knots](../img/guide_17.png)

![Notes 3: degree of the basis functions](../img/guide_18.png)

![Notes 4: additive versus multiplicative form](../img/guide_19.png)

![Default CATE settings](../img/guide_20.png)

![Custom CATE settings](../img/guide_21.png)

## 6. ML settings

![Selection of the Y-model and the T-model](../img/guide_22.png)

![n_folds setting](../img/guide_23.png)

![Time budget and number of cores](../img/guide_24.png)

## 7. Ground truth

![Ground truth setup](../img/guide_25.png)

![Ground truth comparison plot](../img/guide_26.png)

## 8. Sensitivity analysis

![Sensitivity analysis setup](../img/guide_27.png)

## 9. Calculation of ROI

![Calculation of ROI: setup](../img/guide_28.png)

![ROI output directory structure](../img/guide_29.png)

![Sample ROI visualizations](../img/guide_30.png)

## 10. Optimization

![Optimization setup](../img/guide_31.png)

![Example of optimization in doubleml-pipeline](../img/guide_32.png)

![Optimization detailed output](../img/guide_33.png)

## Validation

![Validation points after causal ML modeling](../img/guide_34.png)

![Evaluation of heterogeneity](../img/guide_35.png)

![Evaluation of positivity: KDE heat map](../img/guide_36.png)

![Evaluation of positivity: confidence-interval spike](../img/guide_37.png)

## Checklist

![DML checklist](../img/guide_38.png)
